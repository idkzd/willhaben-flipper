"""Evaluate resale potential of a free-item advert via LLM providers.

Primary provider: OpenRouter (free-tier multimodal models with KeyPool
round-robin for multiple accounts).

Fallback provider: Any OpenAI-compatible endpoint configured via
``SECOND_PROVIDER_BASE_URL / API_KEY / MODEL`` env vars.  Currently used
with TokenHarbor (mimo-v2.5:free) which is also vision-capable.
"""
import base64
import json
import os
import random
import re
import threading
import time
from typing import Any

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRIES_PER_MODEL = 4
BACKOFF_BASE = 2.0
RETRYABLE_STATUS = (429, 500, 502, 503, 504)
DEFAULT_EXHAUSTION_COOLDOWN = 3600.0

DEFAULT_VISION_MODELS = "dots-studio/dots-3-note-preview:free"

SYSTEM_PROMPT = (
    "Ти — експерт з перепродажу (фліпінгу) речей на австрійському маркетплейсі "
    "willhaben.at. Тобі дають оголошення з категорії «Zu verschenken» (віддають "
    "безкоштовно). Оціни, наскільки реально й вигідно забрати річ безкоштовно і "
    "швидко перепродати її у Відні. Якщо до повідомлення прикріплено фото, "
    "обов'язково врахуй його для оцінки стану речі.\n\n"
    "Відповідай ТІЛЬКИ одним JSON-об'єктом без markdown-розмітки, за такою схемою:\n"
    '{"resale_score": 0-10, "resale_potential": "very_good|good|average|poor", '
    '"estimated_price_eur": "діапазон, напр. 20-40", "effort_to_flip": '
    '"low|medium|high", '
    '"reasoning": "українською: 1-2 речення, чому цю річ легко/вигідно перепродати", '
    '"message_to_seller": "німецькою: коротке неформальне дружнє повідомлення '
    'продавцю (1-2 речення) з проханням забрати річ безкоштовно і домовитися про час"}\n\n'
    "resale_score — наскільки легко/вигідно перепродати (10 = майже гарантований "
    "швидкий і вигідний перепродаж). resale_potential=very_good лише якщо річ "
    "ліквідна, затребувана, в робочому/гарному стані й її реально продати за "
    "помітні гроші. Зламане, громіздке, вузькоспеціалізоване сміття = poor. "
    "message_to_seller має бути коротким, неформальним, природним, людяним, "
    "німецькою. Стиль як у прикладі: 'Hallo! Ist die Lampe noch verfügbar? Ich "
    "würde sie gerne nehmen 😊' — тільки замість Lampe підстав назву речі з "
    "оголошення з правильним артиклем і займенником (die Lampe → sie, der Tisch "
    "→ ihn, das Sofa → es). Жодних офіційних формулювань, жодних 'Sehr geehrte'."
)


# ---------------------------------------------------------------------------
# KeyPool (OpenRouter multi-key round-robin)
# ---------------------------------------------------------------------------

class KeyPool:
    """Round-robins across multiple OpenRouter API keys."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("KeyPool needs at least one API key")
        self._keys = list(dict.fromkeys(keys))
        self._index = 0
        self._exhausted_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def size(self) -> int:
        return len(self._keys)

    @staticmethod
    def label(key: str) -> str:
        return f"...{key[-4:]}" if len(key) > 4 else "key"

    def _available(self, key: str) -> bool:
        until = self._exhausted_until.get(key)
        return until is None or time.time() >= until

    def mark_exhausted(self, key: str, reset_at: float | None) -> None:
        with self._lock:
            self._exhausted_until[key] = reset_at or (
                time.time() + DEFAULT_EXHAUSTION_COOLDOWN
            )
            remaining = sum(1 for k in self._keys if self._available(k))
        print(
            f"🔑 Ключ {self.label(key)} вичерпано (rate limit). "
            f"Доступно ще {remaining}/{len(self._keys)} ключів.",
            flush=True,
        )

    def next_key(self) -> str | None:
        with self._lock:
            n = len(self._keys)
            for _ in range(n):
                key = self._keys[self._index]
                self._index = (self._index + 1) % n
                if self._available(key):
                    return key
            return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_reset_epoch(resp: requests.Response) -> float | None:
    header_val = resp.headers.get("X-RateLimit-Reset")
    if not header_val:
        try:
            body = resp.json()
            header_val = (
                body.get("error", {})
                .get("metadata", {})
                .get("headers", {})
                .get("X-RateLimit-Reset")
            )
        except ValueError:
            header_val = None
    if not header_val:
        return None
    try:
        return float(header_val) / 1000.0
    except (TypeError, ValueError):
        return None


def _request_once(
    base_url: str,
    api_key: str,
    model: str,
    user_content: Any,
    timeout: int,
) -> requests.Response:
    """Single chat-completions request to any OpenAI-compatible endpoint."""
    return requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
        },
        timeout=timeout,
    )


def _extract_content(resp: requests.Response) -> str | None:
    payload = resp.json()
    choices = payload.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message", {}) or {}
    return message.get("content") or message.get("reasoning_content")


def _parse_content(content: str) -> dict[str, Any]:
    if not content or not content.strip():
        raise ValueError("model returned empty content")
    content = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", content, re.S)
    if fence:
        content = fence.group(1).strip()
    else:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            content = content[start : end + 1]

    data = json.loads(content)
    data.setdefault("resale_score", 0)
    data.setdefault("resale_potential", "average")
    data.setdefault("estimated_price_eur", "?")
    data.setdefault("effort_to_flip", "medium")
    data.setdefault("reasoning", "")
    data.setdefault("message_to_seller", "")
    return data


def _fetch_image_data_url(url: str, timeout: int = 20) -> str | None:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            content_type = "image/jpeg"
        payload = base64.b64encode(resp.content).decode("ascii")
        return f"data:{content_type};base64,{payload}"
    except requests.RequestException:
        return None


def _build_user_content(user_prompt: str, image_data_url: str | None) -> Any:
    if not image_data_url:
        return user_prompt
    return [
        {"type": "text", "text": user_prompt},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]


# ---------------------------------------------------------------------------
# Provider 1: OpenRouter (KeyPool round-robin)
# ---------------------------------------------------------------------------

def _try_openrouter(
    model: str,
    key_pool: KeyPool,
    user_content: Any,
    timeout: int,
) -> dict[str, Any]:
    last_detail = "unknown error"
    attempts = max(RETRIES_PER_MODEL, key_pool.size())
    for attempt in range(attempts):
        key = key_pool.next_key()
        if key is None:
            raise RuntimeError("all OpenRouter API keys are rate-limited")

        resp = _request_once(OPENROUTER_URL, key, model, user_content, timeout)

        if resp.status_code == 200:
            try:
                return _parse_content(_extract_content(resp))
            except (ValueError, json.JSONDecodeError) as exc:
                last_detail = f"bad reply: {exc}"
        elif resp.status_code == 429:
            reset_at = _parse_reset_epoch(resp)
            key_pool.mark_exhausted(key, reset_at)
            last_detail = f"HTTP 429 ({KeyPool.label(key)}): {resp.text[:200]}"
            continue
        elif resp.status_code in RETRYABLE_STATUS:
            last_detail = f"HTTP {resp.status_code}: {resp.text[:300]}"
        else:
            resp.raise_for_status()

        if attempt < attempts - 1:
            time.sleep(BACKOFF_BASE ** attempt + random.uniform(0, 0.5))

    raise RuntimeError(
        f"OpenRouter model {model} failed after {attempts} tries ({last_detail})"
    )


# ---------------------------------------------------------------------------
# Provider 2: any single-key OpenAI-compatible endpoint (fallback)
# ---------------------------------------------------------------------------

def _try_single_key(
    model: str,
    api_key: str,
    base_url: str,
    user_content: Any,
    timeout: int,
) -> dict[str, Any]:
    last_detail = "unknown error"
    for attempt in range(RETRIES_PER_MODEL):
        resp = _request_once(base_url, api_key, model, user_content, timeout)

        if resp.status_code == 200:
            try:
                return _parse_content(_extract_content(resp))
            except (ValueError, json.JSONDecodeError) as exc:
                last_detail = f"bad reply: {exc}"
        elif resp.status_code in RETRYABLE_STATUS:
            last_detail = f"HTTP {resp.status_code}: {resp.text[:300]}"
        else:
            resp.raise_for_status()

        if attempt < RETRIES_PER_MODEL - 1:
            time.sleep(BACKOFF_BASE ** attempt + random.uniform(0, 0.5))

    raise RuntimeError(
        f"Model {model} on fallback provider failed after {RETRIES_PER_MODEL} "
        f"tries ({last_detail})"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _fallback_seller_message(title: str) -> str:
    t = (title or "").strip()
    if not t:
        return "Hallo! Ist der Artikel noch verfügbar? Ich würde ihn gerne nehmen 😊"
    if len(t) > 60:
        t = t[:60].rsplit(" ", 1)[0] + "…"
    return f"Hallo! Ist „{t}“ noch verfügbar? Ich würde es gerne nehmen 😊"


def evaluate(
    advert: dict[str, Any],
    key_pool: KeyPool,
    models: str,
    timeout: int = 90,
) -> dict[str, Any]:
    """Return {resale_score, resale_potential, estimated_price_eur, ...}.

    Tries OpenRouter models first (via KeyPool), then falls back to the
    second provider configured via SECOND_PROVIDER_* env vars.
    """
    user_prompt = (
        f"Назва: {advert.get('title', '')}\n"
        f"Опис: {advert.get('body', '')[:800]}\n"
        f"Локація: {advert.get('location', '')} ({advert.get('postcode', '')})\n"
        f"Ціна: {advert.get('price', '0')} EUR\n"
    )

    # --- build ordered list of (model, is_vision) ---
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        model_list = ["dots-studio/dots-3-note-preview:free"]

    vision_models = os.environ.get("VISION_MODELS", DEFAULT_VISION_MODELS)
    vision_set = {m.strip() for m in vision_models.split(",") if m.strip()}

    # Second provider (optional fallback)
    second_models = [
        m.strip()
        for m in os.environ.get("SECOND_PROVIDER_MODEL", "").split(",")
        if m.strip()
    ]
    second_base_url = os.environ.get("SECOND_PROVIDER_BASE_URL", "").strip()
    second_api_key = os.environ.get("SECOND_PROVIDER_API_KEY", "").strip()

    # Ordered attempts: OpenRouter first, second provider after
    or_attempts = [(m, m in vision_set) for m in model_list]
    sec_attempts = [(m, m in vision_set) for m in second_models]
    all_attempts = or_attempts + sec_attempts

    # --- download the first photo once if any vision model is present ---
    image_data_url: str | None = None
    images = advert.get("images") or []
    if any(is_v for _, is_v in all_attempts) and images:
        image_data_url = _fetch_image_data_url(images[0])

    # --- try each model/provider ---
    last_error: Exception | None = None
    for model, is_vision in all_attempts:
        content = _build_user_content(
            user_prompt, image_data_url if is_vision else None
        )
        try:
            if model in {m for m, _ in sec_attempts} and second_api_key:
                # second provider — single key, no KeyPool
                result = _try_single_key(
                    model, second_api_key, second_base_url, content, timeout
                )
            else:
                result = _try_openrouter(model, key_pool, content, timeout)

            if not (result.get("message_to_seller") or "").strip():
                result["message_to_seller"] = _fallback_seller_message(
                    advert.get("title", "")
                )
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"All providers/models failed: {last_error}")
