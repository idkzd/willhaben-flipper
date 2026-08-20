"""Evaluate resale potential of a free-item advert via OpenRouter LLM.

The preferred model (``dots-studio/dots-3-note-preview:free``) is multimodal,
so it also receives the advert's first photo for a more accurate call. If it
is rate-limited/unavailable, we fall back to the text-only models in the list
(no photo for those).

Free OpenRouter models are frequently rate-limited (HTTP 429), and each
OpenRouter account also has its own daily free-tier request quota. To work
around both, requests are spread across a pool of API keys (see
``config.get_list("OPENROUTER_API_KEYS")``) via :class:`KeyPool`, which
round-robins keys and parks any key that hits its daily limit until it
resets. Each request also retries with exponential backoff and falls back
to the next model.
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
DEFAULT_EXHAUSTION_COOLDOWN = 3600.0  # fallback if the API doesn't send a reset time

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


class KeyPool:
    """Round-robins across multiple OpenRouter API keys.

    Each key has its own daily free-tier quota, so spreading requests across
    several accounts' keys multiplies the effective daily allowance. A key
    that comes back HTTP 429 for the free-tier daily limit is parked until
    its reported reset time (or a 1h cooldown if none is given) and skipped
    until then.
    """

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("KeyPool needs at least one API key")
        self._keys = list(dict.fromkeys(keys))  # de-dupe, keep order
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
        """Return the next available key (round robin), or None if all are
        currently exhausted."""
        with self._lock:
            n = len(self._keys)
            for _ in range(n):
                key = self._keys[self._index]
                self._index = (self._index + 1) % n
                if self._available(key):
                    return key
            return None


def _parse_reset_epoch(resp: requests.Response) -> float | None:
    """Best-effort extraction of the rate-limit reset time (epoch seconds)."""
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
        return float(header_val) / 1000.0  # OpenRouter sends milliseconds
    except (TypeError, ValueError):
        return None


def _parse_content(content: str) -> dict[str, Any]:
    """Extract a JSON object from the model reply (tolerant of fences)."""
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
    """Download the advert photo and return it as a base64 data URL."""
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
    """Text-only message, or text + image for multimodal models."""
    if not image_data_url:
        return user_prompt
    return [
        {"type": "text", "text": user_prompt},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]


def _try_model(
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

        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {key}",
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

        if resp.status_code == 200:
            payload = resp.json()
            choices = payload.get("choices") or []
            content = choices[0].get("message", {}).get("content") if choices else None
            try:
                return _parse_content(content)
            except (ValueError, json.JSONDecodeError) as exc:
                last_detail = f"bad reply: {exc}"
        elif resp.status_code == 429:
            reset_at = _parse_reset_epoch(resp)
            key_pool.mark_exhausted(key, reset_at)
            last_detail = f"HTTP 429 ({KeyPool.label(key)}): {resp.text[:200]}"
            continue  # switch to the next key right away, no need to back off
        elif resp.status_code in RETRYABLE_STATUS:
            last_detail = f"HTTP {resp.status_code}: {resp.text[:300]}"
        else:
            resp.raise_for_status()

        if attempt < attempts - 1:
            time.sleep(BACKOFF_BASE ** attempt + random.uniform(0, 0.5))

    raise RuntimeError(
        f"Model {model} failed after {attempts} tries ({last_detail})"
    )


def _fallback_seller_message(title: str) -> str:
    """Short, informal German message in the style the user requested."""
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

    ``models`` is a comma-separated list tried in order (first success wins).
    Models listed in the ``VISION_MODELS`` env var receive the first photo.
    """
    user_prompt = (
        f"Назва: {advert.get('title', '')}\n"
        f"Опис: {advert.get('body', '')[:800]}\n"
        f"Локація: {advert.get('location', '')} ({advert.get('postcode', '')})\n"
        f"Ціна: {advert.get('price', '0')} EUR\n"
    )

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        model_list = ["dots-studio/dots-3-note-preview:free"]

    vision_models = os.environ.get("VISION_MODELS", DEFAULT_VISION_MODELS)
    vision_set = {m.strip() for m in vision_models.split(",") if m.strip()}

    # Download the first photo once — only if at least one vision model will run.
    image_data_url: str | None = None
    images = advert.get("images") or []
    if any(m in vision_set for m in model_list) and images:
        image_data_url = _fetch_image_data_url(images[0])

    last_error: Exception | None = None
    for model in model_list:
        try:
            content = _build_user_content(
                user_prompt, image_data_url if model in vision_set else None
            )
            result = _try_model(model, key_pool, content, timeout)
            if not (result.get("message_to_seller") or "").strip():
                result["message_to_seller"] = _fallback_seller_message(
                    advert.get("title", "")
                )
            return result
        except Exception as exc:  # noqa: BLE001 - try the next model
            last_error = exc
    raise RuntimeError(f"All models failed: {last_error}")
