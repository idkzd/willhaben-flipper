"""Send evaluated adverts to Telegram."""
import html
from typing import Any

import requests

TELEGRAM_API = "https://api.telegram.org"


def _escape(text: str) -> str:
    return html.escape(str(text), quote=False)


def _build_caption(advert: dict[str, Any], evaluation: dict[str, Any]) -> str:
    title = _escape(advert.get("title", "") or "Без назви")
    location = _escape(advert.get("location", "") or advert.get("state", ""))
    price = _escape(evaluation.get("estimated_price_eur", "?"))
    score = evaluation.get("resale_score", "?")
    potential = _escape(evaluation.get("resale_potential", ""))
    reasoning = _escape(evaluation.get("reasoning", ""))

    message_to_seller = _escape(evaluation.get("message_to_seller", "")).strip()

    lines = [
        f"🟢 <b>{title}</b>",
        f"📍 {location}",
        f"💰 Оцінка перепродажу: <b>{score} з 10</b> ({potential}) — орієнтовно <b>{price} €</b>",
    ]
    if reasoning:
        lines.append(f"ℹ️ Чому вигідно: <i>{reasoning}</i>")
    if message_to_seller:
        lines.append("✉️ <b>Продавцю (нім.):</b>")
        lines.append(f"<pre>{message_to_seller}</pre>")
    lines.append(f'🔗 <a href="{_escape(advert.get("url", ""))}">Відкрити оголошення</a>')
    return "\n".join(lines)


def send(
    advert: dict[str, Any],
    evaluation: dict[str, Any],
    bot_token: str,
    chat_id: str,
    timeout: int = 30,
) -> bool:
    """Send the advert as a photo (with caption) or plain text fallback."""
    caption = _build_caption(advert, evaluation)
    images = advert.get("images") or []

    # Try photo first (looks much better in Telegram).
    if images:
        try:
            resp = requests.post(
                f"{TELEGRAM_API}/bot{bot_token}/sendPhoto",
                json={
                    "chat_id": chat_id,
                    "photo": images[0],
                    "caption": caption,
                    "parse_mode": "HTML",
                },
                timeout=timeout,
            )
            if resp.json().get("ok"):
                return True
        except requests.RequestException:
            pass  # fall through to plain text

    resp = requests.post(
        f"{TELEGRAM_API}/bot{bot_token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": caption,
            "parse_mode": "HTML",
            "disable_web_page_preview": False,
        },
        timeout=timeout,
    )
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram error: {data}")
    return True
