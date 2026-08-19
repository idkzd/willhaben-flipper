"""Multi-user Telegram subscriptions via long polling.

Anyone who sends /start to the bot is added as a subscriber, and every new
match is broadcast to all subscribers. Subscribers and the last processed
``update_id`` are stored in subscribers.json (note: on Render's free tier the
filesystem is ephemeral, so this resets on redeploys/restarts).
"""
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests

import notify

TELEGRAM_API = "https://api.telegram.org"
STATE_FILE = Path(__file__).resolve().parent / "subscribers.json"

WELCOME = (
    "Привіт! 👋\n"
    "Я надсилатиму сюди безкоштовні речі з willhaben (Відень), які вигідно "
    "перепродати — з оцінкою, чому вигідно, і готовим повідомленням продавцю.\n\n"
    "Команда: /stop — відписатися"
)

HELP_TEXT = "Команди: /start — підписатися, /stop — відписатися"

_lock = threading.Lock()


def _load() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {"subscribers": [], "offset": 0}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"subscribers": [], "offset": 0}
        data.setdefault("subscribers", [])
        data.setdefault("offset", 0)
        return data
    except (ValueError, OSError):
        return {"subscribers": [], "offset": 0}


def _save(data: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def subscribers() -> list[str]:
    with _lock:
        return list(_load().get("subscribers", []))


def subscribe(chat_id: str) -> None:
    with _lock:
        data = _load()
        subs = set(data.get("subscribers", []))
        subs.add(chat_id)
        data["subscribers"] = sorted(subs)
        _save(data)


def unsubscribe(chat_id: str) -> None:
    with _lock:
        data = _load()
        subs = set(data.get("subscribers", []))
        subs.discard(chat_id)
        data["subscribers"] = sorted(subs)
        _save(data)


def broadcast(
    advert: dict[str, Any],
    evaluation: dict[str, Any],
    bot_token: str,
) -> int:
    """Send one match to every subscriber; return the number delivered."""
    sent = 0
    for chat_id in subscribers():
        try:
            notify.send(advert, evaluation, bot_token, chat_id)
            sent += 1
        except Exception:  # noqa: BLE001 - keep going for other subscribers
            continue
    return sent


def _send_text(bot_token: str, chat_id: str, text: str) -> None:
    try:
        requests.post(
            f"{TELEGRAM_API}/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=20,
        )
    except requests.RequestException:
        pass


def _handle_update(update: dict[str, Any], bot_token: str) -> None:
    message = update.get("message")
    if not message:
        return
    chat_id = str(message.get("chat", {}).get("id", "") or "")
    if not chat_id:
        return
    text = (message.get("text") or "").strip()

    if text == "/start":
        subscribe(chat_id)
        _send_text(bot_token, chat_id, WELCOME)
    elif text == "/stop":
        unsubscribe(chat_id)
        _send_text(bot_token, chat_id, "Відписано 👋")
    else:
        _send_text(bot_token, chat_id, HELP_TEXT)


def _poll(bot_token: str) -> None:
    while True:
        try:
            with _lock:
                offset = _load().get("offset", 0)

            resp = requests.get(
                f"{TELEGRAM_API}/bot{bot_token}/getUpdates",
                params={"offset": offset, "timeout": 25},
                timeout=35,
            )
            data = resp.json()
            if not data.get("ok"):
                time.sleep(3)
                continue

            for update in data.get("result", []):
                _handle_update(update, bot_token)
                offset = max(offset, int(update.get("update_id", 0)) + 1)

            with _lock:
                state = _load()
                state["offset"] = offset
                _save(state)
        except (requests.RequestException, ValueError, OSError):
            time.sleep(3)


def start_polling(bot_token: str) -> None:
    """Start the getUpdates long-poll loop on a background daemon thread."""
    threading.Thread(target=_poll, args=(bot_token,), daemon=True).start()
