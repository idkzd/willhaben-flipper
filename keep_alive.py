"""Minimal HTTP server so Render sees the web service as healthy.

Render free web services must bind to the ``$PORT`` environment variable and
answer HTTP requests, otherwise the deploy never becomes "live".

Render free web services also spin down after 15 minutes without *inbound*
traffic. The bot only makes outbound calls (willhaben, Telegram), so without
help it would be put to sleep every ~15 minutes — and the local filesystem
(including subscribers.json) is wiped on every spin-down.

To prevent that, we ping our own public URL (RENDER_EXTERNAL_URL) every few
minutes from a background thread. Render's load balancer sees an inbound
request and keeps the instance awake. GitHub Actions remains as a backup to
re-wake the instance after a deploy/restart.

Uses only the Python standard library — no extra dependencies.
"""
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Well under Render's 15-minute idle timeout.
SELF_PING_INTERVAL = 300  # seconds
SELF_PING_TIMEOUT = 15


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"I'm alive")

    def log_message(self, *args) -> None:  # noqa: ANN002 - keep logs clean
        pass


def _serve() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    server.serve_forever()


def _self_ping_loop() -> None:
    """Ping our own public URL so Render never puts the instance to sleep."""
    url = os.environ.get("RENDER_EXTERNAL_URL", "").strip()
    if not url:
        print("⚠️  RENDER_EXTERNAL_URL відсутній — self-ping вимкнено", flush=True)
        return  # local dev — nothing to keep awake
    print(f"🔁 Self-ping увімкнено: {url} (кожні {SELF_PING_INTERVAL // 60} хв)", flush=True)
    while True:
        time.sleep(SELF_PING_INTERVAL)
        try:
            urllib.request.urlopen(url, timeout=SELF_PING_TIMEOUT).read()
        except Exception:  # noqa: BLE001 - keepalive must never crash the bot
            pass


def keep_alive() -> None:
    """Start the HTTP server and the self-ping thread (daemon threads)."""
    threading.Thread(target=_serve, daemon=True).start()
    threading.Thread(target=_self_ping_loop, daemon=True).start()
