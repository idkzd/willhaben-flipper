"""Minimal HTTP server so Render sees the web service as healthy.

Render free web services must bind to the ``$PORT`` environment variable and
answer HTTP requests, otherwise the deploy never becomes "live". This also
gives an external uptime pinger (e.g. UptimeRobot) a URL to hit so the free
instance doesn't spin down after 15 minutes of inactivity.

Uses only the Python standard library — no extra dependencies.
"""
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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


def keep_alive() -> None:
    """Start the HTTP server on a background daemon thread."""
    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
