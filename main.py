#!/usr/bin/env python3
"""Parse new willhaben "Zu verschenken" ads, score resale via OpenRouter,
and broadcast the promising ones to every Telegram user who sent /start.

Usage:
    python3 main.py             # poll continuously (default every 10 s)
    python3 main.py --once      # run a single pass and exit
    python3 main.py --loop 30   # poll every 30 seconds
    python3 main.py --dry-run   # evaluate and print, but don't send anything
"""
import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import config
import evaluate
import keep_alive
import telegram_bot
import willhaben

STATE_FILE = Path(__file__).resolve().parent / "state.json"

AREA_VIENNA_ID = "900"  # willhaben areaId for Wien


def load_state() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    try:
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")))
    except (ValueError, OSError):
        return set()


MAX_STATE_IDS = 5000


def save_state(seen: set[str]) -> None:
    # Cap growth so a long-running bot doesn't bloat the file forever.
    ids = sorted(seen)
    if len(ids) > MAX_STATE_IDS:
        ids = ids[-MAX_STATE_IDS:]
    STATE_FILE.write_text(json.dumps(ids, indent=2), encoding="utf-8")


def verify_vienna_filter(url: str) -> None:
    """Warn loudly if the search URL is not scoped to Vienna."""
    query = parse_qs(urlparse(url).query)
    area_id = (query.get("areaId") or [""])[0]
    if area_id == AREA_VIENNA_ID:
        print(f"✅ Фільтр: areaId={area_id} → Відень (Wien)")
    else:
        print(
            f"⚠️  УВАГА: areaId={area_id or 'відсутній'} — це НЕ Відень. "
            f"Додай &areaId={AREA_VIENNA_ID} у WILLHABEN_URL.",
            file=sys.stderr,
        )


def page_url(base_url: str, page: int) -> str:
    parsed = urlparse(base_url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query["page"] = [str(page)]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def build_candidate_list(args: argparse.Namespace, verbose: bool = True) -> list[dict]:
    """Collect adverts across the configured pages."""
    adverts: list[dict] = []
    for page in range(1, args.pages + 1):
        url = page_url(args.willhaben_url, page)
        if verbose:
            print(f"📥 Завантажую сторінку {page}...")
        batch = willhaben.fetch_adverts(url)
        if verbose:
            print(f"   Отримано {len(batch)} оголошень")
        adverts.extend(batch)
    return adverts


def process_once(args: argparse.Namespace, seen: set[str], verbose: bool = True) -> int:
    sent = 0
    evaluated = 0
    adverts = build_candidate_list(args, verbose=verbose)

    # Newest ads are at the top, so iterate in that order.
    for advert in adverts:
        advert_id = advert["id"]
        if not advert_id or advert_id in seen:
            continue

        # Cost control: cap the number of LLM calls per run.
        if evaluated >= args.max_llm_calls:
            print(
                f"⏹  Досягнуто ліміт {args.max_llm_calls} оцінок за запуск, "
                "решта — наступного разу."
            )
            break

        # Gentle pacing to avoid tripping the free models' rate limits.
        if evaluated > 0 and args.llm_delay:
            time.sleep(args.llm_delay)

        evaluation = evaluate.evaluate(
            advert, args.openrouter_api_key, args.openrouter_model
        )
        evaluated += 1
        score = evaluation.get("resale_score", 0)
        potential = evaluation.get("resale_potential", "")

        print(
            f"🔍 {advert['title'][:60]!r} → {score}/10 "
            f"({potential}) — {evaluation.get('estimated_price_eur')} €",
            flush=True,
        )

        if score >= args.min_score or potential == "very_good":
            if args.dry_run:
                subs = len(telegram_bot.subscribers())
                print(
                    f"   ✅ [DRY-RUN] відправив би {subs} підписникам: {advert['url']}",
                    flush=True,
                )
                sent += 1
            else:
                count = telegram_bot.broadcast(
                    advert, evaluation, args.telegram_bot_token
                )
                print(f"   ✅ Надіслано {count} підписникам у Telegram", flush=True)
                sent += count

        seen.add(advert_id)

    # Dry runs shouldn't leave side effects behind.
    if not args.dry_run:
        save_state(seen)
    return sent


def main() -> int:
    config.load_env()

    # Bind $PORT so Render (and an uptime pinger) can reach the service.
    try:
        keep_alive.keep_alive()
    except OSError as exc:
        print(f"⚠️  keep_alive не запустився (локально це ок): {exc}", flush=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate but do not send to Telegram",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single pass and exit",
    )
    parser.add_argument(
        "--loop",
        type=int,
        metavar="SECONDS",
        help="polling interval in seconds (overrides LOOP_INTERVAL)",
    )
    args = parser.parse_args()

    args.willhaben_url = config.get(
        "WILLHABEN_URL",
        "https://www.willhaben.at/iad/kaufen-und-verkaufen/zu-verschenken/marktplatz"
        "?sfId=188bc8f2-24d3-4bc4-9a7c-acead7e55731&rows=30&isNavigation=true"
        "&areaId=900&page=1&PRICE_FROM=0&PRICE_TO=0",
    )
    args.pages = config.get_int("PAGES", 1)
    args.min_score = config.get_int("MIN_RESALE_SCORE", 7)
    args.max_llm_calls = config.get_int("MAX_LLM_CALLS_PER_RUN", 30)
    args.llm_delay = config.get_int("LLM_REQUEST_DELAY", 2)
    args.loop_interval = args.loop if args.loop else config.get_int("LOOP_INTERVAL", 10)
    args.openrouter_api_key = config.get("OPENROUTER_API_KEY")
    args.openrouter_model = config.get(
        "OPENROUTER_MODEL",
        "google/gemma-4-26b-a4b-it:free,google/gemma-4-31b-it:free,z-ai/glm-5.2:free",
    )
    args.telegram_bot_token = config.get("TELEGRAM_BOT_TOKEN")

    if not args.openrouter_api_key:
        print("❌ Відсутній OPENROUTER_API_KEY у .env", file=sys.stderr)
        return 1
    if not args.dry_run and not args.telegram_bot_token:
        print("❌ Відсутній TELEGRAM_BOT_TOKEN у .env", file=sys.stderr)
        return 1

    verify_vienna_filter(args.willhaben_url)
    seen = load_state()

    if args.once:
        sent = process_once(args, seen)
        print(f"\n🏁 Готово. Надіслано в Telegram: {sent}")
        return 0

    print(
        f"🔁 Запускаю цикл: перевірка кожні {args.loop_interval} с "
        "(Ctrl+C для виходу)",
        flush=True,
    )
    if not args.dry_run:
        telegram_bot.start_polling(args.telegram_bot_token)
        print("🤖 Telegram-бот слухає /start та /stop", flush=True)
    try:
        while True:
            sent = process_once(args, seen, verbose=False)
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] нових надіслано: {sent}", flush=True)
            time.sleep(args.loop_interval)
    except KeyboardInterrupt:
        print("\n👋 Зупинено")
    return 0


if __name__ == "__main__":
    sys.exit(main())
