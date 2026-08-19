"""Fetch and parse the willhaben.at "zu verschenken" search result list.

The page is Next.js-rendered and ships the full result set inside a
``<script id="__NEXT_DATA__">`` JSON blob, so we don't need any internal
private API — a single GET per page gives us structured advert data.
"""
import json
import re
from typing import Any

import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

BASE_URL = "https://www.willhaben.at/iad/"
IMAGE_BASE = "https://cache.willhaben.at/mmo/"

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)


def _attr_map(advert: dict[str, Any]) -> dict[str, str]:
    """Flatten attributes.attribute -> {NAME: first value}."""
    out: dict[str, str] = {}
    attributes = advert.get("attributes", {}).get("attribute", [])
    for attr in attributes:
        values = attr.get("values") or []
        if values:
            out[attr.get("name", "")] = str(values[0])
    return out


def _parse_advert(advert: dict[str, Any]) -> dict[str, Any]:
    attrs = _attr_map(advert)
    seo_url = attrs.get("SEO_URL", "")
    if not seo_url.startswith("http"):
        seo_url = BASE_URL + seo_url.lstrip("/")

    images: list[str] = []
    for path in attrs.get("ALL_IMAGE_URLS", "").split(";"):
        path = path.strip()
        if path:
            images.append(IMAGE_BASE + path)

    return {
        "id": str(advert.get("id", "")),
        "title": attrs.get("HEADING") or advert.get("description", ""),
        "body": attrs.get("BODY_DYN", ""),
        "location": attrs.get("LOCATION", ""),
        "state": attrs.get("STATE", ""),
        "district": attrs.get("DISTRICT", ""),
        "postcode": attrs.get("POSTCODE", ""),
        "price": attrs.get("PRICE/AMOUNT", "0"),
        "published": attrs.get("PUBLISHED_String", ""),
        "url": seo_url,
        "images": images,
    }


def fetch_adverts(url: str, timeout: int = 30) -> list[dict[str, Any]]:
    """Fetch one search-result page and return a list of parsed adverts."""
    resp = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "de-AT,de;q=0.9,en;q=0.8",
        },
        timeout=timeout,
    )
    resp.raise_for_status()

    match = _NEXT_DATA_RE.search(resp.text)
    if not match:
        raise RuntimeError("__NEXT_DATA__ blob not found in the page")

    data = json.loads(match.group(1))
    search_result = (
        data.get("props", {})
        .get("pageProps", {})
        .get("searchResult", {})
    )
    advert_list = (
        search_result.get("advertSummaryList", {}).get("advertSummary", [])
    )
    return [_parse_advert(a) for a in advert_list if isinstance(a, dict)]
