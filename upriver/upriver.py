import json
import time
from pathlib import Path
from config import UPRIVER_API_KEY

import requests

from config import UPRIVER_API_KEY

BASE_URL = "https://api.upriver.ai"
CACHE_DIR = Path(__file__).parent / "cache"
CATEGORIES_PATH = CACHE_DIR / "media_categories.json"

_session = requests.Session()
_session.headers["X-API-Key"] = UPRIVER_API_KEY


class UpriverError(RuntimeError):
    pass


def _request(method, path, retries=3, **kwargs):
    for attempt in range(retries + 1):
        resp = _session.request(method, BASE_URL + path, timeout=60, **kwargs)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(2 ** attempt)
            continue
        if not resp.ok:
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text
            raise UpriverError(f"{method} {path} -> {resp.status_code}: {detail}")
        return resp.json()


def get_categories(refresh=False):
    """{'l1': [...], 'l2': [...]} from /v1/media_categories, cached to disk."""
    if CATEGORIES_PATH.exists() and not refresh:
        return json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
    data = _request("GET", "/v1/media_categories")
    CACHE_DIR.mkdir(exist_ok=True)
    CATEGORIES_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data

VERTICALS = ("tech", "sports", "politics")


def search_topics(query, vertical=None, limit=10):
    """Breakout topics matching free text. Returns a list of topic dicts."""
    if vertical is not None and vertical not in VERTICALS:
        raise ValueError(f"vertical must be one of {VERTICALS} or None, got {vertical!r}")
    body = {"query": query, "limit": limit, "include": ["citations"]}
    if vertical:
        body["vertical"] = vertical
    data = _request("POST", "/v1/topics/breakout/search", json=body)
    return data.get("topics", [])