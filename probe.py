"""probe.py - re-run the API checks behind the deck.

Usage (from the repo folder, with UPRIVER_API_KEY in .env):

    python probe.py docs        # slide 7: relevance_score, llm_relevance, verticals, unknown keys
    python probe.py score       # slide 6: score moves with the query and is pre-sorted
    python probe.py coverage    # slide 9: verticals accepted, result counts stop early
    python probe.py fields      # slide 10: citations, trend, status, entities
    python probe.py drown --desc "<a ~50 word photo description>"   # slide 8
    python probe.py all

Nothing here imports the project, so it keeps working if the pipeline changes.
Numbers drift day to day: look for the pattern, not the exact value.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = os.environ.get("UPRIVER_BASE", "https://api.upriver.ai")
KEY = os.environ.get("UPRIVER_API_KEY")
SEARCH = BASE + "/v1/topics/breakout/search"
METADATA = BASE + "/v1/topics/breakout/metadata"


def search(query, **kw):
    """POST one search. Returns (status, seconds, payload). Never raises on 4xx."""
    body = {"query": query, "mode": "vector", "limit": 12,
            "include": ["citations", "entities"], **kw}
    body = {k: v for k, v in body.items() if v is not None}
    start = time.time()
    r = requests.post(SEARCH, json=body, headers={"X-API-Key": KEY}, timeout=60)
    secs = time.time() - start
    try:
        payload = r.json()
    except ValueError:
        payload = {"raw": r.text[:300]}
    return r.status_code, secs, payload


def topics_of(payload):
    return payload.get("topics") or payload.get("results") or []


def head(text):
    print("\n" + text)
    print("-" * len(text))


def line(label, observed, expected):
    print(f"  {label}")
    print(f"    saw      : {observed}")
    print(f"    expected : {expected}")


# ---------------------------------------------------------------- slide 7
def check_docs():
    head("Slide 7 - the docs and the live endpoint disagree")

    status, secs, payload = search("ai coding assistants")
    blob = json.dumps(payload)
    tops = topics_of(payload)
    line("relevance_score in the payload",
         f'"relevance" in response = {"relevance" in blob}  (status {status}, {secs:.2f}s)',
         "False")
    if tops:
        print(f"    keys     : {sorted(tops[0].keys())}")

    def ids(payload):
        return [(t.get("topic_id"), t.get("score")) for t in topics_of(payload)]

    a = ids(search("ai data centers", llm_relevance=True)[2])
    b = ids(search("ai data centers", llm_relevance=False)[2])
    c = ids(search("ai data centers")[2])
    line("llm_relevance true / false / unset",
         f"identical ids and scores = {a == b == c}",
         "True, so the parameter does nothing")

    for key in ("bogus_param", "search_mode"):
        status, _, _ = search("ai data centers", **{key: 1 if key == "bogus_param" else "vector"})
        line(f"unknown request key {key}", f"HTTP {status}", "200, accepted and ignored")

    status, _, payload = search("test", vertical="lifestyle", limit=1)
    detail = json.dumps(payload)[:200]
    line("vertical=lifestyle", f"HTTP {status}: {detail}",
         "400 naming sports, tech, politics only")

    try:
        meta = requests.get(METADATA, headers={"X-API-Key": KEY}, timeout=30).json()
        print(f"    metadata : {json.dumps(meta)[:300]}")
    except requests.RequestException as exc:
        print(f"    metadata : request failed ({exc})")


# ---------------------------------------------------------------- slide 6
def check_score():
    head("Slide 6 - score ranks within a query, not across queries")

    _, _, payload = search("ai coding assistants", limit=50)
    scores = [t.get("score") or 0 for t in topics_of(payload)]
    line("results arrive sorted by score",
         f"sorted descending = {scores == sorted(scores, reverse=True)} over {len(scores)} rows",
         "True")

    _, _, p1 = search("man in a gaming chair at his desk", limit=50)
    _, _, p2 = search("gaming setup at home", limit=50)
    a = {t["topic_id"]: t.get("score") for t in topics_of(p1)}
    b = {t["topic_id"]: t.get("score") for t in topics_of(p2)}
    shared = a.keys() & b.keys()
    differ = sum(a[i] != b[i] for i in shared)
    line("same topic under two queries",
         f"{differ} of {len(shared)} shared topics scored differently",
         "most of them, so score moves with the query")

    _, _, p3 = search("man in a gaming chair at his desk", limit=5)
    top = topics_of(p3)
    if top:
        print(f"    top hit  : {round(top[0].get('score') or 0, 3)}  {top[0].get('topic_name')}")
        print("    note     : re-run later to see the same hit decay (0.684 -> 0.673 in my test)")


# ---------------------------------------------------------------- slide 9
def check_coverage():
    head("Slide 9 - coverage is news-shaped")

    for vertical in ["tech", "sports", "politics", "creative",
                     "lifestyle", "food", "travel", "entertainment",
                     "business", "science", "fashion", "gaming"]:
        status, _, _ = search("test", vertical=vertical, limit=1)
        print(f"    {vertical:<14} HTTP {status}")

    for query in ["golden retriever on the beach", "latte art coffee shop morning",
                  "sunset hike mountain trail", "ai coding assistants"]:
        _, _, payload = search(query, limit=100)
        print(f"    limit=100 -> {len(topics_of(payload)):>3} rows   {query}")
    print("    expected : everyday subjects stop well short of 100; the tech query fills it")

    _, _, payload = search("street photography", limit=20)
    print(f"    verticals: {Counter(t.get('vertical') for t in topics_of(payload))}")


# ---------------------------------------------------------------- slide 10
def check_fields():
    head("Slide 10 - what the pipeline trusts in each response")

    queries = ["ai coding assistants", "nfl season", "election debate", "street photography",
               "gaming setup", "latte art", "new iphone launch", "data centers",
               "college football", "ai regulation", "camera gear", "coffee shops"]
    topics = []
    for query in queries:
        _, _, payload = search(query, limit=20)
        topics.extend(topics_of(payload))
    cites = [c for t in topics for c in (t.get("citations") or [])]
    print(f"    sample   : {len(topics)} topics, {len(cites)} citations")

    def pct(count, total):
        return f"{count} of {total} ({round(100 * count / total)}%)" if total else "n/a"

    line("citation title vs display",
         "title empty " + pct(sum(not c.get("title") for c in cites), len(cites)) +
         ", display empty " + pct(sum(not c.get("display") for c in cites), len(cites)),
         "title empty on about a quarter, display never empty")
    line("trend block",
         "missing on " + pct(sum(t.get("trend") is None for t in topics), len(topics)),
         "missing on most topics")
    line("status",
         str(Counter(t.get("status") for t in topics)),
         "almost all declining")
    line("entities",
         "absent on " + pct(sum(not t.get("entities") for t in topics), len(topics)),
         "absent on about a third")

    ents = [e for t in topics for e in (t.get("entities") or [])]
    if ents:
        high = sum((e.get("confidence") or 0) >= 0.8 for e in ents)
        line("entity confidence floor",
             f"{high} of {len(ents)} entities already score 0.8 or higher",
             "the 0.8 floor rarely bites")


# ---------------------------------------------------------------- slide 8
def check_drown(desc):
    head("Slide 8 - a long description drowns a short story")
    if not desc:
        print("    skipped: pass --desc with a ~50 word photo description")
        print("    get one from: python json_to_table.py -d last")
        return

    story = "lets build a data center"
    for label, query in [("story alone", story), ("description + story", desc + " " + story)]:
        _, _, payload = search(query)
        tops = topics_of(payload)[:3]
        print(f"    {label}")
        for t in tops:
            print(f"      {round(t.get('score') or 0, 3)}  {t.get('topic_name')}")
    print("    expected : data-center topics alone, unrelated topics once the description leads")


CHECKS = {"docs": check_docs, "score": check_score,
          "coverage": check_coverage, "fields": check_fields}


def main():
    parser = argparse.ArgumentParser(description="Re-run the API checks behind the deck.")
    parser.add_argument("check", nargs="?", default="docs",
                        choices=list(CHECKS) + ["drown", "all"])
    parser.add_argument("--desc", default="", help="photo description for the drown check")
    args = parser.parse_args()

    if not KEY:
        sys.exit("UPRIVER_API_KEY is not set. Run this from the repo folder, with .env in place.")

    print(f"probe.py against {BASE} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.check == "all":
        for fn in CHECKS.values():
            fn()
        check_drown(args.desc)
    elif args.check == "drown":
        check_drown(args.desc)
    else:
        CHECKS[args.check]()


if __name__ == "__main__":
    main()
