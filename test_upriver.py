"""Diagnostic for the Upriver trend layer.

    python test_upriver.py          # full run,  ~15 API calls
    python test_upriver.py -q       # quick run, ~8 API calls

Answers four questions:
  1. Does the key work and what verticals does the API accept?
  2. Are the fields pipeline.py reads still present in the response?
  3. Does gate() actually fire, and on which kinds of subject?
  4. Are the topics it lets through plausibly related to the query?

Does NOT load the VLM. classify() and judge() need a GPU and a human to judge
the output, so they are out of scope here - this only covers the API layer and
the pure-python filtering on top of it.

Exit code is 0 when the trend layer is usable, 1 when something structural is
broken (bad key, missing fields, gate that never fires).
"""

import argparse
import inspect
import sys
from collections import Counter

import pipeline
import upriver.upriver as up

# Subjects that should find trends: Upriver only covers these verticals.
LIVE_SUBJECTS = [
    ("ai coding assistants", "tech"),
    ("new iphone launch", "tech"),
    ("nfl season", "sports"),
    ("election debate", "politics"),
    ("street photography", None),
]

# Ordinary Instagram subject matter, which Upriver does not cover. These are
# expected to come back empty - that is the designed "no trend fits" path,
# not a failure.
DEAD_SUBJECTS = [
    ("latte art coffee shop morning", None),
    ("golden retriever on the beach", None),
    ("sunset hike mountain trail", None),
]

# Values probed against the search endpoint. "creative" is accepted but is not
# listed by /v1/topics/breakout/metadata, so it is checked separately.
PROBE_VERTICALS = ["tech", "sports", "politics", "creative", "lifestyle", "food"]

# pipeline.title_of / score_of read these. A rename upstream silently empties
# the trend layer, so they are asserted rather than assumed.
REQUIRED_FIELDS = ["topic_name", "score"]
OPTIONAL_FIELDS = ["canonical_name", "trend", "status", "vertical", "citations"]

calls = 0
failures = []
warnings = []


def ok(msg):
    print(f"  [OK]   {msg}")


def bad(msg):
    print(f"  [FAIL] {msg}")
    failures.append(msg)


def warn(msg):
    print(f"  [WARN] {msg}")
    warnings.append(msg)


def search(query, vertical=None, limit=8):
    global calls
    calls += 1
    return up.search_topics(query, vertical=vertical, limit=limit)


def section(title):
    print(f"\n{'-' * 68}\n{title}\n{'-' * 68}")


def check_auth():
    section("1. auth and advertised verticals")
    global calls
    try:
        calls += 1
        meta = up._request("GET", "/v1/topics/breakout/metadata")
    except Exception as exc:
        bad(f"cannot reach the API: {type(exc).__name__}: {exc}")
        return []
    listed = [v.get("id") for v in meta.get("verticals", [])]
    ok(f"key accepted; metadata lists {len(listed)} verticals: {listed}")
    for v in meta.get("verticals", []):
        cats = v.get("categories") or []
        if cats:
            print(f"         {v.get('id')} has {len(cats)} categories: "
                  f"{[c.get('id') for c in cats][:6]}...")
    return listed


def check_verticals(listed):
    section("2. which vertical values does search actually accept?")
    accepted = []
    for v in PROBE_VERTICALS:
        global calls
        try:
            calls += 1
            up._request("POST", "/v1/topics/breakout/search",
                        json={"query": "test", "limit": 1, "vertical": v})
            accepted.append(v)
            tag = "" if v in listed else "  <- accepted but NOT in metadata"
            print(f"  [OK]   {v:12} accepted{tag}")
        except up.UpriverError:
            print(f"  ..     {v:12} rejected")

    client_side = set(up.VERTICALS)
    missing = [v for v in accepted if v not in client_side]
    stale = [v for v in client_side if v not in accepted]
    if missing:
        warn(f"upriver.VERTICALS is missing accepted vertical(s): {missing} "
             f"- search_topics() will raise ValueError for them")
    if stale:
        bad(f"upriver.VERTICALS lists vertical(s) the API rejects: {stale}")
    if not missing and not stale:
        ok(f"upriver.VERTICALS matches what the API accepts: {sorted(client_side)}")
    return accepted


def check_schema(subjects):
    section("3. are the fields pipeline.py depends on still there?")
    sample = []
    for query, vertical in subjects:
        try:
            sample += search(query, vertical)
        except Exception as exc:
            bad(f"search failed for {query!r}: {type(exc).__name__}: {exc}")
            return []
    if not sample:
        bad("no topics returned at all - cannot verify the schema")
        return []

    print(f"  sampled {len(sample)} topics")
    for field in REQUIRED_FIELDS:
        have = sum(1 for t in sample if t.get(field) is not None)
        pct = 100 * have // len(sample)
        (ok if pct == 100 else bad)(
            f"{field:16} present on {have}/{len(sample)} ({pct}%)")
    for field in OPTIONAL_FIELDS:
        have = sum(1 for t in sample if t.get(field) is not None)
        print(f"         {field:16} present on {have}/{len(sample)}")

    # The accessors are what actually matter - a field can exist while the
    # accessor still returns nothing, which is how the title bug hid.
    titled = sum(1 for t in sample if pipeline.title_of(t))
    scored = sum(1 for t in sample if pipeline.score_of(t) > 0)
    momentum = sum(1 for t in sample if pipeline.momentum_of(t) > 0)
    print()
    if titled == len(sample):
        ok(f"pipeline.title_of()    resolves {titled}/{len(sample)}")
    else:
        bad(f"pipeline.title_of()    resolves only {titled}/{len(sample)} "
            f"- gate() drops every topic it cannot title")

    # Some topics legitimately carry score 0 (58/61 in an earlier sample), so
    # only a total absence means the field was renamed upstream.
    if scored == len(sample):
        ok(f"pipeline.score_of()    non-zero on all {scored}")
    elif scored:
        warn(f"pipeline.score_of()    non-zero on only {scored}/{len(sample)} "
             f"- topics without a score can never pass gate()")
    else:
        bad("pipeline.score_of()    non-zero on 0 topics - 'score' was likely "
            "renamed upstream; gate() can never fire")
    print(f"  [--]   pipeline.momentum_of() non-zero {momentum}/{len(sample)} "
          f"(sparse upstream; not safe to filter on)")

    print(f"\n  status values : {dict(Counter(t.get('status') for t in sample))}")
    print(f"  vertical values: {dict(Counter(t.get('vertical') for t in sample))}")
    return sample


def check_gate(live, dead):
    section("4. does gate() fire, and only where it should?")
    print(f"  installed gate signature: gate{inspect.signature(pipeline.gate)}\n")

    fired_live, fired_dead = 0, 0

    print("  subjects Upriver covers - these SHOULD produce trends:")
    for query, vertical in live:
        topics = search(query, vertical)
        passed = pipeline.gate(topics)
        best = max((pipeline.score_of(t) for t in topics), default=0.0)
        fired_live += bool(passed)
        flag = "fired" if passed else "EMPTY"
        print(f"    {flag:6} {query[:30]:32} raw={len(topics):2d} "
              f"kept={len(passed)} best={best:.3f}")
        for t in passed[:2]:
            print(f"           -> {pipeline.score_of(t):.3f}  "
                  f"{pipeline.title_of(t)[:54]}")

    print("\n  ordinary Instagram subjects - these SHOULD stay empty:")
    for query, vertical in dead:
        topics = search(query, vertical)
        passed = pipeline.gate(topics)
        best = max((pipeline.score_of(t) for t in topics), default=0.0)
        fired_dead += bool(passed)
        flag = "LEAKED" if passed else "empty"
        print(f"    {flag:6} {query[:30]:32} raw={len(topics):2d} "
              f"kept={len(passed)} best={best:.3f}")
        for t in passed[:2]:
            print(f"           -> {pipeline.score_of(t):.3f}  "
                  f"{pipeline.title_of(t)[:54]}")

    print()
    if fired_live == 0:
        bad(f"gate() fired on 0/{len(live)} covered subjects - the trend layer "
            f"is dead. Check the thresholds and the accessors above.")
    elif fired_live < len(live) / 2:
        warn(f"gate() fired on only {fired_live}/{len(live)} covered subjects "
             f"- thresholds may be too strict")
    else:
        ok(f"gate() fired on {fired_live}/{len(live)} covered subjects")

    if fired_dead == len(dead):
        warn(f"gate() let topics through on all {fired_dead}/{len(dead)} "
             f"uncovered subjects - thresholds may be too loose, so judge() "
             f"is carrying the whole relevance burden")
    else:
        ok(f"gate() stayed empty on {len(dead) - fired_dead}/{len(dead)} "
           f"uncovered subjects")

    return fired_live, fired_dead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--quick", action="store_true",
                        help="fewer queries, fewer API calls")
    args = parser.parse_args()

    live = LIVE_SUBJECTS[:2] if args.quick else LIVE_SUBJECTS
    dead = DEAD_SUBJECTS[:1] if args.quick else DEAD_SUBJECTS

    print("Upriver trend-layer diagnostic")
    print(f"base url: {up.BASE_URL}")

    listed = check_auth()
    if not listed:
        print("\nRESULT: cannot reach the API - nothing else can be checked.")
        return 1

    check_verticals(listed)
    check_schema(live[:2])
    check_gate(live, dead)

    section("summary")
    print(f"  api calls used : {calls}")
    print(f"  failures       : {len(failures)}")
    print(f"  warnings       : {len(warnings)}")
    for f in failures:
        print(f"    FAIL {f}")
    for w in warnings:
        print(f"    WARN {w}")

    if failures:
        print("\nRESULT: the trend layer is BROKEN. See failures above.")
        return 1
    print("\nRESULT: the trend layer works. Relevance of the kept topics still "
          "needs a human or judge() to confirm - this script only proves the "
          "API, the schema and the filtering behave.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
