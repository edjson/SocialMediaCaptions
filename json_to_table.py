"""Read runs/runs.jsonl and print it as a table.

    python json_to_table.py              # one row per run
    python json_to_table.py -n 5         # last 5 runs
    python json_to_table.py -d 3         # everything about run 3
    python json_to_table.py -d last      # everything about the latest run

The summary answers "what happened", the detail view answers "why".
"""

import argparse

import runlog


def cut(text, width):
    """One line, at most `width` chars, ellipsis when trimmed."""
    text = " ".join(str(text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def table(rows, columns):
    """columns: list of (header, width, fn). Widths are minimums."""
    widths = [max(w, len(h)) for h, w, _ in columns]
    head = "  ".join(h.ljust(w) for (h, _, _), w in zip(columns, widths))
    print(head)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(cut(fn(row), w).ljust(w)
                        for (_, _, fn), w in zip(columns, widths)))


def load():
    """Generate events, each with its matching accept event attached."""
    rows = runlog.read()
    accepted = {r.get("run_id"): r for r in rows if r.get("event") == "accepted"}
    runs = [r for r in rows if r.get("event") == "generate"]
    for r in runs:
        r["_final"] = (accepted.get(r.get("run_id")) or {}).get("caption_final")
    return runs


def edited(run):
    final = run.get("_final")
    if final is None:
        return "-"
    return "yes" if final.strip() != (run.get("caption") or "").strip() else "no"


def summary(runs):
    columns = [
        ("#", 3, lambda r: r["_n"]),
        ("time", 8, lambda r: r.get("ts", "")[11:19]),
        ("story", 20, lambda r: r.get("story")),
        ("caption", 46, lambda r: r.get("caption")),
        ("raw/gate/kept", 13,
         lambda r: f"{len(r.get('raw') or [])}/{len(r.get('gate_ids') or [])}"
                   f"/{len(r.get('kept_ids') or [])}"),
        ("judge", 8, lambda r: r.get("judge_verdict") or "-"),
        ("regen", 5, lambda r: {True: "yes", False: "no"}.get(r.get("regenerated"), "-")),
        ("edit", 4, edited),
        ("secs", 5, lambda r: f"{sum((r.get('timings') or {}).values()):.1f}"),
    ]
    table(runs, columns)


def detail(run):
    timings = run.get("timings") or {}
    print(f"run {run['_n']}   {run.get('ts')}   {run.get('run_id')}")
    print(f"image     {run.get('image')}  {run.get('image_size')}  "
          f"vertical={run.get('vertical')}")
    print(f"total     {sum(timings.values()):.1f}s  " +
          "  ".join(f"{k}={v}s" for k, v in timings.items()))
    print(f"settings  {run.get('settings')}")

    for label in ("story", "voice", "description"):
        print(f"\n{label.upper()}\n  {run.get(label) or '(empty)'}")

    print("\nQUERIES SENT")
    for q in run.get("queries") or [run.get("query")]:
        print(f"  - {cut(q, 100)}")

    gate_ids = set(run.get("gate_ids") or [])
    kept_ids = set(run.get("kept_ids") or [])
    print(f"\nCANDIDATES  ({len(run.get('raw') or [])} returned, "
          f"{len(gate_ids)} passed gate, {len(kept_ids)} kept by judge)")
    for t in sorted(run.get("raw") or [], key=lambda t: -(t.get("score") or 0)):
        if t.get("id") in kept_ids:
            mark = "KEPT "
        elif t.get("id") in gate_ids:
            mark = "judge"
        else:
            mark = "gate "
        print(f"  {mark} {t.get('score'):.3f} [{t.get('vertical'):<9}] "
              f"{cut(t.get('title'), 58)}")
    print(f"  judge replied: {run.get('judge_verdict') or '(nothing)'}")
    print(f"  keywords     : {run.get('keywords') or '(none)'}")

    if run.get("regenerated"):
        print(f"\nECHO RETRY\n  first attempt: {run.get('caption_first')!r}")
    print(f"\nCAPTION\n  raw   {run.get('caption_raw')!r}")
    print(f"  clean {run.get('caption')!r}")
    if run.get("_final") is not None:
        print(f"  final {run['_final']!r}   edited={edited(run)}")

    print("\nPROMPT")
    for line in (run.get("prompt") or "").splitlines():
        print(f"  {line}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--limit", type=int, help="show only the last N runs")
    parser.add_argument("-d", "--detail", help="run number, or 'last'")
    args = parser.parse_args()

    runs = load()
    for i, r in enumerate(runs, 1):
        r["_n"] = i
    if not runs:
        print("no runs logged yet")
        return

    if args.detail:
        n = len(runs) if args.detail == "last" else int(args.detail)
        match = [r for r in runs if r["_n"] == n]
        if not match:
            print(f"no run {n}; there are {len(runs)}")
            return
        detail(match[0])
        return

    summary(runs[-args.limit:] if args.limit else runs)
    print(f"\n{len(runs)} runs   "
          f"{sum(1 for r in runs if r.get('kept_ids'))} used a trend   "
          f"{sum(1 for r in runs if r.get('regenerated'))} regenerated   "
          f"{sum(1 for r in runs if edited(r) == 'yes')} edited")


if __name__ == "__main__":
    main()
