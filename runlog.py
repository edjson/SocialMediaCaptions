"""Append-only record of every caption run.

One JSON object per line, so the file survives a crash mid-write, greps
cleanly, and never needs a schema migration. Nothing here may break a
caption: a missing log line is cheaper than a failed generate.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent / "runs"
LOG_PATH = LOG_DIR / "runs.jsonl"


def new_run_id():
    """Ties the 'generate' line to the 'accepted' line written later."""
    return uuid.uuid4().hex[:12]


def image_key(image):
    """Stable id for a photo so repeat runs on it can be grouped.

    Hashes the pixels, not the file, so the same photo re-uploaded or
    re-encoded still groups together.
    """
    try:
        head = f"{image.mode}{image.size}".encode()
        return hashlib.sha1(head + image.tobytes()).hexdigest()[:12]
    except Exception:
        return ""


def log(event, **fields):
    try:
        LOG_DIR.mkdir(exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            **fields,
        }
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def read(path=None):
    """Every well-formed line. Bad lines are skipped, not fatal."""
    path = Path(path) if path else LOG_PATH
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows
