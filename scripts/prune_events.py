"""Disk hygiene for data/events: keep every event from the last N days,
and forever keep events the verifier flagged positive (they are the future
training set); delete older negatives. Never touches anything else.

Usage: python scripts/prune_events.py [--root data/events] [--keep-days 14] [--dry-run]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def is_positive(event_dir):
    analysis = event_dir / "analysis.json"
    if not analysis.exists():
        return False
    try:
        return bool(json.loads(analysis.read_text(encoding="utf-8"))
                    .get("final_abnormal_event"))
    except Exception:
        return True          # unreadable verdict: keep, don't guess


def dir_size(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def prune(root, keep_days, dry_run=False, now=None):
    now = time.time() if now is None else now
    cutoff = now - keep_days * 86400
    removed, freed = [], 0
    for ev in sorted(Path(root).glob("event_*")):
        if not ev.is_dir() or ev.stat().st_mtime >= cutoff or is_positive(ev):
            continue
        freed += dir_size(ev)
        removed.append(ev.name)
        if not dry_run:
            shutil.rmtree(ev)
    return removed, freed


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/events", type=Path)
    ap.add_argument("--keep-days", default=14, type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    removed, freed = prune(args.root, args.keep_days, args.dry_run)
    verb = "would remove" if args.dry_run else "removed"
    print(f"{verb} {len(removed)} negative event(s) older than {args.keep_days}d, "
          f"{freed / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
