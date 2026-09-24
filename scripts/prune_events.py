"""Disk hygiene for data/events: keep every event from the N days before
the newest event, and forever keep events the verifier flagged positive
(they are the future training set); delete older negatives. Never touches
anything else.

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
    """The window is measured back from the newest event, not from now.
    While the monitor is blind no new events arrive, and a now-based window
    kept eating the archive anyway: it deleted 129 negatives during the
    2026-09 blind spell. min(now, ...) keeps a future-dated event (clock
    jump) from sliding the window forward."""
    now = time.time() if now is None else now
    events = [ev for ev in sorted(Path(root).glob("event_*")) if ev.is_dir()]
    if not events:
        return [], 0
    newest = min(now, max(ev.stat().st_mtime for ev in events))
    cutoff = newest - keep_days * 86400
    removed, freed = [], 0
    for ev in events:
        if ev.stat().st_mtime >= cutoff or is_positive(ev):
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
    print(f"{verb} {len(removed)} negative event(s) more than {args.keep_days}d "
          f"older than the newest event, {freed / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
