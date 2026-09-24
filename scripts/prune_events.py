"""Disk hygiene for data/events: keep every event from the N days before
the newest event, forever keep events the verifier flagged positive (they
are the future training set), and forever keep one verified negative per
camera per day as a hard negative; delete the other older negatives. Never
touches anything else.

Usage: python scripts/prune_events.py [--root data/events] [--keep-days 14] [--dry-run]
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

# Everything after the time is the camera key. monitor.py puts a "_<n>"
# collision suffix in front of the camera name; it is deliberately not
# stripped: a camera named like "2_cam" cannot be told apart from it, and an
# ambiguous name keeping its own slot (one extra event kept) is safe where
# merging two cameras would delete one camera's only hard negative.
NAME_RE = re.compile(r"^event_(\d{8})_\d{6}_(.+)$")


def is_positive(event_dir):
    analysis = event_dir / "analysis.json"
    if not analysis.exists():
        return False
    try:
        return bool(json.loads(analysis.read_text(encoding="utf-8"))
                    .get("final_abnormal_event"))
    except Exception:
        return True          # unreadable verdict: keep, don't guess


def sample_score(event_dir):
    """How sign-like a verified negative looked: the number of observed-sign
    entries marked present across its batches, using decide_signs' present
    test (entries are counted, not de-duplicated by sign name as
    decide_signs does per batch). None when the event is not a clean verified
    negative — no verdict, a positive, or batches that failed (an unanalyzed
    batch is not evidence of normal behaviour).

    The claude-cli verifier stores observed_signs exactly as the model
    returned them, so any shape can appear. An odd shape must never raise:
    this runs over every event before anything is deleted, and one exception
    would stop every nightly prune for good."""
    try:
        a = json.loads((event_dir / "analysis.json").read_text(encoding="utf-8"))
        if (not isinstance(a, dict) or a.get("final_abnormal_event") is not False
                or a.get("failed_batches")):
            return None
        batches = a.get("batches")
        score = 0
        for b in batches if isinstance(batches, list) else []:
            signs = b.get("observed_signs") if isinstance(b, dict) else None
            if isinstance(signs, list):
                score += sum(1 for s in signs if isinstance(s, dict) and s.get("present"))
        return score
    except Exception:
        return None


def daily_samples(events):
    """One verified negative per camera per calendar day is kept as a hard
    negative: the one the verifier found most sign-like, ties to the earliest.
    Normal life of this dog in this room is the false-positive regression
    set, and pruning every negative left nothing to regress against. The
    choice is stable across runs: the kept event stays the best of what
    remains, unless a new event for that same day and camera turns up."""
    best = {}
    for ev in events:                      # sorted by name, i.e. by time
        m = NAME_RE.match(ev.name)
        score = sample_score(ev) if m else None
        if score is None:
            continue
        key = m.groups()                   # (day, camera)
        if key not in best or score > best[key][0]:
            best[key] = (score, ev)
    return {ev for _, ev in best.values()}


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
    samples = daily_samples(events)
    removed, freed = [], 0
    for ev in events:
        if ev.stat().st_mtime >= cutoff or ev in samples or is_positive(ev):
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
