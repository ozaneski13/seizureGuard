"""Disk hygiene for data/events: per camera, keep every event from the N
most recent days that camera recorded anything; forever keep events the
verifier flagged positive (they are the future training set), events it
never fully checked, events the monitor never finished (not processed, or
their alert never delivered), and one verified negative per camera per day
as a hard negative; delete the other older negatives. Never touches
anything else.

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


def is_unchecked(event_dir):
    """The verdict says the event was never fully checked: a batch failed or
    the backend was down. verify_event writes final_abnormal_event=false for
    these too, but a failed verification is not a negative, and an outage
    sends no per-event alert or clip, so this dir is the only copy. No
    analysis.json at all is not this case: with verify on, that event
    already got its own "unverified" alert with the clip (is_unfinished
    keeps it when that alert was not delivered). Never raises."""
    try:
        a = json.loads((event_dir / "analysis.json").read_text(encoding="utf-8"))
        return bool(a.get("failed_batches") or a.get("backend_outage"))
    except Exception:
        return False         # missing: see above; unreadable: is_positive keeps it


def is_unfinished(event_dir):
    """The monitor never finished with this event: captured (event_meta.json)
    but never processed (no handled.json; its restart sweep processes it),
    or alerted but the alert was never delivered (it keeps retrying). Nobody
    has seen it, so this dir is the only copy. A marker that does not read
    as a finished event is kept, not guessed as done. Dirs from before these
    files existed have neither and keep the other rules. Never raises."""
    try:
        handled = event_dir / "handled.json"
        if not handled.exists():
            return (event_dir / "event_meta.json").exists()
        h = json.loads(handled.read_text(encoding="utf-8"))
    except Exception:
        return True
    if not isinstance(h, dict):
        return True
    return bool(h.get("alerted")) and h.get("delivered") is not True


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


def recent_events(events, keep_days, now):
    """The window: per camera, every event from that camera's keep_days most
    recent days with events, by the day in the name. It counts monitored
    time, not calendar time. A now-based window deleted 129 negatives during
    the 2026-09 blind spell, and one anchored on the newest event still let
    a recording camera age out a blind one's archive, and let the first
    event after a gap wipe everything from before it. Now a blind camera
    loses nothing, and its pre-gap days go one per new recorded day. The
    camera key is NAME_RE's, so a rare "_<n>_" collision name counts as its
    own camera and is kept: one extra event, never a camera merged away.

    Names without a day and camera (hand-made copies such as
    event_vid20260111_seizure_window_fable) keep the old rule: keep_days
    back from the newest of them by mtime, min(now, ...) so a future-dated
    one (clock jump) cannot slide it forward. Anchored on those events only:
    anchoring on all events would let a recording camera age them out on
    the calendar again, and they are few and deliberate, so erring toward
    keeping them is cheap."""
    days, named, unnamed = {}, [], []
    for ev in events:
        m = NAME_RE.match(ev.name)
        if m:
            day, camera = m.groups()
            days.setdefault(camera, set()).add(day)
            named.append((ev, day, camera))
        else:
            unnamed.append(ev)
    recent = {camera: set(sorted(d, reverse=True)[:keep_days])
              for camera, d in days.items()}
    keep = {ev for ev, day, camera in named if day in recent[camera]}
    if unnamed:
        newest = min(now, max(ev.stat().st_mtime for ev in unnamed))
        cutoff = newest - keep_days * 86400
        keep.update(ev for ev in unnamed if ev.stat().st_mtime >= cutoff)
    return keep


def prune(root, keep_days, dry_run=False, now=None):
    """Delete every event outside its camera's keep_days most recent active
    days (recent_events) that is not a positive, an unchecked or unfinished
    event or a daily hard-negative sample."""
    now = time.time() if now is None else now
    events = [ev for ev in sorted(Path(root).glob("event_*")) if ev.is_dir()]
    if not events:
        return [], 0
    recent = recent_events(events, keep_days, now)
    samples = daily_samples(events)
    removed, freed = [], 0
    for ev in events:
        if (ev in recent or ev in samples or is_positive(ev)
                or is_unchecked(ev) or is_unfinished(ev)):
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
    print(f"{verb} {len(removed)} negative event(s) outside the {args.keep_days} "
          f"most recent active days per camera, {freed / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
