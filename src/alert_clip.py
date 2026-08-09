"""Builds the short alert video: picks the time window of the
highest-confidence positive batch from analysis.json (the moment the
verifier was most sure), clamps it to 5-10 s, and encodes the event's
saved frames from that window into a small MP4 at ~real-time speed.

Every failure returns None — the alert itself must never be blocked by
clip encoding; the caller falls back to the peak-frame photo.
"""
from pathlib import Path

from verify_event import BATCH_SIZE, frame_time

MIN_CLIP_SEC = 5.0
MAX_CLIP_SEC = 10.0
MIN_FPS, MAX_FPS = 2.0, 15.0


def event_frames(event_dir):
    """All saved frames of an event, chronologically — the exact ordering
    verify_event batches over, so batch indexes map 1:1."""
    event_dir = Path(event_dir)
    frames = sorted(
        list((event_dir / "base").glob("frame_*.jpg"))
        + list((event_dir / "burst").glob("frame_*.jpg")),
        key=frame_time,
    )
    return frames


def clamp_window(t_lo, t_hi, min_len=MIN_CLIP_SEC, max_len=MAX_CLIP_SEC):
    """Center-clamp a window into [min_len, max_len]."""
    mid = (t_lo + t_hi) / 2.0
    length = max(min_len, min(max_len, t_hi - t_lo))
    return mid - length / 2.0, mid + length / 2.0


def best_batch_window(analysis, frames, batch_size=BATCH_SIZE):
    """Time window of the highest-confidence positive batch, or None."""
    batches = (analysis or {}).get("batches") or []
    best_i, best_conf = None, -1.0
    for i, b in enumerate(batches):
        if b.get("abnormal_event") is True and float(b.get("confidence", 0.0)) > best_conf:
            best_i, best_conf = i, float(b.get("confidence", 0.0))
    if best_i is None:
        return None
    chunk = frames[best_i * batch_size:(best_i + 1) * batch_size]
    if not chunk:
        return None
    return clamp_window(frame_time(chunk[0]), frame_time(chunk[-1]))


def peak_window(peaks, t_start):
    """Fallback window around the first motion peak (unverified events)."""
    if not peaks:
        return None
    center = peaks[0] - t_start
    return clamp_window(center - MAX_CLIP_SEC / 2.0, center + MAX_CLIP_SEC / 2.0)


def make_clip(event_dir, window, out_path=None):
    """Encode the frames inside `window` into an MP4; returns path or None."""
    try:
        import cv2

        if window is None:
            return None
        t_lo, t_hi = window
        frames = [p for p in event_frames(event_dir) if t_lo <= frame_time(p) <= t_hi]
        if len(frames) < 4:
            return None
        duration = max(0.5, frame_time(frames[-1]) - frame_time(frames[0]))
        fps = max(MIN_FPS, min(MAX_FPS, len(frames) / duration))

        first = cv2.imread(str(frames[0]))
        if first is None:
            return None
        h, w = first.shape[:2]
        out_path = Path(out_path or Path(event_dir) / "alert_clip.mp4")
        writer = None
        for fourcc in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if writer.isOpened():
                break
            writer.release()
            writer = None
        if writer is None:
            return None
        for p in frames:
            img = cv2.imread(str(p))
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
        writer.release()
        return out_path if out_path.exists() and out_path.stat().st_size > 0 else None
    except Exception:
        return None
