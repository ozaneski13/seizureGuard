"""Builds the short alert video: picks the time window of the
highest-confidence positive batch from analysis.json (the moment the
verifier was most sure), clamps it to 5-10 s, and cuts a small MP4 from
that window — preferring the full-rate event recording (event.mp4, saved
from the ring buffer at capture time) and falling back to the sparse
analysis frames for older events.

Every failure returns None — the alert itself must never be blocked by
clip encoding; the caller falls back to the peak-frame photo.
"""
import json
from pathlib import Path

from verify_event import BATCH_SIZE, frame_time

MIN_CLIP_SEC = 5.0
MAX_CLIP_SEC = 10.0
MIN_FPS, MAX_FPS = 2.0, 20.0


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


def _open_writer(cv2, out_path, fps, size):
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    return None


def save_event_video(fb, event_dir, t_start, t_end):
    """Encode the ring buffer's full-rate JPEG frames for the event window
    into event.mp4 + a times sidecar. This is what alert clips are cut
    from, at the ring's native rate instead of the sparse analysis
    sampling. Returns the path or None; never raises."""
    try:
        import cv2
        import numpy as np

        items = [(t, jpg) for t, jpg in zip(fb.times, fb.frames)
                 if t_start <= t <= t_end]
        if len(items) < 4:
            return None
        duration = max(0.5, items[-1][0] - items[0][0])
        fps = max(MIN_FPS, min(MAX_FPS, len(items) / duration))

        first = cv2.imdecode(np.frombuffer(items[0][1].tobytes(), np.uint8),
                             cv2.IMREAD_COLOR)
        if first is None:
            return None
        h, w = first.shape[:2]
        out_path = Path(event_dir) / "event.mp4"
        writer = _open_writer(cv2, out_path, fps, (w, h))
        if writer is None:
            return None
        times = []
        for t, jpg in items:
            img = cv2.imdecode(np.frombuffer(jpg.tobytes(), np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
            times.append(round(t - t_start, 3))
        writer.release()
        (Path(event_dir) / "event_times.json").write_text(
            json.dumps({"times": times}), encoding="utf-8")
        return out_path if out_path.exists() and out_path.stat().st_size > 0 else None
    except Exception:
        return None


def _clip_from_event_video(cv2, event_dir, t_lo, t_hi, out_path):
    """Cut the window from the full-rate event recording."""
    video = Path(event_dir) / "event.mp4"
    sidecar = Path(event_dir) / "event_times.json"
    if not (video.exists() and sidecar.exists()):
        return None
    times = json.loads(sidecar.read_text(encoding="utf-8"))["times"]
    keep = [i for i, t in enumerate(times) if t_lo <= t <= t_hi]
    if len(keep) < 4:
        return None
    duration = max(0.5, times[keep[-1]] - times[keep[0]])
    fps = max(MIN_FPS, min(MAX_FPS, len(keep) / duration))

    cap = cv2.VideoCapture(str(video))
    writer = None
    wanted = set(keep)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i > keep[-1]:
            break
        if i in wanted:
            if writer is None:
                h, w = frame.shape[:2]
                writer = _open_writer(cv2, out_path, fps, (w, h))
                if writer is None:
                    break
            writer.write(frame)
        i += 1
    cap.release()
    if writer is None:
        return None
    writer.release()
    return out_path if out_path.exists() and out_path.stat().st_size > 0 else None


def make_clip(event_dir, window, out_path=None):
    """Cut the window into an MP4; returns path or None. Prefers the
    full-rate event recording, falls back to the sparse saved frames."""
    try:
        import cv2

        if window is None:
            return None
        t_lo, t_hi = window
        out_path = Path(out_path or Path(event_dir) / "alert_clip.mp4")
        from_video = _clip_from_event_video(cv2, event_dir, t_lo, t_hi, out_path)
        if from_video is not None:
            return from_video
        frames = [p for p in event_frames(event_dir) if t_lo <= frame_time(p) <= t_hi]
        if len(frames) < 4:
            return None
        duration = max(0.5, frame_time(frames[-1]) - frame_time(frames[0]))
        fps = max(MIN_FPS, min(MAX_FPS, len(frames) / duration))

        first = cv2.imread(str(frames[0]))
        if first is None:
            return None
        h, w = first.shape[:2]
        writer = _open_writer(cv2, out_path, fps, (w, h))
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
