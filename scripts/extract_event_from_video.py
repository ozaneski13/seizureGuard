import sys
import time
from pathlib import Path

import cv2

from sampling import (
    BASE_FPS, BURST_FPS,
    FrameBuffer, encode_jpg, is_global_change, motion_score, resize_frame,
    save_event_frames,
)

VIDEO_PATH = Path("test_video.mp4")

OUT_ROOT = Path("data/test_events")
EVENT_SECONDS = 60.0

def extract_event(video_path, event_dir, max_seconds=EVENT_SECONDS):
    """Single sequential pass over a video file -> base/burst event dir.

    Returns {"base": n, "burst": n, "peaks": [...], "window": seconds}."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # (no per-frame seeking: faster and immune to keyframe seek inaccuracy)
    prev_gray = None
    motion_series = []      # (ts, score)
    buffer = FrameBuffer()  # (ts, jpg)

    while True:
        ret, fr = cap.read()
        if not ret:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ts > max_seconds:
            break
        fr = resize_frame(fr)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            sc = motion_score(prev_gray, gray)
            # Skip lighting/exposure events so bursts stay on real motion
            if not is_global_change(sc):
                motion_series.append((ts, sc))
        prev_gray = gray

        jpg = encode_jpg(fr)
        if jpg is not None:
            buffer.append(ts, jpg)

    cap.release()

    if len(buffer) == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    window = buffer.times[-1]

    base_saved, burst_saved, peaks = save_event_frames(
        buffer, motion_series, event_dir, 0.0, window)
    return {"base": base_saved, "burst": burst_saved, "peaks": peaks,
            "window": window}


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if not VIDEO_PATH.exists():
        raise RuntimeError(f"Video not found: {VIDEO_PATH.resolve()}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    event_dir = OUT_ROOT / f"event_{stamp}_video"

    stats = extract_event(VIDEO_PATH, event_dir)

    print("✅ Event created:", event_dir.resolve())
    print(f"Base saved: {stats['base']} frames (@{BASE_FPS} fps for {stats['window']:.1f}s)")
    print(f"Burst peaks (s): {[round(p, 3) for p in stats['peaks']]}")
    print(f"Burst saved: {stats['burst']} frames (@{BURST_FPS} fps bursts)")

if __name__ == "__main__":
    main()
