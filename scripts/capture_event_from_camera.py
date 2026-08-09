import sys
import time
from pathlib import Path

import cv2

from sampling import (
    WIDTH, HEIGHT, FrameBuffer, encode_jpg, is_global_change,
    motion_score, resize_frame, save_event_frames,
)

CAM_INDEX = 0
OUT_ROOT = Path("data/events")

EVENT_SECONDS = 60.0

def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    event_dir = OUT_ROOT / f"event_{stamp}_camera"

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    # Record JPEG-encoded frames in memory for 60s (for peak detection)
    start = time.time()
    end = start + EVENT_SECONDS

    prev_gray = None
    motion_series = []      # (t, score)
    buffer = FrameBuffer()  # (t, jpg)

    print("✅ Recording 60s from camera...")

    while time.time() < end:
        ret, frame = cap.read()
        if not ret:
            continue
        t = time.time()
        frame = resize_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            sc = motion_score(prev_gray, gray)
            # Skip lighting/exposure events so bursts stay on real motion
            if not is_global_change(sc):
                motion_series.append((t, sc))
        prev_gray = gray

        jpg = encode_jpg(frame)
        if jpg is not None:
            buffer.append(t, jpg)

    cap.release()

    if len(buffer) == 0:
        raise RuntimeError("No frames captured from camera")

    base_saved, burst_saved, peaks = save_event_frames(
        buffer, motion_series, event_dir, start, end)

    print("✅ Event created:", event_dir.resolve())
    print(f"Base saved: {base_saved}")
    print(f"Burst saved: {burst_saved}")
    print(f"Peak times (rel s): {[round(p - start, 3) for p in peaks]}")

if __name__ == "__main__":
    main()
