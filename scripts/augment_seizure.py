"""Inject synthetic seizure-like rhythmic jitter into the dog region of a video.

Takes real (normal) dog footage and oscillates the detected dog's bounding-box
region at a clonic frequency (default 3 Hz) for a time window — producing a
recall-test clip whose motion signature matches seizure dynamics while
everything else (room, dog, lighting) stays real. Validates the motion
trigger and the pose gate's 1.5-4.8 Hz detector on your own footage without
waiting for a real seizure (synthetic-anomaly validation, cf. DATASETS.md).

The dog detector needs ultralytics — run inside .venv-pose:
    .venv-pose\\Scripts\\python scripts/augment_seizure.py in.mp4 out.mp4
"""
import argparse
import math
import sys
from pathlib import Path

import cv2


def yolo_bbox_fn():
    """Returns frame -> (x1, y1, x2, y2) or None, using the pose model."""
    import os

    from ultralytics import YOLO

    model = YOLO(os.environ.get("SEIZUREGUARD_POSE_MODEL", "models/dog-pose.pt"))

    def detect(frame):
        result = model.predict(frame, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x1, y1, x2, y2 = boxes[areas.argmax()]
        return int(x1), int(y1), int(x2), int(y2)

    return detect


def shift_region(frame, bbox, dx, dy):
    """Paste the bbox region back shifted by (dx, dy), clipped to the frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    region = frame[y1:y2, x1:x2].copy()
    out = frame.copy()
    nx1, ny1 = x1 + dx, y1 + dy
    sx1, sy1 = max(0, -nx1), max(0, -ny1)
    nx1, ny1 = max(0, nx1), max(0, ny1)
    nx2 = min(w, nx1 + region.shape[1] - sx1)
    ny2 = min(h, ny1 + region.shape[0] - sy1)
    if nx2 <= nx1 or ny2 <= ny1:
        return out
    out[ny1:ny2, nx1:nx2] = region[sy1:sy1 + (ny2 - ny1), sx1:sx1 + (nx2 - nx1)]
    return out


def augment(video_in, video_out, freq=3.0, amp=12, start=5.0, dur=8.0,
            bbox_fn=None):
    """Returns the number of frames that were actually jittered."""
    if bbox_fn is None:
        bbox_fn = yolo_bbox_fn()

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_in}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    jittered = 0
    frame_idx = 0
    last_bbox = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        frame_idx += 1
        if start <= t < start + dur:
            bbox = bbox_fn(frame) or last_bbox
            if bbox is not None:
                last_bbox = bbox
                phase = 2 * math.pi * freq * t
                dx = int(round(amp * math.sin(phase)))
                dy = int(round(0.6 * amp * math.sin(phase + math.pi / 2)))
                frame = shift_region(frame, bbox, dx, dy)
                jittered += 1
        writer.write(frame)

    cap.release()
    writer.release()
    return jittered


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_in", type=Path)
    parser.add_argument("video_out", type=Path)
    parser.add_argument("--freq", type=float, default=3.0,
                        help="jitter frequency in Hz (clonic band: 1.5-4.8)")
    parser.add_argument("--amp", type=int, default=12,
                        help="jitter amplitude in pixels")
    parser.add_argument("--start", type=float, default=5.0)
    parser.add_argument("--dur", type=float, default=8.0)
    args = parser.parse_args()

    jittered = augment(args.video_in, args.video_out, freq=args.freq,
                       amp=args.amp, start=args.start, dur=args.dur)
    print(f"✅ Wrote {args.video_out} ({jittered} frames jittered "
          f"@{args.freq}Hz amp={args.amp}px, window {args.start}-"
          f"{args.start + args.dur}s)")
    if jittered == 0:
        print("[WARN] no dog detected in the jitter window — output is "
              "identical to the input")


if __name__ == "__main__":
    main()
