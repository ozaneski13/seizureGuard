"""Synthetic test video generator (not collected by pytest — no test_ prefix).

Timeline (70s @ 30fps, 640x360):
  - drifting circle throughout (ambient motion)
  - 20-23s: violent jitter  -> should win a burst peak
  - 40-41s: full-frame brightness flash -> global change, must never be a peak
  - 50-52s: moderate jitter -> should win a second peak
"""
import cv2
import numpy as np

W, H, FPS, SECONDS = 640, 360, 30, 70

VIOLENT_START, VIOLENT_END = 20.0, 23.0
FLASH_START, FLASH_END = 40.0, 41.0
MODERATE_START, MODERATE_END = 50.0, 52.0


def make_video(path):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(42)

    for f in range(FPS * SECONDS):
        t = f / FPS
        frame = np.full((H, W, 3), 40, np.uint8)

        cx = int(W / 2 + 100 * np.sin(t * 0.3))
        cy = int(H / 2 + 40 * np.cos(t * 0.2))

        if VIOLENT_START <= t <= VIOLENT_END:
            cx += int(rng.integers(-60, 61))
            cy += int(rng.integers(-60, 61))

        if MODERATE_START <= t <= MODERATE_END:
            cx += int(rng.integers(-25, 26))
            cy += int(rng.integers(-25, 26))

        if FLASH_START <= t <= FLASH_END:
            frame[:] = 220

        cv2.circle(frame, (cx, cy), 30, (0, 200, 255), -1)
        out.write(frame)

    out.release()
    return path
