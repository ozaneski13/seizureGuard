import bisect
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 360

# Base sampling
BASE_FPS = 2.0
BASE_INTERVAL = 1.0 / BASE_FPS

# Burst sampling (during peaks)
BURST_FPS = 10.0
BURST_INTERVAL = 1.0 / BURST_FPS
BURST_SECONDS = 3.0
NUM_BURSTS = 3
MIN_PEAK_SEP_SEC = 6.0
BUCKET_SEC = 30.0
PEAK_FLOOR = 1.0         # drift-level scores never earn a coverage burst

# Motion score config
DIFF_THRESH = 10
GLOBAL_CHANGE_MAX = 0.6  # >60% of pixels changed = lighting/exposure event, not motion

def motion_score(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    return float(np.mean(mask))

def is_global_change(score):
    return score / 255.0 > GLOBAL_CHANGE_MAX

def encode_jpg(frame):
    ok, jpg = cv2.imencode(".jpg", frame)
    return jpg if ok else None

def resize_frame(frame, max_side=640):
    """Scale so the longest side is max_side, preserving aspect ratio.
    A fixed WIDTHxHEIGHT resize distorts portrait sources (phone footage),
    which hurts both pose confidence and VLM judgment."""
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))

class FrameBuffer:
    """Time-ordered buffer of JPEG-encoded frames (RAM-friendly vs raw arrays)."""

    def __init__(self):
        self.times = []
        self.frames = []

    def append(self, t, jpg):
        self.times.append(t)
        self.frames.append(jpg)

    def __len__(self):
        return len(self.times)

    def nearest(self, target_t):
        if not self.times:
            return None
        i = bisect.bisect_left(self.times, target_t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.times):
                if best is None or abs(self.times[j] - target_t) < abs(self.times[best] - target_t):
                    best = j
        return self.times[best], self.frames[best]

def select_peaks(motion_series):
    """Strongest NUM_BURSTS peaks (>= MIN_PEAK_SEP_SEC apart) plus a coverage
    guarantee: every BUCKET_SEC region whose best score reaches PEAK_FLOOR
    contributes its best moment. Global top-N alone starved a 40 s sustained
    seizure on real footage because three isolated spikes elsewhere outranked
    the whole region."""
    if not motion_series:
        return []
    peaks = []
    for t, sc in sorted(motion_series, key=lambda x: x[1], reverse=True):
        if sc < PEAK_FLOOR:
            break
        if all(abs(t - p) >= MIN_PEAK_SEP_SEC for p in peaks):
            peaks.append(t)
        if len(peaks) >= NUM_BURSTS:
            break
    t0 = min(t for t, _ in motion_series)
    buckets = {}
    for t, sc in motion_series:
        b = int((t - t0) // BUCKET_SEC)
        if sc >= PEAK_FLOOR and (b not in buckets or sc > buckets[b][1]):
            buckets[b] = (t, sc)
    for t, _ in buckets.values():
        if all(abs(t - p) >= MIN_PEAK_SEP_SEC for p in peaks):
            peaks.append(t)
    peaks.sort()
    return peaks

def save_event_frames(fb, motion_series, out_dir, t_start, t_end):
    """Save base (2 fps) + burst (10 fps around motion peaks) frames from a
    FrameBuffer into out_dir/{base,burst}, timestamps relative to t_start.
    Never saves the same buffered frame twice. Returns (base_saved,
    burst_saved, peaks) with peaks in absolute time."""
    out_dir = Path(out_dir)
    (out_dir / "base").mkdir(parents=True, exist_ok=True)
    (out_dir / "burst").mkdir(parents=True, exist_ok=True)

    saved_ts = set()
    base_saved = 0
    n_base = int((t_end - t_start) * BASE_FPS) + 1
    for i in range(n_base):
        tt = t_start + i * BASE_INTERVAL
        item = fb.nearest(tt)
        if item is None:
            continue
        t0, jpg = item
        if t0 in saved_ts:
            continue
        saved_ts.add(t0)
        rel = t0 - t_start
        out = out_dir / "base" / f"frame_{i:03d}_t_{rel:0.3f}s.jpg"
        out.write_bytes(jpg.tobytes())
        base_saved += 1

    series = [(t, s) for t, s in motion_series if t_start <= t <= t_end]
    peaks = select_peaks(series)
    half = BURST_SECONDS / 2.0
    burst_saved = 0
    for p in peaks:
        s = max(t_start, p - half)
        e = min(t_end, p + half)
        tt = s
        while tt <= e + 1e-6:
            item = fb.nearest(tt)
            tt += BURST_INTERVAL
            if item is None:
                continue
            t0, jpg = item
            if t0 in saved_ts:
                continue
            saved_ts.add(t0)
            rel = t0 - t_start
            out = out_dir / "burst" / f"frame_{burst_saved:03d}_t_{rel:0.3f}s.jpg"
            out.write_bytes(jpg.tobytes())
            burst_saved += 1

    return base_saved, burst_saved, peaks
