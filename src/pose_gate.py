"""Local pose gate: cheap seizure-likelihood pre-filter for captured events.

Runs dog pose estimation on an event's burst frames (10 fps), extracts
limb-keypoint displacement time series, and measures rhythmic power in the
1.5-4.8 Hz band (clonic jerking sits around 2-6 Hz clinically; 4.8 Hz is the
practical ceiling at 10 fps sampling — Nyquist is 5 Hz).

Decision contract is FAIL-OPEN: no dog, low keypoint confidence, missing
model, or any backend error escalates to the (expensive) VLM verify step.
The gate may only ever save money, never silently drop a seizure.

The scoring core is pure numpy so it imports and tests in the global env;
the YOLO backend lazy-imports ultralytics and is meant to run inside
.venv-pose (see SEIZUREGUARD_POSE_PYTHON in monitor.py).
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

BURST_FPS = 10.0
BAND_LO, BAND_HI = 1.5, 4.8
GATE_THRESHOLD = 0.45        # peak-concentration ratio above this = seizure-like
MIN_CONF = 0.30              # mean keypoint confidence below this = fail open
MIN_FRAMES = 10
MIN_DETECTION_RATE = 0.6     # dog missing from too many frames = fail open


PEAK_WINDOW_HZ = 0.4
GRID_LO_HZ = 0.5         # analysis grid floor (walk cadence and up)
GRID_STEP_HZ = 0.1


def oscillation_score(kpts, times=None, fps=BURST_FPS):
    """Rhythmic-motion score from a keypoint trajectory.

    kpts: float array (T, K, 3) — x, y, confidence per keypoint per frame.
    times: optional (T,) seconds. Burst frames are irregularly sampled
    (dedup holes), so spectra are computed with a non-uniform DFT over the
    true timestamps — a uniform FFT would smear high-band coherence (a
    0.2 s hole is half a cycle at 3 Hz).

    For each keypoint axis, finds the dominant frequency inside the
    1.5-4.8 Hz band of its linearly-detrended, confidence-weighted position
    series and measures how much of the total 0.5-4.8 Hz power concentrates
    within +-0.4 Hz of it. Rhythmic jerking concentrates power in that
    narrow window; broadband noise and smooth motion do not. Returns the
    max over keypoints, in [0, 1].
    """
    kpts = np.asarray(kpts, dtype=np.float64)
    if kpts.ndim != 3 or kpts.shape[0] < MIN_FRAMES:
        return 0.0
    T = kpts.shape[0]
    if times is None:
        t = np.arange(T) / fps
    else:
        t = np.asarray(times, dtype=np.float64)
    t = t - t.mean()

    xy = kpts[:, :, :2]
    conf = kpts[:, :, 2]

    grid = np.arange(GRID_LO_HZ, BAND_HI + GRID_STEP_HZ / 2, GRID_STEP_HZ)
    band_mask = grid >= BAND_LO
    if not band_mask.any():
        return 0.0
    basis = np.exp(-2j * np.pi * np.outer(grid, t))    # (F, T)

    best = 0.0
    for k in range(kpts.shape[1]):
        weight = conf[:, k]
        for axis in range(2):
            # Detrended position: locomotion drift disappears into the
            # trend, white keypoint jitter stays flat, real oscillation
            # stands out. (Differencing would tilt noise toward high
            # frequencies; a |velocity| magnitude would rectify to 2x.)
            series = xy[:, k, axis] * weight
            if np.allclose(series, series[0]):
                continue
            trend = np.polyval(np.polyfit(t, series, 1), t)
            series = series - trend
            power = np.abs(basis @ series) ** 2
            total = power.sum()
            if total <= 0.0:
                continue
            peak_freq = grid[band_mask][np.argmax(power[band_mask])]
            window = np.abs(grid - peak_freq) <= PEAK_WINDOW_HZ
            best = max(best, float(power[window].sum() / total))
    return best


def gate_decision(score, mean_conf):
    """'skip' only when we confidently saw a dog moving non-rhythmically."""
    if mean_conf < MIN_CONF:
        return "escalate"
    return "escalate" if score >= GATE_THRESHOLD else "skip"


# ---------------------------------------------------------------- backend

def yolo_keypoints(frame_paths):
    """frames -> (kpts (N, K, 3), kept_indices) via ultralytics YOLO pose.

    Frames where no dog is detected are skipped rather than fatal — a
    curled-up or partially occluded dog in a few frames must not kill the
    whole analysis. Raises only when NO frame has a detection."""
    from ultralytics import YOLO

    model_path = os.environ.get("SEIZUREGUARD_POSE_MODEL", "models/dog-pose.pt")
    model = YOLO(model_path)
    rows, kept = [], []
    for i, p in enumerate(frame_paths):
        result = model.predict(str(p), verbose=False)[0]
        if result.keypoints is None or result.keypoints.data.shape[0] == 0:
            continue
        rows.append(result.keypoints.data[0].cpu().numpy())  # (K, 3)
        kept.append(i)
    if not rows:
        raise RuntimeError("no dog detected in any burst frame")
    return np.stack(rows), kept


def _frame_time(path):
    m = re.search(r"_t_([0-9.]+)s", path.name)
    return float(m.group(1)) if m else 0.0


SEGMENT_MAX_GAP = 0.35   # tolerate dedup holes (~0.2s); real peak gaps are >=3s


def _segments(times, max_gap=SEGMENT_MAX_GAP):
    """Split frame indices into contiguous runs — burst frames come from
    disjoint peak windows, and treating them as one continuous series would
    smear spectral peaks with phase jumps at the window boundaries."""
    segs, cur = [], [0]
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > max_gap:
            segs.append(cur)
            cur = [i]
        else:
            cur.append(i)
    segs.append(cur)
    return segs


def analyze_event(event_dir, keypoint_fn=yolo_keypoints):
    frames = sorted((Path(event_dir) / "burst").glob("frame_*.jpg"), key=_frame_time)
    if len(frames) < MIN_FRAMES:
        return {"score": 0.0, "decision": "escalate", "mean_conf": 0.0,
                "backend": "none", "detection_rate": 0.0,
                "reason": "too few burst frames"}
    try:
        kpts, kept = keypoint_fn(frames)
    except Exception as e:
        return {"score": 0.0, "decision": "escalate", "mean_conf": 0.0,
                "backend": "yolo", "detection_rate": 0.0,
                "reason": f"backend failed: {e}"}

    kpts = np.asarray(kpts)
    detection_rate = len(kept) / len(frames)
    mean_conf = float(kpts[:, :, 2].mean())
    if len(kept) < MIN_FRAMES:
        return {"score": 0.0, "decision": "escalate", "mean_conf": mean_conf,
                "backend": "yolo", "detection_rate": detection_rate,
                "reason": "too few frames with a detected dog"}
    if detection_rate < MIN_DETECTION_RATE:
        # A convulsing dog is far from the pose model's training poses; when
        # detection is this spotty, "no rhythm seen" means "couldn't look".
        return {"score": 0.0, "decision": "escalate", "mean_conf": mean_conf,
                "backend": "yolo", "detection_rate": detection_rate,
                "reason": "detection rate below fail-open floor"}
    all_times = np.array([_frame_time(p) for p in frames])
    times = all_times[kept]
    score, scored_segments = 0.0, 0
    for seg in _segments(times):
        if len(seg) < MIN_FRAMES:
            continue
        scored_segments += 1
        score = max(score, oscillation_score(kpts[seg], times=times[seg]))
    if scored_segments == 0:
        # score=0.0 here would read as "confidently non-rhythmic" when the
        # detections were too fragmented to measure anything at all.
        return {"score": 0.0, "decision": "escalate", "mean_conf": mean_conf,
                "backend": "yolo", "detection_rate": detection_rate,
                "reason": "no segment long enough to score"}
    return {"score": score, "decision": gate_decision(score, mean_conf),
            "mean_conf": mean_conf, "backend": "yolo",
            "detection_rate": detection_rate, "reason": None}


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if len(sys.argv) != 2:
        print("Usage: python src/pose_gate.py <event_dir>")
        raise SystemExit(1)
    event_dir = Path(sys.argv[1]).resolve()
    if not event_dir.exists():
        raise RuntimeError(f"Event dir not found: {event_dir}")

    result = analyze_event(event_dir)
    out = event_dir / "pose_gate.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("✅ Pose gate:", json.dumps(result))


if __name__ == "__main__":
    main()
