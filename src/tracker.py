"""Dog-centering PTZ tracker for a pan-capable camera behind a local PTZ
gateway. Detects the dog with YOLO, keeps it horizontally centered with
small rate-limited pan steps, and coordinates with the monitor through a
moving-flag file so self-commanded motion never reads as an event.

The gateway itself is NOT part of this repo — any HTTP service satisfying
one endpoint works:

    GET {SEIZUREGUARD_PTZ_URL}/api/ptz?cam=<name>&dir=<left|right|up|down>&steps=<n>
    -> {"ok": true}

Design constraints measured on the mi360 (2026-08-09): one motor step is
~18 px at 640 width, command-to-settle is ~1.5 s, and rapid command bursts
can wedge the motor controller — hence the strict rate limits. The gateway's
direction labels are unverified (SSIM proves motion, not direction), so the
pan sign is learned from observed effect and flips itself if a correction
makes the offset worse.

Runs inside .venv-pose (ultralytics). The pure decision logic imports
without it. Only ever pans; tilt is deliberately out of scope.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

TRACK_INTERVAL = 2.0        # seconds between decisions
DEADBAND_FRAC = 0.14        # ignore offsets inside this fraction of width
STEP_PX = 18.0              # measured px @640 per motor step
MAX_STEPS_PER_CMD = 6
GAIN = 0.5                  # half-correction per cycle avoids overshoot
MIN_CMD_GAP_SEC = 4.0
MAX_CMDS_PER_MIN = 8        # motor-controller wedge protection
SETTLE_BASE_SEC = 1.5
SETTLE_PER_STEP_SEC = 0.35
LOST_HOME_SEC = 300.0       # dog gone this long -> undo net steps
DOG_CLASS = 16              # COCO 'dog'
DET_CONF = 0.35


def decide_command(cx, frame_w, pan_sign=1):
    """Dog center x -> (direction, steps) or None. pan_sign=+1 means
    'right' command moves the dog left in frame (the expected geometry)."""
    off = cx - frame_w / 2.0
    if abs(off) <= frame_w * DEADBAND_FRAC:
        return None
    steps = int(min(MAX_STEPS_PER_CMD, max(1, round(abs(off) * GAIN / STEP_PX))))
    toward = "right" if off > 0 else "left"
    if pan_sign < 0:
        toward = "left" if toward == "right" else "right"
    return toward, steps


class PanSign:
    """Learns whether the gateway's left/right labels match reality: if a
    correction increases the offset by more than a step, the sign flips."""

    def __init__(self):
        self.sign = 1
        self._before = None

    def sent(self, off_before):
        self._before = off_before

    def observed(self, off_after):
        if self._before is None:
            return
        if (abs(off_after) - abs(self._before)) > STEP_PX:
            self.sign = -self.sign
        self._before = None


class CommandBudget:
    """Token-bucket rate limit: minimum gap plus a per-minute cap."""

    def __init__(self, min_gap=MIN_CMD_GAP_SEC, per_min=MAX_CMDS_PER_MIN):
        self.min_gap = min_gap
        self.per_min = per_min
        self.sent_times = []

    def allow(self, now):
        self.sent_times = [t for t in self.sent_times if now - t < 60.0]
        if self.sent_times and now - self.sent_times[-1] < self.min_gap:
            return False
        return len(self.sent_times) < self.per_min

    def record(self, now):
        self.sent_times.append(now)


def write_moving_flag(path, until):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"moving_until": until}), encoding="utf-8")


# ---------------------------------------------------------------- backends

def ptz_move(base_url, cam, direction, steps, timeout=20):
    url = f"{base_url}/api/ptz?cam={cam}&dir={direction}&steps={steps}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read()).get("ok") is True


def detect_dog(model, frame):
    """-> (cx, conf) of the highest-confidence dog box, or None."""
    result = model.predict(frame, classes=[DOG_CLASS], conf=DET_CONF,
                           verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    i = int(boxes.conf.argmax())
    x1, _, x2, _ = boxes.xyxy[i].tolist()
    return (x1 + x2) / 2.0, float(boxes.conf[i])


def run(source, cam, ptz_url, flag_path):
    import cv2
    from ultralytics import YOLO

    model = YOLO(os.environ.get("SEIZUREGUARD_TRACK_MODEL", "yolo11n.pt"))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    print(f"✅ Tracking {cam} via {ptz_url} (source {source})", flush=True)

    budget = CommandBudget()
    pan_sign = PanSign()
    net_steps = 0            # +right, -left; undone when the dog is lost
    lost_since = None
    last_decision = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1.0)
            cap.release()
            cap = cv2.VideoCapture(source)
            continue
        now = time.time()
        if now - last_decision < TRACK_INTERVAL:
            continue
        last_decision = now

        h, w = frame.shape[:2]
        scale = 640.0 / w
        small = cv2.resize(frame, (640, int(h * scale)))
        hit = detect_dog(model, small)

        if hit is None:
            if lost_since is None:
                lost_since = now
            elif now - lost_since >= LOST_HOME_SEC and net_steps != 0:
                direction = "left" if net_steps > 0 else "right"
                steps = min(MAX_STEPS_PER_CMD, abs(net_steps))
                if budget.allow(now):
                    settle = SETTLE_BASE_SEC + SETTLE_PER_STEP_SEC * steps
                    write_moving_flag(flag_path, now + settle + 1.0)
                    if ptz_move(ptz_url, cam, direction, steps):
                        budget.record(now)
                        net_steps += steps if direction == "right" else -steps
                        print(f"[HOME] {direction} x{steps} (net {net_steps})",
                              flush=True)
            continue

        cx, conf = hit
        lost_since = None
        off = cx - 320.0
        pan_sign.observed(off)
        cmd = decide_command(cx, 640, pan_sign.sign)
        if cmd is None or not budget.allow(now):
            continue
        direction, steps = cmd
        settle = SETTLE_BASE_SEC + SETTLE_PER_STEP_SEC * steps
        write_moving_flag(flag_path, now + settle + 1.0)
        if ptz_move(ptz_url, cam, direction, steps):
            budget.record(now)
            pan_sign.sent(off)
            net_steps += steps if direction == "right" else -steps
            print(f"[TRACK] dog at {cx:.0f} conf {conf:.2f} -> {direction} "
                  f"x{steps} (sign {pan_sign.sign}, net {net_steps})", flush=True)


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="seizureGuard PTZ dog tracker")
    parser.add_argument("--source", required=True, help="RTSP stream URL")
    parser.add_argument("--cam", default="mi360", help="PTZ gateway camera name")
    parser.add_argument("--ptz-url", default=os.environ.get(
        "SEIZUREGUARD_PTZ_URL", "http://127.0.0.1:1985"))
    parser.add_argument("--flag", default=None,
                        help="moving-flag path shared with the monitor")
    args = parser.parse_args()
    flag = args.flag or str(Path("data/tracker") / f"{args.cam}_moving.json")
    run(args.source, args.cam, args.ptz_url, flag)


if __name__ == "__main__":
    main()
