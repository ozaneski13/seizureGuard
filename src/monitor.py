"""Continuous seizure monitor: watches a camera/stream/file through a ring
buffer, turns sustained motion into captured events, and runs each event
through the pose gate and VLM verification before alerting.

Usage: python src/monitor.py --source <index|file|rtsp url> [--name cam]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import alert_clip  # noqa: E402
import alerts  # noqa: E402
from sampling import (  # noqa: E402
    WIDTH, HEIGHT, FrameBuffer, encode_jpg, is_global_change,
    motion_score, resize_frame, save_event_frames,
)

OUT_ROOT = Path("data/events")

# Ring buffer
BUFFER_SECONDS = 90.0
RING_INTERVAL = 1.0 / 15.0   # store at most 15 fps in the ring (~40 MB) —
                             # also the frame rate of the alert video clips

# Trigger FSM (motion_score units; calibrate with --log-motion on real footage)
MOTION_ON = 2.5              # sustained score to arm an event
MOTION_OFF = 1.0             # below this counts as quiet
TRIGGER_ON_SEC = 2.0         # sustained motion required to start an event
QUIET_OFF_SEC = 10.0         # quiet time that ends an event
EVENT_MAX_SEC = 60.0         # hard cap per event
PRE_ROLL_SEC = 5.0           # context saved from before the trigger

# Live-stream health (camera index or RTSP/HTTP URL sources)
STREAM_RETRY_SEC = 3.0       # sustained read failure before a reconnect attempt
STREAM_BLIND_ALERT_SEC = 60.0  # still dead after this long -> one alert


class RingBuffer:
    """Bounded (t, jpg) history; drops entries older than BUFFER_SECONDS."""

    def __init__(self, seconds=BUFFER_SECONDS):
        self.seconds = seconds
        self.items = deque()

    def append(self, t, jpg):
        self.items.append((t, jpg))
        cutoff = t - self.seconds
        while self.items and self.items[0][0] < cutoff:
            self.items.popleft()

    def snapshot(self, t0, t1):
        fb = FrameBuffer()
        for t, jpg in self.items:
            if t0 <= t <= t1:
                fb.append(t, jpg)
        return fb


class MotionTrigger:
    """Pure FSM: idle -> pending (sustained motion) -> active -> event.

    feed() returns a completed (start, end) window or None. Global-change
    frames (lighting/exposure) never count as motion."""

    def __init__(self):
        self.state = "idle"
        self.pending_since = None
        self.event_start = None
        self.last_motion_t = None

    def _reset(self):
        self.state = "idle"
        self.pending_since = None
        self.event_start = None
        self.last_motion_t = None

    def feed(self, t, score, is_global):
        moving = (not is_global) and score >= MOTION_ON
        active_motion = (not is_global) and score > MOTION_OFF

        if self.state == "idle":
            if moving:
                self.state = "pending"
                self.pending_since = t
        elif self.state == "pending":
            if moving and t - self.pending_since >= TRIGGER_ON_SEC:
                self.state = "active"
                self.event_start = self.pending_since
                self.last_motion_t = t
            elif not active_motion:
                self._reset()
        elif self.state == "active":
            if active_motion:
                self.last_motion_t = t
            if t - self.event_start >= EVENT_MAX_SEC:
                ev = (self.event_start, t)
                self._reset()
                return ev
            if t - self.last_motion_t >= QUIET_OFF_SEC:
                ev = (self.event_start, self.last_motion_t)
                self._reset()
                return ev
        return None

    def flush(self):
        """End-of-stream: finalize an in-flight event."""
        if self.state == "active":
            ev = (self.event_start, self.last_motion_t)
            self._reset()
            return ev
        return None


class MovingFlag:
    """Reads the tracker-written flag marking self-commanded camera motion
    (SEIZUREGUARD_MOVING_FLAG). Frames inside the window count as global
    change, never as motion. Stale or malformed flags mean 'not moving' —
    a crashed tracker must never be able to blind the monitor."""

    def __init__(self, path):
        self.path = Path(path) if path else None
        self._mtime = None
        self._until = 0.0

    def moving(self, now):
        if self.path is None:
            return False
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return False
        if mtime != self._mtime:
            self._mtime = mtime
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._until = float(data.get("moving_until", 0.0))
            except Exception:
                self._until = 0.0
        return now < self._until


class StreamWatchdog:
    """Pure FSM for live-source read health: schedules reconnect attempts and
    a single 'monitor blind' alert. A blind seizure monitor must say so —
    silently spinning on a dead stream is the one failure the alert channel
    exists for."""

    def __init__(self, retry_sec=STREAM_RETRY_SEC, alert_sec=STREAM_BLIND_ALERT_SEC):
        self.retry_sec = retry_sec
        self.alert_sec = alert_sec
        self.stalled_since = None
        self.last_attempt = None
        self.alerted = False

    def ok(self):
        self.stalled_since = None
        self.last_attempt = None
        self.alerted = False

    def failed(self, now):
        """Returns actions to take: 'reconnect' and/or 'blind_alert'."""
        if self.stalled_since is None:
            self.stalled_since = now
            self.last_attempt = now
        actions = []
        if now - self.last_attempt >= self.retry_sec:
            self.last_attempt = now
            actions.append("reconnect")
        if not self.alerted and now - self.stalled_since >= self.alert_sec:
            self.alerted = True
            actions.append("blind_alert")
        return actions


# ---------------------------------------------------------------- pipeline

def claude_available():
    exe = shutil.which("claude")
    if exe is None:
        return False
    try:
        proc = subprocess.run([exe, "auth", "status"], capture_output=True,
                              text=True, timeout=30)
        return json.loads(proc.stdout).get("loggedIn") is True
    except Exception:
        return False


def verify_enabled():
    flag = os.environ.get("SEIZUREGUARD_VERIFY")
    if flag is not None:
        return flag not in ("0", "false", "no")
    if os.environ.get("SEIZUREGUARD_BACKEND") == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    return claude_available()


def run_pose_gate(event_dir):
    """Returns 'skip' | 'escalate'. Fail-open: any problem means escalate."""
    pose_python = os.environ.get("SEIZUREGUARD_POSE_PYTHON")
    if not pose_python:
        return "escalate"
    try:
        subprocess.run([pose_python, str(REPO / "src" / "pose_gate.py"),
                        str(event_dir)], capture_output=True, timeout=600)
        gate = json.loads((event_dir / "pose_gate.json").read_text())
        return gate.get("decision", "escalate")
    except Exception as e:
        print(f"[WARN] pose gate failed ({e}); escalating")
        return "escalate"


def run_verify(event_dir):
    """Returns analysis dict or None on failure."""
    try:
        subprocess.run([sys.executable, str(REPO / "src" / "verify_event.py"),
                        str(event_dir)], capture_output=True, timeout=3600)
        return json.loads((event_dir / "analysis.json").read_text())
    except Exception as e:
        print(f"[WARN] verify failed ({e})")
        return None


def peak_frame_path(event_dir, peaks, t_start):
    frames = list((event_dir / "burst").glob("frame_*.jpg")) or \
        list((event_dir / "base").glob("frame_*.jpg"))
    if not frames:
        return None
    if not peaks:
        return frames[0]
    target = peaks[0] - t_start

    def t_of(p):
        m = re.search(r"_t_([0-9.]+)s", p.name)
        return float(m.group(1)) if m else 0.0

    return min(frames, key=lambda p: abs(t_of(p) - target))


def handle_event(ring, motion_history, start, end, out_root, use_verify, name="monitor"):
    t0 = start - PRE_ROLL_SEC
    fb = ring.snapshot(t0, end)
    if len(fb) == 0:
        print("[WARN] event window empty, skipping")
        return None

    stamp = time.strftime("%Y%m%d_%H%M%S")
    event_dir = Path(out_root) / f"event_{stamp}_{name}"
    n = 1
    while event_dir.exists():
        event_dir = Path(out_root) / f"event_{stamp}_{n}_{name}"
        n += 1
    base_saved, burst_saved, peaks = save_event_frames(
        fb, list(motion_history), event_dir, t0, end)
    alert_clip.save_event_video(fb, event_dir, t0, end)
    print(f"✅ Event captured: {event_dir} "
          f"({base_saved} base + {burst_saved} burst, {end - start:.1f}s of motion)")

    decision = run_pose_gate(event_dir)
    if decision == "skip":
        print("[INFO] pose gate: no seizure-like oscillation; not alerting")
        return event_dir

    verdict = run_verify(event_dir) if use_verify else None
    if verdict is not None:
        if not verdict.get("final_abnormal_event"):
            print("[INFO] verify: event negative; not alerting")
            return event_dir
        text = (f"Abnormal motor event detected "
                f"(confidence {verdict.get('final_confidence', 0):.2f}) — {event_dir.name}")
    else:
        text = f"Motion event captured (unverified) — {event_dir.name}"

    # Clip from the window the verifier was most confident about; peak
    # window for unverified events; photo fallback if encoding fails.
    if verdict is not None:
        window = alert_clip.best_batch_window(verdict, alert_clip.event_frames(event_dir))
    else:
        window = alert_clip.peak_window(peaks, t0)
    clip = alert_clip.make_clip(event_dir, window)

    alerts.send_alert(text, photo_path=peak_frame_path(event_dir, peaks, t0),
                      video_path=clip)
    return event_dir


def open_capture(source):
    src = str(source)
    path = Path(src)
    if path.exists():
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        file_mode = True
    elif src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        fps, file_mode = None, False
    else:
        cap = cv2.VideoCapture(src)   # stream URL (rtsp://, http://), wall-clock
        fps, file_mode = None, False
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return cap, file_mode, fps


def _open_live(source):
    """Open a live source, retrying forever — at boot the restreamer may not
    be up yet, and a monitor that dies on a slow dependency never watches."""
    watchdog = StreamWatchdog()
    while True:
        try:
            return open_capture(source)
        except RuntimeError as e:
            now = time.time()
            if "blind_alert" in watchdog.failed(now):
                alerts.send_alert(f"Monitor cannot open {source}: {e}")
            print(f"[WARN] open failed ({e}); retrying in 5s", flush=True)
            time.sleep(5)


def run(source, out_root=OUT_ROOT, log_motion=False, name="monitor"):
    if Path(str(source)).exists():
        cap, file_mode, fps = open_capture(source)
    else:
        cap, file_mode, fps = _open_live(source)
    use_verify = verify_enabled()
    print(f"✅ Monitoring {'file' if file_mode else 'camera'} {source} "
          f"(verify: {'on' if use_verify else 'off'})")

    ring = RingBuffer()
    trigger = MotionTrigger()
    motion_history = deque()
    watchdog = StreamWatchdog()
    moving_flag = MovingFlag(os.environ.get("SEIZUREGUARD_MOVING_FLAG"))
    events = []
    prev_gray = None
    frame_idx = 0
    last_ring_t = None

    while True:
        ret, frame = cap.read()
        if not ret:
            if file_mode:
                break
            now = time.time()
            for action in watchdog.failed(now):
                if action == "reconnect":
                    print(f"[WARN] stream stalled; reconnecting to {source}", flush=True)
                    cap.release()
                    try:
                        cap, _, _ = open_capture(source)
                    except RuntimeError as e:
                        print(f"[WARN] reconnect failed: {e}", flush=True)
                else:
                    alerts.send_alert(
                        f"Monitor blind: no frames from {source} for "
                        f"{int(now - watchdog.stalled_since)}s")
            time.sleep(0.5)
            continue
        watchdog.ok()
        t = frame_idx / fps if file_mode else time.time()
        frame_idx += 1

        frame = resize_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score, glob = 0.0, False
        if prev_gray is not None:
            score = motion_score(prev_gray, gray)
            glob = is_global_change(score)
        if not file_mode and moving_flag.moving(t):
            glob = True
        prev_gray = gray

        if log_motion and frame_idx % 30 == 0:
            print(f"[MOTION] t={t:.1f}s score={score:.2f} global={glob} "
                  f"state={trigger.state}")

        if last_ring_t is None or t - last_ring_t >= RING_INTERVAL:
            jpg = encode_jpg(frame)
            if jpg is not None:
                ring.append(t, jpg)
                last_ring_t = t

        if not glob:
            motion_history.append((t, score))
            while motion_history and motion_history[0][0] < t - BUFFER_SECONDS:
                motion_history.popleft()

        ev = trigger.feed(t, score, glob)
        if ev is not None:
            done = handle_event(ring, motion_history, ev[0], ev[1], out_root,
                                use_verify, name)
            if done is not None:
                events.append(done)

    ev = trigger.flush()
    if ev is not None:
        done = handle_event(ring, motion_history, ev[0], ev[1], out_root,
                            use_verify, name)
        if done is not None:
            events.append(done)

    cap.release()
    return events


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="seizureGuard continuous monitor")
    parser.add_argument("--source", default="0",
                        help="camera index (default 0) or video file path")
    parser.add_argument("--log-motion", action="store_true",
                        help="print motion score stats for threshold calibration")
    parser.add_argument("--name", default="monitor",
                        help="camera tag used in event directory names")
    args = parser.parse_args()
    run(args.source, log_motion=args.log_motion, name=args.name)


if __name__ == "__main__":
    main()
