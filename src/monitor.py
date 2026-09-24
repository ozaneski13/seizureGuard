"""Continuous seizure monitor: watches a camera/stream/file through a ring
buffer, turns sustained motion into captured events, and runs each event
through the pose gate and VLM verification before alerting.

Usage: python src/monitor.py --source <index|file|rtsp url> [--name cam]
"""
import argparse
import contextlib
import faulthandler
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
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
STREAM_BLIND_ALERT_SEC = 60.0  # still dead after this long -> alert
STREAM_BLIND_REMIND_SEC = 6 * 3600  # ...repeated this often while it stays dead
STREAM_OPEN_TIMEOUT_MS = 10000   # FFMPEG open/read timeouts: a dead stream
STREAM_READ_TIMEOUT_MS = 10000   # must fail fast, never block the loop
WATCHDOG_PING_SEC = 10.0         # systemd WatchdogSec must be well above this


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


def format_duration(sec):
    if sec < 3600:
        return f"{max(1, round(sec / 60))} min"
    if sec < 86400:
        return f"{sec / 3600:.1f} h"
    return f"{sec / 86400:.1f} days"


class StreamWatchdog:
    """Pure FSM for live-source read health: schedules reconnect attempts and
    'monitor blind' alerts. A blind seizure monitor must say so — silently
    spinning on a dead stream is the one failure the alert channel exists
    for. One alert is not enough: a single message on day one of a 20-day
    blind spell is easy to forget, so it repeats while the stream stays dead,
    and the caller announces recovery."""

    def __init__(self, retry_sec=STREAM_RETRY_SEC, alert_sec=STREAM_BLIND_ALERT_SEC,
                 remind_sec=STREAM_BLIND_REMIND_SEC):
        self.retry_sec = retry_sec
        self.alert_sec = alert_sec
        self.remind_sec = remind_sec
        self.stalled_since = None
        self.last_attempt = None
        self.last_alert = None

    def ok(self, now):
        """Marks the source healthy. Returns how long it was down if a blind
        alert went out for that stall (recovery must be announced), else None."""
        down = None
        if self.last_alert is not None:
            down = now - self.stalled_since
        self.stalled_since = None
        self.last_attempt = None
        self.last_alert = None
        return down

    def failed(self, now):
        """Returns actions to take: 'reconnect' and/or 'blind_alert'."""
        if self.stalled_since is None:
            self.stalled_since = now
            self.last_attempt = now
        actions = []
        if now - self.last_attempt >= self.retry_sec:
            self.last_attempt = now
            actions.append("reconnect")
        if now - self.stalled_since >= self.alert_sec and (
                self.last_alert is None or now - self.last_alert >= self.remind_sec):
            self.last_alert = now
            actions.append("blind_alert")
        return actions


class SystemdWatchdog:
    """Pings systemd's service watchdog (sd_notify "WATCHDOG=1").

    On 2026-09-04 the monitor wedged inside native video code and stayed
    wedged for 20 days: Python never ran again, so no in-process check could
    fire, while systemd kept reporting the unit "active". With WatchdogSec=
    on the unit, systemd restarts a process that stops pinging. No-op when
    NOTIFY_SOCKET is unset (Windows, tests, units without WatchdogSec)."""

    def __init__(self, interval=WATCHDOG_PING_SEC):
        self.addr = os.environ.get("NOTIFY_SOCKET")
        self.interval = interval
        self.last = None

    def ping(self, now):
        """Rate-limited ping from the main loop. Returns True when sent."""
        if not self.addr:
            return False
        if self.last is not None and now - self.last < self.interval:
            return False
        self.last = now
        self._send(b"WATCHDOG=1")
        return True

    def _address(self):
        # "@name" is the abstract-namespace form systemd uses
        return chr(0) + self.addr[1:] if self.addr.startswith("@") else self.addr

    def _send(self, message):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
                sock.sendto(message, self._address())
        except OSError:
            pass

    @contextlib.contextmanager
    def keepalive(self):
        """Keep pinging from a thread while the main loop is blocked in
        handle_event; every wait inside it (pose gate, verify, Telegram)
        carries its own timeout, so this cannot mask a wedge forever."""
        if not self.addr:
            yield
            return
        stop = threading.Event()

        def beat():
            while not stop.wait(self.interval):
                self._send(b"WATCHDOG=1")

        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=2)


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


def verify_probe():
    """One real inference call proving verification actually works.

    `claude auth status` keeps reporting loggedIn after the OAuth token
    expires, so a monitor can log "verify: on" while every call fails with
    401 — that happened for 13 days on the production Pi. Returns
    (ok, reason)."""
    exe = shutil.which("claude")
    if exe is None:
        return False, "claude CLI not found on PATH"
    model = os.environ.get("SEIZUREGUARD_PROBE_MODEL", "claude-haiku-4-5")
    try:
        proc = subprocess.run(
            [exe, "-p", "--model", model, "--max-turns", "1",
             "--no-session-persistence", "Reply with exactly: OK"],
            capture_output=True, text=True, timeout=120)
    except Exception as e:
        return False, str(e)[:200]
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode == 0 and "OK" in out:
        return True, None
    return False, (out[:200] or f"exit {proc.returncode}")


OUTAGE_REMINDER_SEC = 6 * 3600   # how often to repeat "still blind"


class OutageNotifier:
    """Announces a backend outage once, then at most every few hours.

    Alerting per event during an outage is worse than useless: the backend
    fails identically on all of them, so ~30 identical messages a day train
    the owner to ignore the channel that must never be ignored. Silence is
    not an option either, so the outage itself is the alert."""

    def __init__(self, interval=OUTAGE_REMINDER_SEC):
        self.interval = interval
        self.last_notice = None
        self.suppressed = 0

    def should_notify(self, now):
        if self.last_notice is None or now - self.last_notice >= self.interval:
            self.last_notice = now
            n, self.suppressed = self.suppressed, 0
            return True, n
        self.suppressed += 1
        return False, self.suppressed

    def recovered(self):
        was_down = self.last_notice is not None
        self.last_notice = None
        self.suppressed = 0
        return was_down


def alert_text_for(verdict, event_name):
    """Alert text for a verify result, or None when the event was actually
    analyzed and found negative.

    Batches that never returned a verdict are NOT evidence of absence: the
    unanalyzed batch may be the one holding the seizure. Treating a failed
    verification as "negative" is silent blindness — the one failure mode
    this project refuses."""
    if verdict is None:
        return f"Motion event captured (unverified) - {event_name}"
    if verdict.get("final_abnormal_event"):
        conf = float(verdict.get("final_confidence") or 0.0)
        pos = verdict.get("positive_batches")
        total = len(verdict.get("batches") or [])
        # How much of the event looked abnormal is the fastest triage signal:
        # a real seizure flagged 6/7 batches, false alarms flagged 1/7.
        span = f", {pos}/{total} segments" if pos and total else ""
        return (f"Abnormal motor event detected (confidence {conf:.2f}{span})"
                f" - {event_name}")
    if verdict.get("backend_outage"):
        return None          # handled once by OutageNotifier, not per event
    failed = int(verdict.get("failed_batches") or 0)
    if failed:
        total = len(verdict.get("batches") or []) or failed
        return (f"UNVERIFIED motion event - AI verification failed on "
                f"{failed}/{total} batches - {event_name}")
    return None


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


def handle_event(ring, motion_history, start, end, out_root, use_verify,
                 name="monitor", outage_notifier=None):
    outage_notifier = outage_notifier or OutageNotifier()
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
    outage = (verdict or {}).get("backend_outage")
    if outage:
        notify, skipped = outage_notifier.should_notify(time.time())
        if notify:
            extra = f" ({skipped} further events since the last notice)" if skipped else ""
            alerts.send_alert(
                f"seizureGuard: AI verification unavailable{extra} - motion is "
                f"still being recorded but NOT checked. Reason: {outage}")
        else:
            print(f"[WARN] verification unavailable ({outage}); "
                  f"{skipped} events unchecked", flush=True)
    elif outage_notifier.recovered():
        alerts.send_alert("seizureGuard: AI verification is back online.")

    text = alert_text_for(verdict, event_dir.name)
    if text is None:
        print("[INFO] verify: no alert for this event")
        return event_dir

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
        # Stream URL (rtsp://, http://), wall-clock. FFMPEG is forced: when
        # the restreamer answered 404, OpenCV silently fell back to
        # GStreamer, which leaked a pipeline per reconnect and deadlocked in
        # native code (2026-09-04, 20 days blind). Bounded timeouts make a
        # dead stream fail fast so StreamWatchdog can do its job.
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_OPEN_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_READ_TIMEOUT_MS,
        ])
        fps, file_mode = None, False
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return cap, file_mode, fps


def _open_live(source, systemd_watchdog=None):
    """Open a live source, retrying forever — at boot the restreamer may not
    be up yet, and a monitor that dies on a slow dependency never watches."""
    watchdog = StreamWatchdog()
    while True:
        if systemd_watchdog is not None:
            systemd_watchdog.ping(time.monotonic())
        try:
            opened = open_capture(source)
        except RuntimeError as e:
            now = time.time()
            if "blind_alert" in watchdog.failed(now):
                alerts.send_alert(
                    f"Monitor cannot open {source} for "
                    f"{format_duration(now - watchdog.stalled_since)}: {e}")
            print(f"[WARN] open failed ({e}); retrying in 5s", flush=True)
            time.sleep(5)
            continue
        down = watchdog.ok(time.time())
        if down is not None:
            alerts.send_alert(f"Monitor recovered: {source} opened after "
                              f"{format_duration(down)} blind")
        return opened


def run(source, out_root=OUT_ROOT, log_motion=False, name="monitor"):
    systemd_watchdog = SystemdWatchdog()
    if Path(str(source)).exists():
        cap, file_mode, fps = open_capture(source)
    else:
        cap, file_mode, fps = _open_live(source, systemd_watchdog)
    use_verify = verify_enabled()
    print(f"✅ Monitoring {'file' if file_mode else 'camera'} {source} "
          f"(verify: {'on' if use_verify else 'off'})")
    if use_verify and os.environ.get("SEIZUREGUARD_BACKEND", "claude-cli") == "claude-cli":
        ok, reason = verify_probe()
        if not ok:
            warning = (f"seizureGuard: AI verification is DOWN ({reason}). "
                       "Motion alerts will be sent unverified until it is fixed.")
            print("[WARN]", warning, flush=True)
            alerts.send_alert(warning)

    ring = RingBuffer()
    trigger = MotionTrigger()
    motion_history = deque()
    watchdog = StreamWatchdog()
    moving_flag = MovingFlag(os.environ.get("SEIZUREGUARD_MOVING_FLAG"))
    outage_notifier = OutageNotifier()
    events = []
    prev_gray = None
    frame_idx = 0
    last_ring_t = None

    while True:
        systemd_watchdog.ping(time.monotonic())
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
                        f"{format_duration(now - watchdog.stalled_since)}")
            time.sleep(0.5)
            continue
        t = frame_idx / fps if file_mode else time.time()
        down = watchdog.ok(time.time())
        if down is not None:
            alerts.send_alert(f"Monitor recovered: frames from {source} again "
                              f"after {format_duration(down)} blind")
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
            with systemd_watchdog.keepalive():
                done = handle_event(ring, motion_history, ev[0], ev[1], out_root,
                                    use_verify, name, outage_notifier)
            if done is not None:
                events.append(done)

    ev = trigger.flush()
    if ev is not None:
        with systemd_watchdog.keepalive():
            done = handle_event(ring, motion_history, ev[0], ev[1], out_root,
                                use_verify, name, outage_notifier)
        if done is not None:
            events.append(done)

    cap.release()
    return events


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    # systemd's watchdog kills a wedged monitor with SIGABRT; this makes that
    # kill print every thread's stack to the journal, so the next wedge can be
    # diagnosed from its stack instead of guessed.
    faulthandler.enable()
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
