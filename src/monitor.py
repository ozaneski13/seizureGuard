"""Continuous seizure monitor: watches a camera/stream/file through a ring
buffer, turns sustained motion into captured events, and runs each event
through the pose gate and VLM verification before alerting. Verification
runs on a worker thread, so the read loop never stops watching.

Usage: python src/monitor.py --source <index|file|rtsp url> [--name cam]
"""
import argparse
import faulthandler
import json
import os
import queue
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
from verify_event import BATCH_SIZE, frame_time  # noqa: E402

OUT_ROOT = Path("data/events")
EVENT_META = "event_meta.json"   # what processing needs, written at capture
HANDLED_MARKER = "handled.json"  # written when processing finished

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
STREAM_BLIND_RETRY_SEC = 60.0  # ...or this soon when the last one was not delivered
STREAM_OPEN_TIMEOUT_MS = 10000   # FFMPEG open/read timeouts: a dead stream
STREAM_READ_TIMEOUT_MS = 10000   # must fail fast, never block the loop
WATCHDOG_PING_SEC = 10.0         # systemd WatchdogSec must be well above this

# Event worker
# One event's processing is bounded by its waits: pose gate 600 s + verify
# 3600 s, plus clip encoding and up to two Telegram sends (2 attempts x 10 s
# timeouts each), which take well under 20 min. Past this deadline the worker
# is wedged, not slow, and the main loop stops pinging systemd so its
# watchdog restarts the monitor.
HANDLE_DEADLINE_SEC = 5400
EVENT_BACKLOG_ALERT = 10         # queued events before the owner is told
EVENT_BACKLOG_REMIND_SEC = 6 * 3600  # ...told again this often while it lasts


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
    and the caller announces recovery. An alert the caller could not deliver
    is retried soon instead of waiting out the reminder interval."""

    def __init__(self, retry_sec=STREAM_RETRY_SEC, alert_sec=STREAM_BLIND_ALERT_SEC,
                 remind_sec=STREAM_BLIND_REMIND_SEC,
                 undelivered_retry_sec=STREAM_BLIND_RETRY_SEC):
        self.retry_sec = retry_sec
        self.alert_sec = alert_sec
        self.remind_sec = remind_sec
        self.undelivered_retry_sec = undelivered_retry_sec
        self.stalled_since = None
        self.last_attempt = None
        self.last_alert = None
        self.next_alert = None

    def ok(self, now):
        """Marks the source healthy. Returns how long it was down if a blind
        alert went out for that stall (recovery must be announced), else None."""
        down = None
        if self.last_alert is not None:
            down = now - self.stalled_since
        self.stalled_since = None
        self.last_attempt = None
        self.last_alert = None
        self.next_alert = None
        return down

    def alert_undelivered(self, now):
        """The last blind alert never reached the owner: try again soon."""
        self.next_alert = now + self.undelivered_retry_sec

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
                self.next_alert is None or now >= self.next_alert):
            self.last_alert = now
            self.next_alert = now + self.remind_sec
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


class EventWorker:
    """Processes captured events one at a time on a background thread.

    Verification takes minutes; run inline, it stopped the read loop for
    all of them, so a seizure starting meanwhile was never seen and no alert
    said the monitor was not watching. The queue never drops an event, and
    holds only its dir: the frames are on disk. An event whose alert was not
    delivered is retried every retry_sec, between queued events, until it
    is. Only the main loop pings systemd, and it stops once stuck() says one
    event has run past its deadline, so a wedged worker still gets
    restarted."""

    def __init__(self, process, deadline=HANDLE_DEADLINE_SEC,
                 backlog_alert=EVENT_BACKLOG_ALERT,
                 backlog_remind=EVENT_BACKLOG_REMIND_SEC,
                 retry_sec=STREAM_BLIND_RETRY_SEC):
        self.process = process              # process(event_dir) -> False while undelivered
        self.deadline = deadline
        self.backlog_alert = backlog_alert
        self.backlog_remind = backlog_remind
        self.retry_sec = retry_sec
        self.queue = queue.Queue()
        self.busy_since = None              # set by the worker, read by the main loop
        self.retries = {}                   # event_dir -> due time; worker only
        self.backlog_last = None            # (time, size) of the last backlog alert;
        self._backlog_undo = None           # these three main thread only
        self._backlog_retry_at = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, event_dir, now=None):
        """Queues an event. Returns True when the caller should send a
        backlog alert: the backlog first went past backlog_alert, has doubled
        since the last alert, or has lasted backlog_remind since it; never
        sooner than retry_sec after one that was not delivered."""
        now = time.monotonic() if now is None else now
        if self.queue.empty():
            self.backlog_last = None
        self.queue.put(event_dir)
        size = self.queue.qsize()
        last = self.backlog_last
        retry_at = self._backlog_retry_at
        if size > self.backlog_alert and (retry_at is None or now >= retry_at) and (
                last is None or size >= 2 * last[1] or now - last[0] >= self.backlog_remind):
            self._backlog_undo = last
            self._backlog_retry_at = None
            self.backlog_last = (now, size)
            return True
        return False

    def backlog_undelivered(self, now):
        """The backlog alert never reached the owner: a submit while the
        backlog lasts tries again, retry_sec from now. Not sooner: a failed
        send blocks the main thread for ~22 s, and the restart sweep submits
        dozens of events in a row."""
        self.backlog_last = self._backlog_undo
        self._backlog_retry_at = now + self.retry_sec

    def stuck(self, now):
        since = self.busy_since             # read once: the worker may clear it
        return since is not None and now - since > self.deadline

    def close(self):
        """Processes everything queued, then stops the thread."""
        self.queue.put(None)
        self.thread.join()

    def _loop(self):
        while True:
            now = time.monotonic()
            for event_dir in [d for d, due in self.retries.items() if due <= now]:
                del self.retries[event_dir]
                self._run(event_dir)
            timeout = None
            if self.retries:
                timeout = max(0.0, min(self.retries.values()) - time.monotonic())
            try:
                event_dir = self.queue.get(timeout=timeout)
            except queue.Empty:
                continue
            try:
                if event_dir is None:
                    return
                self._run(event_dir)
            finally:
                self.queue.task_done()

    def _run(self, event_dir):
        self.busy_since = time.monotonic()
        try:
            delivered = self.process(event_dir) is not False
        except Exception as e:
            # Recall first: an event that crashed its processing is an
            # event nobody checked, so it still alerts, and the alert is
            # recorded like any other so it is retried, not re-verified.
            name = Path(event_dir).name
            print(f"[ERROR] processing {name} failed: {e!r}", flush=True)
            text = f"UNVERIFIED motion event - processing failed ({e}) - {name}"
            delivered = deliver(text)
            try:
                mark_handled(event_dir, alerted=True, delivered=delivered, text=text)
            except Exception as e2:
                print(f"[ERROR] could not record {name} as handled: {e2!r}", flush=True)
        finally:
            self.busy_since = None
        if not delivered:
            print(f"[WARN] alert for {Path(event_dir).name} not delivered; "
                  f"retrying in {self.retry_sec:.0f}s", flush=True)
            self.retries[event_dir] = time.monotonic() + self.retry_sec


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
    not an option either, so the outage itself is the alert. A notice or
    recovery message that was not delivered gives its slot back."""

    def __init__(self, interval=OUTAGE_REMINDER_SEC):
        self.interval = interval
        self.last_notice = None
        self.suppressed = 0
        self._undo = (None, 0)

    def should_notify(self, now):
        if self.last_notice is None or now - self.last_notice >= self.interval:
            # if this notice is lost, its event is one more unchecked one
            self._undo = (self.last_notice, self.suppressed + 1)
            self.last_notice = now
            n, self.suppressed = self.suppressed, 0
            return True, n
        self.suppressed += 1
        return False, self.suppressed

    def recovered(self):
        self._undo = (self.last_notice, self.suppressed)
        # a lost first notice leaves no last_notice, only its unchecked event
        was_down = self.last_notice is not None or self.suppressed > 0
        self.last_notice = None
        self.suppressed = 0
        return was_down

    def undelivered(self):
        """The message just sent for should_notify() or recovered() never
        reached the owner: restore the state so the next event retries."""
        self.last_notice, self.suppressed = self._undo


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
        batches = verdict.get("batches") or []
        total = len(batches)
        # How much of the event looked abnormal is the fastest triage signal:
        # a real seizure flagged 6/7 batches, false alarms flagged 1/7.
        span = f", {pos}/{total} segments" if pos and total else ""
        positives = [b for b in batches if b.get("abnormal_event") is True]
        if positives and all(b.get("salvaged") for b in positives):
            # A salvaged positive carries confidence 0.0 as a marker, not a
            # measurement; printed, it reads like a false alarm.
            confidence = "confidence unknown, the model reply was partly unreadable"
        else:
            confidence = f"confidence {conf:.2f}"
        return f"Abnormal motor event detected ({confidence}{span}) - {event_name}"
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


def deliver(text, **media):
    """send_alert, counting console-only mode as delivered: with no Telegram
    configured there is nothing to retry."""
    return bool(alerts.send_alert(text, **media)) or not alerts.telegram_configured()


def failed_batch_window(verdict, frames, batch_size=BATCH_SIZE):
    """Time window of the first batch that never got a verdict, or None."""
    for i, b in enumerate((verdict or {}).get("batches") or []):
        if b.get("abnormal_event") is None:
            chunk = frames[i * batch_size:(i + 1) * batch_size]
            if chunk:
                return alert_clip.clamp_window(frame_time(chunk[0]), frame_time(chunk[-1]))
            return None
    return None


def clip_window(verdict, frames, peaks, t0):
    """Clip from the window the verifier was most confident about; for an
    unverified alert, the batch it failed on (the part nobody checked), else
    the first motion peak."""
    window = None
    if verdict is not None:
        window = (alert_clip.best_batch_window(verdict, frames)
                  or failed_batch_window(verdict, frames))
    return window or alert_clip.peak_window(peaks, t0)


def capture_event(ring, motion_history, start, end, out_root, name="monitor"):
    """Main-thread half of an event: snapshot the ring and write the frames,
    EVENT_META and the full-rate ring JPEGs, so the rest runs from disk
    alone and nothing waits in memory. Kept cheap (no encoding), as the read
    loop waits for it. Returns event_dir or None."""
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
    (event_dir / EVENT_META).write_text(json.dumps(
        {"t0": t0, "start": start, "end": end, "peaks": peaks}), encoding="utf-8")
    alert_clip.spill_ring(fb, event_dir)
    print(f"✅ Event captured: {event_dir} "
          f"({base_saved} base + {burst_saved} burst, {end - start:.1f}s of motion)")
    return event_dir


def read_event_meta(event_dir):
    """EVENT_META, or safe fallbacks for an event captured before it
    existed: no peaks means the first frame as photo and no peak clip."""
    try:
        meta = json.loads((Path(event_dir) / EVENT_META).read_text(encoding="utf-8"))
        return {"t0": float(meta["t0"]),
                "peaks": [float(p) for p in meta.get("peaks") or []]}
    except Exception:
        return {"t0": 0.0, "peaks": []}


def mark_handled(event_dir, alerted, delivered=None, text=None, photo=None, video=None):
    """Writes HANDLED_MARKER. An alert is recorded with its text and media
    (relative to the event dir), so an undelivered one can be resent as it
    was, without verifying again."""
    event_dir = Path(event_dir)
    record = {"handled_at": time.time(), "alerted": alerted, "delivered": delivered}
    if alerted:
        record.update(
            text=text,
            photo=photo and Path(photo).relative_to(event_dir).as_posix(),
            video=video and Path(video).relative_to(event_dir).as_posix())
    tmp = event_dir / (HANDLED_MARKER + ".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    os.replace(tmp, event_dir / HANDLED_MARKER)


def read_handled(event_dir):
    """HANDLED_MARKER's record, or None when there is none or it cannot be
    read (then the event counts as unfinished)."""
    try:
        record = json.loads((Path(event_dir) / HANDLED_MARKER).read_text(encoding="utf-8"))
    except Exception:
        return None
    return record if isinstance(record, dict) else None


def undelivered(record):
    return bool(record.get("alerted")) and not record.get("delivered")


def read_analysis(event_dir):
    """analysis.json from an earlier run, or None when there is none or it
    cannot be read."""
    try:
        verdict = json.loads((Path(event_dir) / "analysis.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    return verdict if isinstance(verdict, dict) else None


def finished(verdict):
    """A finished verdict: a positive, or an analysis with no failed batch
    and no outage. Anything else is verified (again). A finished verdict is
    never asked for twice: the model is nondeterministic, and a second
    answer could silence a positive the first one found."""
    return verdict is not None and bool(verdict.get("final_abnormal_event") or not (
        verdict.get("failed_batches") or verdict.get("backend_outage")))


def redeliver(event_dir, handled):
    """Resends a recorded, undelivered alert as it was. Returns delivered."""
    photo = event_dir / handled["photo"] if handled.get("photo") else None
    video = event_dir / handled["video"] if handled.get("video") else None
    if not deliver(handled["text"], photo_path=photo, video_path=video):
        return False
    mark_handled(event_dir, True, True, handled["text"], photo, video)
    return True


def process_event(event_dir, use_verify, outage_notifier):
    """Worker half of an event: event video, pose gate, verify, alert, then
    HANDLED_MARKER. Reads everything from disk, so a restart can finish an
    event the previous process never did: an undelivered alert is only
    resent, and a finished verdict is reused. Returns False while its alert
    is undelivered."""
    handled = read_handled(event_dir)
    resend = False
    if handled is not None:
        if not undelivered(handled):
            return True
        if handled.get("text"):
            return redeliver(event_dir, handled)
        # A marker from before the text was recorded: its alert was due, so
        # it is built again from what is on disk, never gated or verified.
        resend = True

    meta = read_event_meta(event_dir)
    t0, peaks = meta["t0"], meta["peaks"]
    alert_clip.save_event_video(event_dir, t0)

    verdict = read_analysis(event_dir)
    if resend:
        print(f"[INFO] {event_dir.name}: resending its undelivered alert", flush=True)
    elif finished(verdict):
        print(f"[INFO] {event_dir.name}: reusing its finished verdict", flush=True)
    else:
        decision = run_pose_gate(event_dir)
        if decision == "skip":
            print("[INFO] pose gate: no seizure-like oscillation; not alerting")
            mark_handled(event_dir, alerted=False)
            return True

        verdict = run_verify(event_dir) if use_verify else None
        outage = (verdict or {}).get("backend_outage")
        if outage:
            notify, skipped = outage_notifier.should_notify(time.time())
            if notify:
                extra = f" ({skipped} further events since the last notice)" if skipped else ""
                if not deliver(
                        f"seizureGuard: AI verification unavailable{extra} - motion is "
                        f"still being recorded but NOT checked. Reason: {outage}"):
                    outage_notifier.undelivered()
            else:
                print(f"[WARN] verification unavailable ({outage}); "
                      f"{skipped} events unchecked", flush=True)
        # No analysis at all (verify timed out or crashed) is no sign of health.
        elif verdict is not None and outage_notifier.recovered():
            if not deliver("seizureGuard: AI verification is back online."):
                outage_notifier.undelivered()

    text = alert_text_for(verdict, event_dir.name)
    if text is None and resend:
        text = alert_text_for(None, event_dir.name)
    if text is None:
        print("[INFO] verify: no alert for this event")
        mark_handled(event_dir, alerted=False)
        return True

    # make_clip returns None when encoding fails; the photo still goes out.
    window = clip_window(verdict, alert_clip.event_frames(event_dir), peaks, t0)
    clip = alert_clip.make_clip(event_dir, window)
    photo = peak_frame_path(event_dir, peaks, t0)
    delivered = deliver(text, photo_path=photo, video_path=clip)
    mark_handled(event_dir, alerted=True, delivered=delivered, text=text,
                 photo=photo, video=clip)
    return delivered


def unhandled_events(out_root, name):
    """This monitor's unfinished events (both monitors share out_root),
    oldest first: no HANDLED_MARKER, or one recording an undelivered alert.
    Only dirs with EVENT_META count, whatever their age; the code from
    before it finished its events inline, so their dirs are never swept."""
    pattern = re.compile(rf"^event_(\d{{8}}_\d{{6}})(?:_(\d+))?_{re.escape(name)}$")
    found = []
    try:
        dirs = list(Path(out_root).iterdir())
    except OSError:
        return []
    for d in dirs:
        m = pattern.match(d.name)
        try:
            if m is None or not d.is_dir() or not (d / EVENT_META).exists():
                continue
        except OSError:
            continue
        handled = read_handled(d)
        if handled is not None and not undelivered(handled):
            continue
        found.append(((m.group(1), int(m.group(2) or 0)), d))
    return [d for _, d in sorted(found)]


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


def _open_live(source, keepalive=None):
    """Open a live source, retrying forever — at boot the restreamer may not
    be up yet, and a monitor that dies on a slow dependency never watches.
    keepalive() runs once per attempt (the caller's systemd ping)."""
    watchdog = StreamWatchdog()
    while True:
        if keepalive is not None:
            keepalive()
        try:
            opened = open_capture(source)
        except RuntimeError as e:
            now = time.time()
            if "blind_alert" in watchdog.failed(now) and not deliver(
                    f"Monitor cannot open {source} for "
                    f"{format_duration(now - watchdog.stalled_since)}: {e}"):
                watchdog.alert_undelivered(now)
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
    file_mode = Path(str(source)).exists()
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
    outage_notifier = OutageNotifier()     # used by the worker thread only
    worker = EventWorker(
        lambda event_dir: process_event(event_dir, use_verify, outage_notifier))
    events = []
    prev_gray = None
    frame_idx = 0
    last_ring_t = None
    wedge_reported = False

    def keepalive():
        # Only this thread pings systemd, and not for a wedged worker.
        nonlocal wedge_reported
        now_mono = time.monotonic()
        if not worker.stuck(now_mono):
            wedge_reported = False
            systemd_watchdog.ping(now_mono)
        elif not wedge_reported:
            wedge_reported = True
            print(f"[ERROR] one event has been processing for over "
                  f"{HANDLE_DEADLINE_SEC}s; the worker is wedged. No longer "
                  "pinging systemd, so its watchdog restarts the monitor.", flush=True)

    def enqueue(event_dir):
        if worker.submit(event_dir, time.monotonic()):
            waiting = worker.queue.qsize()
            print(f"[WARN] {waiting} events waiting for verification", flush=True)
            if not deliver(
                    f"seizureGuard: {waiting} motion events are waiting for "
                    "verification - their alerts are delayed. The camera is still "
                    "being watched."):
                worker.backlog_undelivered(time.monotonic())

    def capture(ev):
        event_dir = capture_event(ring, motion_history, ev[0], ev[1], out_root, name)
        if event_dir is None:
            return
        events.append(event_dir)
        enqueue(event_dir)

    if not file_mode:
        # A restart (deploy, token setup, watchdog kill, power cut) must not
        # drop the event it interrupted or an alert that never arrived,
        # whatever their age and whether verify is on. Swept before the
        # stream is opened: the camera may stay down for hours. A backlog
        # alert can block here on a slow send, so the sweep pings systemd.
        for event_dir in unhandled_events(out_root, name):
            keepalive()
            state = "alert not delivered" if read_handled(event_dir) else "not finished"
            print(f"[INFO] re-queued {event_dir.name} ({state})", flush=True)
            enqueue(event_dir)

    if file_mode:
        cap, _, fps = open_capture(source)
    else:
        cap, _, fps = _open_live(source, keepalive)

    while True:
        keepalive()
        ret, frame = cap.read()
        if not ret:
            if file_mode:
                break
            now = time.time()
            if watchdog.stalled_since is None:
                # First failed read of a stall: capture the motion in flight
                # now, before the frames after the stall evict it from the ring.
                ev = trigger.flush()
                if ev is not None:
                    capture(ev)
            for action in watchdog.failed(now):
                if action == "reconnect":
                    print(f"[WARN] stream stalled; reconnecting to {source}", flush=True)
                    cap.release()
                    try:
                        cap, _, _ = open_capture(source)
                    except RuntimeError as e:
                        print(f"[WARN] reconnect failed: {e}", flush=True)
                elif not deliver(f"Monitor blind: no frames from {source} for "
                                 f"{format_duration(now - watchdog.stalled_since)}"):
                    watchdog.alert_undelivered(now)
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
            capture(ev)

    ev = trigger.flush()
    if ev is not None:
        capture(ev)

    cap.release()
    worker.close()
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
