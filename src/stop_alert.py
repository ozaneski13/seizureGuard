"""Abnormal-stop alert for a monitor unit, run by systemd's ExecStopPost.

The first abnormal stop (watchdog kill, crash, signal) is announced at once.
While stops keep coming - a monitor that crash-loops restarts every ~20 s -
at most one message per STOP_ALERT_INTERVAL_SEC goes out, and it says how
many stops went unreported since the last one. A message Telegram did not
take is not counted as sent, so the next stop tries again. A clean
stop/restart (SERVICE_RESULT=success) stays silent.

Usage, from the unit: python3 src/stop_alert.py %N
Deliberately does not import monitor: a stop hook must not pull in cv2.
"""
import json
import os
import sys
import time
from pathlib import Path

import alerts

STOP_ALERT_INTERVAL_SEC = 30 * 60
STATE_DIR = Path("data")     # relative to the unit's WorkingDirectory


def should_send(state, now, interval=STOP_ALERT_INTERVAL_SEC):
    """Returns (send now?, stops this message covers, this one included)."""
    stops = int(state.get("pending") or 0) + 1
    last = state.get("last_sent")
    return last is None or now - last >= interval, stops


def message(unit, result, code, status, stops):
    unreported = stops - 1
    more = (f" ({unreported} more abnormal stop{'s' if unreported != 1 else ''} "
            "since the last message)") if unreported else ""
    return (f"seizureGuard {unit} stopped abnormally ({result}, {code}/{status}){more}; "
            "systemd is restarting it - the camera is not watched until it is back")


def load_state(path):
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}        # missing or unreadable: announce, don't guess


def save_state(path, state):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        print(f"[WARN] stop alert state not saved ({e})")


def main(argv, env=os.environ, now=None, state_dir=STATE_DIR):
    result = env.get("SERVICE_RESULT", "")
    if result == "success":
        return 0
    unit = argv[1] if len(argv) > 1 else "monitor"
    now = time.time() if now is None else now
    path = Path(state_dir) / f"stop-alert-{unit}.json"
    state = load_state(path)
    send, stops = should_send(state, now)
    last_sent = state.get("last_sent")
    if send:
        text = message(unit, result, env.get("EXIT_CODE", "?"), env.get("EXIT_STATUS", "?"), stops)
        # console-only mode counts as delivered: there is nothing to retry
        if alerts.send_alert(text) or not alerts.telegram_configured():
            last_sent, stops = now, 0
    else:
        print(f"[INFO] {unit} stopped abnormally ({result}); {stops} stop(s) since the "
              f"last message, next message after {STOP_ALERT_INTERVAL_SEC // 60} min")
    save_state(path, {"last_sent": last_sent, "pending": stops})
    return 0


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main(sys.argv))
