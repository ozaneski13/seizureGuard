"""Alert delivery: Telegram (with optional photo) when SEIZUREGUARD_TG_TOKEN
and SEIZUREGUARD_TG_CHAT are set, console fallback otherwise. Never raises.

Test your setup: python src/alerts.py "hello"
"""
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TIMEOUT_SEC = 10
MAX_ATTEMPTS = 2

_BOUNDARY = "----seizureGuardBoundary7MA4YWxkTrZu0gW"


def _multipart(fields, file_field, filename, data):
    """Hand-built RFC 2388 multipart body (stdlib has no builder)."""
    lines = []
    for name, value in fields.items():
        lines.append(f"--{_BOUNDARY}\r\n"
                     f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                     f"{value}\r\n")
    body = "".join(lines).encode("utf-8")
    body += (f"--{_BOUNDARY}\r\n"
             f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
             f"Content-Type: application/octet-stream\r\n\r\n").encode("utf-8")
    body += data
    body += f"\r\n--{_BOUNDARY}--\r\n".encode("utf-8")
    return f"multipart/form-data; boundary={_BOUNDARY}", body


def _post(url, data, content_type):
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        return resp.status == 200


def _send_telegram(token, chat, text, photo_path, video_path=None):
    if video_path is not None and Path(video_path).exists():
        content_type, body = _multipart(
            {"chat_id": chat, "caption": text},
            "video", Path(video_path).name, Path(video_path).read_bytes(),
        )
        url = f"https://api.telegram.org/bot{token}/sendVideo"
    elif photo_path is not None and Path(photo_path).exists():
        content_type, body = _multipart(
            {"chat_id": chat, "caption": text},
            "photo", Path(photo_path).name, Path(photo_path).read_bytes(),
        )
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
    else:
        body = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode("utf-8")
        content_type = "application/x-www-form-urlencoded"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
    return _post(url, body, content_type)


def send_alert(text, photo_path=None, video_path=None):
    """Deliver an alert. Telegram when configured, console otherwise.
    Prefers the video clip, falls back to the photo, then to plain text.

    Never raises — an alert failure must not kill the monitor loop.
    Returns True only when a Telegram message was actually delivered.
    """
    token = os.environ.get("SEIZUREGUARD_TG_TOKEN")
    chat = os.environ.get("SEIZUREGUARD_TG_CHAT")

    if not token or not chat:
        print("=" * 60)
        print("🚨 ALERT:", text)
        if video_path is not None:
            print("   clip:", video_path)
        if photo_path is not None:
            print("   frame:", photo_path)
        print("   (set SEIZUREGUARD_TG_TOKEN / SEIZUREGUARD_TG_CHAT for Telegram delivery)")
        print("=" * 60)
        return False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if _send_telegram(token, chat, text, photo_path, video_path):
                return True
        except Exception as e:
            print(f"[WARN] Telegram alert attempt {attempt}/{MAX_ATTEMPTS} failed: {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(2)
        # A failing video upload must not take the whole alert down with it:
        # retry without the clip so at least the photo/text gets through.
        if video_path is not None and attempt == MAX_ATTEMPTS - 1:
            video_path = None
    print("🚨 ALERT (telegram delivery failed):", text)
    return False


if __name__ == "__main__":
    import sys
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    message = " ".join(sys.argv[1:]) or "seizureGuard alert test"
    delivered = send_alert(message)
    print("telegram delivered:", delivered)
