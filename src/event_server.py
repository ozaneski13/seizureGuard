"""Ag uzerinden olay arsivi: kaydedilmis nobet adayi videolarini tarih/saat,
kamera ve model kararina gore listeler, kucuk resim uretir ve tarayicida
oynatir.

    python3 src/event_server.py [--port 8090] [--root data/events]

Bagimlilik yok (PEP 668 nedeniyle Pi'de pip kapali); thumbnail icin ffmpeg
kullanilir, yoksa liste yine calisir sadece kucuk resim cikmaz.

ONEMLI: analysis.json icindeki confidence, "nobet olasiligi" DEGIL, modelin
kendi kararindan ne kadar emin oldugudur. 0.95 confidence + negatif karar =
"nobet olmadigindan cok eminim". Bu yuzden varsayilan siralama once karara,
sonra zamana gore yapilir; ham confidence siralamasi ayri bir secenektir.

final_abnormal_event=false yalnizca "ANALIZ EDILEN hicbir batch pozitif
degil" demektir. Batch'lerden biri hic dogrulanamadiysa (failed_batches /
backend_outage) olay negatif degil KARARSIZ gosterilir: bakilmayan parca
nobetin kendisi olabilir.
"""
import argparse
import html
import json
import mimetypes
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

REPO = Path(__file__).resolve().parent.parent
NAME_RE = re.compile(r"^event_(\d{8})_(\d{6})(?:_\d+)?_(.+)$")

_index_lock = threading.Lock()
_index_cache = {"stamp": 0.0, "items": []}
INDEX_TTL = 20.0


def parse_name(name):
    """event_20260903_183211_c700-pi -> (datetime, 'c700-pi'). Elle eklenmis
    olaylar bu kaliba uymaz; onlarda klasor mtime'ina duseriz."""
    m = NAME_RE.match(name)
    if not m:
        return None, name.replace("event_", "", 1)
    day, clock, cam = m.groups()
    try:
        return datetime.strptime(day + clock, "%Y%m%d%H%M%S"), cam
    except ValueError:
        return None, cam


def read_analysis(event_dir):
    path = event_dir / "analysis.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_signs(analysis):
    """Pozitif batch'lerde gozlenen isaretleri benzersizlestir."""
    signs = []
    for b in (analysis.get("batches") or []):
        if b.get("abnormal_event") is not True:
            continue
        for s in (b.get("observed_signs") or []):
            if s.get("present") and s.get("sign") not in signs:
                signs.append(s.get("sign"))
    return signs


def peak_note(analysis):
    """Kartta baglam icin kisa not: once pozitif batch'lerin en guvenlisi,
    pozitif yoksa analiz edilmis batch'lerin en guvenlisi. Confidence
    modelin kendi kararindan emin olmasidir; ham en yuksek, pozitif bir
    olayi 0.95'lik negatif batch'in notuyla aciklardi."""
    batches = [b for b in (analysis.get("batches") or []) if isinstance(b, dict)]
    pool = ([b for b in batches if b.get("abnormal_event") is True]
            or [b for b in batches if b.get("abnormal_event") is not None])
    best = max(pool, key=lambda b: float(b.get("confidence") or 0.0), default=None)
    if not best:
        return ""
    # confirm-model note; screen notes only exist in pre-2026-09-24 events
    note = best.get("note")
    if not isinstance(note, str) or not note:
        screen = best.get("screen_verdict")
        note = screen.get("note") if isinstance(screen, dict) else None
    return note[:300] if isinstance(note, str) else ""


def scan_events(root):
    items = []
    for d in sorted(root.glob("event_*")):
        if not d.is_dir():
            continue
        when, cam = parse_name(d.name)
        if when is None:
            when = datetime.fromtimestamp(d.stat().st_mtime)
        a = read_analysis(d)
        video = d / "event.mp4"
        if not video.exists():
            clip = d / "alert_clip.mp4"
            video = clip if clip.exists() else None
        verdict = a.get("final_abnormal_event")
        failed = int(a.get("failed_batches") or 0)
        items.append({
            "id": d.name,
            "camera": cam,
            "when": when.isoformat(timespec="seconds"),
            "when_ts": when.timestamp(),
            "verdict": verdict,
            # verdict True ise dogrulanamayan batch karari degistirmez;
            # degilse eksik bakilmis olay negatif sayilamaz
            "unchecked": verdict is not True and (
                verdict is None or failed > 0 or bool(a.get("backend_outage"))),
            "failed_batches": failed,
            "total_batches": len(a.get("batches") or []),
            "confidence": float(a.get("final_confidence") or 0.0),
            "reason": a.get("final_reason") or "",
            "signs": collect_signs(a),
            "note": peak_note(a),
            "frames": int(a.get("num_frames") or 0),
            "has_video": video is not None,
            "size_mb": round(video.stat().st_size / 1e6, 1) if video else 0.0,
        })
    return items


def drop_orphan_thumbs(thumbs_dir, event_ids):
    """Budanan olaylarin kucuk resimlerini sil; yoksa onbellek sonsuza dek buyur."""
    if thumbs_dir is None or not thumbs_dir.is_dir():
        return 0
    removed = 0
    for t in thumbs_dir.glob("*.jpg"):
        if t.stem not in event_ids:
            try:
                t.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def get_index(root, thumbs=None):
    with _index_lock:
        if time.time() - _index_cache["stamp"] < INDEX_TTL:
            return _index_cache["items"]
        items = scan_events(root)
        drop_orphan_thumbs(thumbs, {e["id"] for e in items})
        _index_cache.update(stamp=time.time(), items=items)
        return items


def sort_items(items, mode):
    if mode == "conf":
        return sorted(items, key=lambda e: (e["confidence"], e["when_ts"]), reverse=True)
    if mode == "time":
        return sorted(items, key=lambda e: e["when_ts"], reverse=True)
    # varsayilan: once pozitifler, sonra kararsizlar, sonra en yeni
    return sorted(items, key=lambda e: (e["verdict"] is True, e["unchecked"], e["when_ts"]),
                  reverse=True)


def video_path(root, event_id):
    d = root / event_id
    for candidate in ("event.mp4", "alert_clip.mp4"):
        p = d / candidate
        if p.exists():
            return p
    return None


def thumb_path(root, event_id, thumbs_dir):
    """Kucuk resmi uret ve onbellege al. Kaynak: burst ortasi kare (olayin
    ilginc yeri), yoksa base ortasi, yoksa videodan tek kare."""
    out = thumbs_dir / (event_id + ".jpg")
    if out.exists():
        return out
    d = root / event_id
    if not d.is_dir():
        return None
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    src = None
    for sub in ("burst", "base"):
        frames = sorted((d / sub).glob("frame_*.jpg")) if (d / sub).is_dir() else []
        if frames:
            src = frames[len(frames) // 2]
            break

    try:
        if src is not None:
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src),
                   "-vf", "scale=320:-1", str(out)]
        else:
            vid = video_path(root, event_id)
            if vid is None:
                return None
            cmd = ["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", str(vid),
                   "-frames:v", "1", "-vf", "scale=320:-1", str(out)]
        subprocess.run(cmd, timeout=30, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out if out.exists() else None
    except Exception:
        return None


def verdict_label(e):
    if e["verdict"] is True:
        return "POZITIF"
    return "KARARSIZ" if e["unchecked"] else "negatif"


def badge(e):
    cls = {"POZITIF": "pos", "KARARSIZ": "unk", "negatif": "neg"}[verdict_label(e)]
    return '<span class="b ' + cls + '">' + verdict_label(e) + '</span>'


def unchecked_text(e):
    """'2/7 dogrulanamadi' — yalnizca pozitif olmayan, eksik bakilmis olayda."""
    if not e["unchecked"] or not e["failed_batches"]:
        return ""
    total = e["total_batches"] or e["failed_batches"]
    return "%d/%d dogrulanamadi" % (e["failed_batches"], total)


def card_html(e):
    when = e["when"].replace("T", " ")
    esc = html.escape
    parts = [
        '<a class="card" href="/event/' + esc(e["id"]) + '">',
        '<div class="thumbwrap">',
        '<img loading="lazy" src="/thumb/' + esc(e["id"]) + '" alt="">',
        '<span class="conf">%.2f</span>' % e["confidence"],
        '</div>',
        '<div class="meta">',
        '<div class="line1">' + badge(e) + '<span class="when">' + esc(when) + '</span></div>',
        '<div class="line2">' + esc(e["camera"]),
        (' &middot; ' + esc(unchecked_text(e))) if unchecked_text(e) else "",
        (' &middot; ' + esc(", ".join(e["signs"][:3]))) if e["signs"] else "",
        '</div>',
        '</div></a>',
    ]
    return "".join(parts)


STYLE = """
:root { color-scheme: dark; }
body { margin:0; padding:18px; background:#101214; color:#e8e8e8;
       font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
h1 { font-size:16px; margin:0 0 4px; }
.sub { color:#8b9199; font-size:12px; margin-bottom:14px; }
.bar { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px; align-items:center; }
.bar a { color:#c9d1d9; text-decoration:none; border:1px solid #2d333b; padding:5px 10px;
         border-radius:6px; font-size:12px; background:#161b22; }
.bar a.on { background:#2f6feb; border-color:#2f6feb; color:#fff; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:14px; }
.card { display:block; background:#161b22; border:1px solid #262c33; border-radius:8px;
        overflow:hidden; text-decoration:none; color:inherit; }
.card:hover { border-color:#2f6feb; }
.thumbwrap { position:relative; background:#000; aspect-ratio:16/9; }
.thumbwrap img { width:100%; height:100%; object-fit:cover; display:block; }
.conf { position:absolute; right:6px; bottom:6px; background:rgba(0,0,0,.78);
        padding:2px 7px; border-radius:5px; font-size:12px; font-variant-numeric:tabular-nums; }
.meta { padding:9px 10px; }
.line1 { display:flex; align-items:center; gap:8px; }
.line2 { color:#8b9199; font-size:12px; margin-top:4px; }
.when { font-size:12px; color:#c9d1d9; }
.b { font-size:10px; letter-spacing:.4px; padding:2px 6px; border-radius:4px; }
.pos { background:#b3271e; color:#fff; }
.neg { background:#21262d; color:#8b9199; }
.unk { background:#7a5c00; color:#fff; }
.note { color:#8b9199; font-size:13px; line-height:1.5; max-width:70ch; }
video { width:100%; max-width:900px; background:#000; border-radius:8px; }
table { border-collapse:collapse; font-size:13px; margin-top:10px; }
td { padding:3px 14px 3px 0; color:#c9d1d9; vertical-align:top; }
td.k { color:#8b9199; }
.back { color:#2f6feb; text-decoration:none; font-size:13px; }
"""


def page(title, body):
    return ("<!doctype html><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>" + html.escape(title) + "</title><style>" + STYLE + "</style>" + body)


def index_html(items, sort_mode, only):
    def link(label, s, o):
        on = " on" if (sort_mode == s and only == o) else ""
        return ('<a class="' + on.strip() + '" href="/?sort=' + s + '&only=' + o + '">'
                + label + '</a>')

    pos = sum(1 for e in items if e["verdict"] is True)
    unk = sum(1 for e in items if e["unchecked"])
    head = ["<h1>seizureGuard olay arsivi</h1>",
            '<div class="sub">', str(len(items)), " olay &middot; ", str(pos),
            " pozitif &middot; ", str(unk), " kararsiz &middot; confidence = modelin kendi kararindan emin olma derecesi, ",
            "nobet olasiligi degil</div>",
            '<div class="bar">',
            link("Once pozitifler", "default", only),
            link("Tarihe gore", "time", only),
            link("Confidence'a gore", "conf", only),
            '<span style="width:14px"></span>',
            link("Hepsi", sort_mode, "all"),
            link("Sadece pozitif", sort_mode, "pos"),
            "</div>"]
    grid = ['<div class="grid">'] + [card_html(e) for e in items] + ["</div>"]
    if not items:
        grid = ['<div class="sub">Bu filtrede olay yok.</div>']
    return page("seizureGuard olaylari", "".join(head + grid))


def detail_html(e):
    esc = html.escape
    rows = [("Zaman", e["when"].replace("T", " ")),
            ("Kamera", e["camera"]),
            ("Karar", verdict_label(e)),
            ("Dogrulama", unchecked_text(e) or "-"),
            ("Confidence", "%.2f" % e["confidence"]),
            ("Gerekce", e["reason"]),
            ("Isaretler", ", ".join(e["signs"]) or "-"),
            ("Kare sayisi", str(e["frames"])),
            ("Video", ("%.1f MB" % e["size_mb"]) if e["has_video"] else "yok")]
    table = "".join('<tr><td class="k">' + esc(k) + "</td><td>" + esc(str(v)) + "</td></tr>"
                    for k, v in rows)
    vid = ('<video controls preload="metadata" poster="/thumb/' + esc(e["id"]) + '" '
           'src="/video/' + esc(e["id"]) + '"></video>') if e["has_video"] else \
          '<div class="sub">Bu olayda video yok.</div>'
    body = ['<a class="back" href="/">&larr; listeye don</a>',
            "<h1>" + badge(e) + " " + esc(e["id"]) + "</h1>",
            vid,
            "<table>" + table + "</table>",
            ('<p class="note">' + esc(e["note"]) + "</p>") if e["note"] else ""]
    return page(e["id"], "".join(body))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    root = None
    thumbs = None

    def _send(self, code, data, ctype, extra=None):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _send_file(self, path, ctype=None):
        """Range destegi sart: tarayicidaki MP4 ileri/geri sarma bunu ister."""
        ctype = ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if rng and rng.startswith("bytes="):
            spec = rng[6:].split(",")[0].strip()
            lo, _, hi = spec.partition("-")
            try:
                if lo:
                    start = int(lo)
                    end = int(hi) if hi else size - 1
                elif hi:
                    start = max(0, size - int(hi))
                status = 206
            except ValueError:
                status = 200
            start = max(0, min(start, size - 1))
            end = max(start, min(end, size - 1))

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(262144, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        path = unquote(u.path)
        q = parse_qs(u.query)
        items = get_index(self.root, self.thumbs)

        if path == "/":
            sort_mode = (q.get("sort") or ["default"])[0]
            only = (q.get("only") or ["all"])[0]
            shown = [e for e in items if e["verdict"] is True] if only == "pos" else items
            return self._send(200, index_html(sort_items(shown, sort_mode), sort_mode, only),
                              "text/html; charset=utf-8")

        if path == "/api/events":
            sort_mode = (q.get("sort") or ["default"])[0]
            return self._send(200, json.dumps(sort_items(items, sort_mode), ensure_ascii=False),
                              "application/json; charset=utf-8")

        if path.startswith("/event/"):
            eid = path[len("/event/"):]
            for e in items:
                if e["id"] == eid:
                    return self._send(200, detail_html(e), "text/html; charset=utf-8")
            return self._send(404, page("yok", "<h1>Olay bulunamadi</h1>"), "text/html; charset=utf-8")

        if path.startswith("/thumb/"):
            eid = path[len("/thumb/"):]
            if "/" in eid or ".." in eid:
                return self._send(400, b"bad id", "text/plain")
            t = thumb_path(self.root, eid, self.thumbs)
            if t is None:
                return self._send(404, b"thumb yok", "text/plain")
            return self._send_file(t, "image/jpeg")

        if path.startswith("/video/"):
            eid = path[len("/video/"):]
            if "/" in eid or ".." in eid:
                return self._send(400, b"bad id", "text/plain")
            v = video_path(self.root, eid)
            if v is None:
                return self._send(404, b"video yok", "text/plain")
            return self._send_file(v, "video/mp4")

        return self._send(404, b"yok", "text/plain")

    def log_message(self, fmt, *args):
        print("%s %s" % (self.address_string(), fmt % args), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=REPO / "data" / "events")
    ap.add_argument("--thumbs", type=Path, default=REPO / "data" / "thumbs")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    Handler.root = args.root.resolve()
    Handler.thumbs = args.thumbs.resolve()
    n = len(get_index(Handler.root, Handler.thumbs))
    print("olay arsivi: http://%s:%d/  (%d olay, kok=%s)"
          % (args.host, args.port, n, Handler.root), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
