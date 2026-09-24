import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import event_server


def _event(root, name, verdict, conf, video=b""):
    ev = root / name
    ev.mkdir(parents=True)
    (ev / "analysis.json").write_text(
        json.dumps({"final_abnormal_event": verdict, "final_confidence": conf}),
        encoding="utf-8")
    if video:
        (ev / "event.mp4").write_bytes(video)
    return ev


class TestNames:
    def test_regular_name(self):
        when, cam = event_server.parse_name("event_20260903_183211_c700-pi")
        assert (when.year, when.month, when.day, when.hour) == (2026, 9, 3, 18)
        assert cam == "c700-pi"

    def test_collision_suffix(self):
        when, cam = event_server.parse_name("event_20260903_183211_2_mi360-pi")
        assert when is not None and cam == "mi360-pi"

    def test_manual_name_has_no_timestamp(self):
        when, cam = event_server.parse_name("event_vid20260111_seizure_window_fable")
        assert when is None
        assert cam == "vid20260111_seizure_window_fable"


class TestIndex:
    def test_default_sort_puts_positives_before_newer_negatives(self, tmp_path):
        _event(tmp_path, "event_20260801_100000_a", True, 0.55)
        _event(tmp_path, "event_20260901_100000_a", False, 0.96)
        order = [e["id"] for e in event_server.sort_items(
            event_server.scan_events(tmp_path), "default")]
        assert order[0] == "event_20260801_100000_a"

    def test_confidence_sort_is_raw(self, tmp_path):
        _event(tmp_path, "event_20260801_100000_a", True, 0.55)
        _event(tmp_path, "event_20260901_100000_a", False, 0.96)
        top = event_server.sort_items(event_server.scan_events(tmp_path), "conf")[0]
        assert top["verdict"] is False and top["confidence"] == 0.96

    def test_orphan_thumbnails_are_dropped(self, tmp_path):
        events, thumbs = tmp_path / "events", tmp_path / "thumbs"
        _event(events, "event_20260901_100000_a", False, 0.9)
        thumbs.mkdir()
        (thumbs / "event_20260901_100000_a.jpg").write_bytes(b"x")
        (thumbs / "event_20260801_100000_pruned.jpg").write_bytes(b"x")
        items = event_server.scan_events(events)
        assert event_server.drop_orphan_thumbs(thumbs, {e["id"] for e in items}) == 1
        assert [p.name for p in thumbs.iterdir()] == ["event_20260901_100000_a.jpg"]


def test_video_range_request_returns_partial_content(tmp_path):
    payload = bytes(range(256)) * 16
    _event(tmp_path, "event_20260901_100000_a", True, 0.8, video=payload)
    event_server.Handler.root = tmp_path
    event_server.Handler.thumbs = tmp_path / "thumbs"
    event_server._index_cache.update(stamp=0.0, items=[])
    srv = ThreadingHTTPServer(("127.0.0.1", 0), event_server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = "http://127.0.0.1:%d/video/event_20260901_100000_a" % srv.server_address[1]
        req = urllib.request.Request(url, headers={"Range": "bytes=100-199"})
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.status == 206
            assert r.headers["Content-Range"] == "bytes 100-199/%d" % len(payload)
            assert r.read() == payload[100:200]
    finally:
        srv.shutdown()
        srv.server_close()
