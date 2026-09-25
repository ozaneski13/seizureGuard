import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import event_server


def _event(root, name, verdict, conf, video=b"", **extra):
    ev = root / name
    ev.mkdir(parents=True)
    (ev / "analysis.json").write_text(
        json.dumps({"final_abnormal_event": verdict, "final_confidence": conf, **extra}),
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


class TestUnchecked:
    """final_abnormal_event=False only means no ANALYZED batch was positive;
    failed batches were never looked at and must not read as negative."""

    def _partial(self, root, name="event_20260901_100000_a"):
        batches = [{"abnormal_event": False, "confidence": 0.9},
                   {"abnormal_event": None, "confidence": 0.0}]
        _event(root, name, False, 0.9, failed_batches=1, batches=batches,
               final_reason="... treat this result as incomplete.")
        return {e["id"]: e for e in event_server.scan_events(root)}[name]

    def test_partial_failure_is_not_negative(self, tmp_path):
        e = self._partial(tmp_path)
        assert e["unchecked"] is True
        assert (e["failed_batches"], e["total_batches"]) == (1, 2)
        assert "KARARSIZ" in event_server.badge(e)
        assert "1/2" in event_server.card_html(e)
        detail = event_server.detail_html(e)
        assert "negatif" not in detail and "1/2" in detail

    def test_outage_is_not_negative(self, tmp_path):
        _event(tmp_path, "event_20260901_100000_a", False, 0.0, failed_batches=2,
               backend_outage="401 OAuth access token has expired",
               batches=[{"abnormal_event": None}, {"abnormal_event": None}])
        e = event_server.scan_events(tmp_path)[0]
        assert e["unchecked"] is True
        assert "KARARSIZ" in event_server.badge(e)

    def test_clean_negative_and_positive_are_checked(self, tmp_path):
        _event(tmp_path, "event_20260901_100000_a", False, 0.9, failed_batches=0)
        _event(tmp_path, "event_20260901_110000_a", True, 0.7, failed_batches=1)
        neg, pos = event_server.scan_events(tmp_path)
        assert neg["unchecked"] is False and "negatif" in event_server.badge(neg)
        # a positive verdict stands even when another batch failed
        assert pos["unchecked"] is False and "POZITIF" in event_server.badge(pos)

    def test_default_sort_is_positive_then_unchecked_then_negative(self, tmp_path):
        _event(tmp_path, "event_20260903_100000_a", False, 0.96)
        self._partial(tmp_path, "event_20260802_100000_a")
        _event(tmp_path, "event_20260801_100000_a", True, 0.55)
        order = [e["id"] for e in event_server.sort_items(
            event_server.scan_events(tmp_path), "default")]
        assert order == ["event_20260801_100000_a", "event_20260802_100000_a",
                         "event_20260903_100000_a"]

    def test_index_header_counts_unchecked(self, tmp_path):
        self._partial(tmp_path)
        page = event_server.index_html(event_server.scan_events(tmp_path), "default", "all")
        assert "1 kararsiz" in page


class TestPeakNote:
    def test_top_level_note_wins(self):
        assert event_server.peak_note({"batches": [
            {"abnormal_event": False, "confidence": 0.2, "note": "low"},
            {"abnormal_event": False, "confidence": 0.9, "screen_verdict": None,
             "note": "paddling on side"}]}) == "paddling on side"

    def test_legacy_screen_verdict_note(self):
        assert event_server.peak_note({"batches": [
            {"abnormal_event": True, "confidence": 0.9,
             "screen_verdict": {"note": "old"}}]}) == "old"

    def test_positive_event_is_explained_by_a_positive_batch(self):
        assert event_server.peak_note({"batches": [
            {"abnormal_event": False, "confidence": 0.95, "note": "normal purposeful walking"},
            {"abnormal_event": True, "confidence": 0.7, "note": "paddling on its side"},
            {"abnormal_event": True, "confidence": 0.6, "note": "rigid limbs"}]}) \
            == "paddling on its side"

    def test_failed_batch_note_is_skipped(self):
        assert event_server.peak_note({"batches": [
            {"abnormal_event": None, "confidence": 0.99, "note": "parse error"},
            {"abnormal_event": False, "confidence": 0.4, "note": "sleeping"}]}) == "sleeping"

    def test_note_is_truncated(self):
        assert len(event_server.peak_note({"batches": [
            {"abnormal_event": True, "confidence": 0.5, "note": "x" * 400}]})) == 300

    def test_non_string_note_does_not_break_the_index(self, tmp_path):
        _event(tmp_path, "event_20260901_100000_a", True, 0.7, batches=[
            {"abnormal_event": True, "confidence": 0.7, "note": {"odd": 1}},
            {"abnormal_event": True, "confidence": 0.5, "note": 42}])
        assert event_server.scan_events(tmp_path)[0]["note"] == ""


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
