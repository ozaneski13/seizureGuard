import contextlib
import types

import pytest

import alerts


class _FakeResponse:
    status = 200

    def read(self):
        return b'{"ok": true}'


@pytest.fixture
def tg_env(monkeypatch):
    monkeypatch.setenv("SEIZUREGUARD_TG_TOKEN", "123:abc")
    monkeypatch.setenv("SEIZUREGUARD_TG_CHAT", "42")
    monkeypatch.setattr(alerts.time, "sleep", lambda s: None)


@pytest.fixture
def no_tg_env(monkeypatch):
    monkeypatch.delenv("SEIZUREGUARD_TG_TOKEN", raising=False)
    monkeypatch.delenv("SEIZUREGUARD_TG_CHAT", raising=False)


def _capture_urlopen(monkeypatch, requests):
    @contextlib.contextmanager
    def fake(req, timeout=None):
        requests.append(req)
        yield _FakeResponse()

    monkeypatch.setattr(alerts.urllib.request, "urlopen", fake)


class TestMultipart:
    def test_body_structure(self):
        content_type, body = alerts._multipart(
            {"chat_id": "42", "caption": "hi"}, "photo", "f.jpg", b"\xff\xd8jpegbytes")
        assert alerts._BOUNDARY in content_type
        assert b'name="chat_id"\r\n\r\n42' in body
        assert b'name="caption"\r\n\r\nhi' in body
        assert b'filename="f.jpg"' in body
        assert b"\xff\xd8jpegbytes" in body
        assert body.endswith(f"--{alerts._BOUNDARY}--\r\n".encode())


class TestConsoleFallback:
    def test_no_env_prints_and_returns_false(self, no_tg_env, monkeypatch, capsys):
        def explode(*a, **k):
            raise AssertionError("network must not be touched")

        monkeypatch.setattr(alerts.urllib.request, "urlopen", explode)
        assert alerts.send_alert("dog event", photo_path="x.jpg") is False
        out = capsys.readouterr().out
        assert "ALERT" in out and "dog event" in out


class TestTelegram:
    def test_text_message_goes_to_sendmessage(self, tg_env, monkeypatch):
        requests = []
        _capture_urlopen(monkeypatch, requests)
        assert alerts.send_alert("hello") is True
        assert len(requests) == 1
        req = requests[0]
        assert "bot123:abc/sendMessage" in req.full_url
        assert b"chat_id=42" in req.data and b"text=hello" in req.data

    def test_photo_goes_to_sendphoto_multipart(self, tg_env, monkeypatch, tmp_path):
        photo = tmp_path / "peak.jpg"
        photo.write_bytes(b"\xff\xd8fakejpg")
        requests = []
        _capture_urlopen(monkeypatch, requests)
        assert alerts.send_alert("seizure?", photo_path=photo) is True
        req = requests[0]
        assert "sendPhoto" in req.full_url
        assert req.headers["Content-type"].startswith("multipart/form-data")
        assert b"\xff\xd8fakejpg" in req.data
        assert b'name="caption"\r\n\r\nseizure?' in req.data

    def test_missing_photo_falls_back_to_text(self, tg_env, monkeypatch, tmp_path):
        requests = []
        _capture_urlopen(monkeypatch, requests)
        assert alerts.send_alert("msg", photo_path=tmp_path / "gone.jpg") is True
        assert "sendMessage" in requests[0].full_url

    def test_video_goes_to_sendvideo_multipart(self, tg_env, monkeypatch, tmp_path):
        clip = tmp_path / "alert_clip.mp4"
        clip.write_bytes(b"\x00\x00\x00 ftypmp42fakevideo")
        photo = tmp_path / "peak.jpg"
        photo.write_bytes(b"\xff\xd8fakejpg")
        requests = []
        _capture_urlopen(monkeypatch, requests)
        assert alerts.send_alert("seizure?", photo_path=photo, video_path=clip) is True
        req = requests[0]
        assert "sendVideo" in req.full_url
        assert b'name="video"' in req.data
        assert b"fakevideo" in req.data

    def test_missing_video_falls_back_to_photo(self, tg_env, monkeypatch, tmp_path):
        photo = tmp_path / "peak.jpg"
        photo.write_bytes(b"\xff\xd8fakejpg")
        requests = []
        _capture_urlopen(monkeypatch, requests)
        assert alerts.send_alert("msg", photo_path=photo,
                                 video_path=tmp_path / "gone.mp4") is True
        assert "sendPhoto" in requests[0].full_url

    def test_failing_video_upload_retries_without_clip(self, tg_env, monkeypatch, tmp_path):
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"vid")
        photo = tmp_path / "p.jpg"
        photo.write_bytes(b"\xff\xd8jpg")
        urls = []

        @__import__("contextlib").contextmanager
        def fake(req, timeout=None):
            urls.append(req.full_url)
            if "sendVideo" in req.full_url:
                raise OSError("upload too slow")
            yield _FakeResponse()

        monkeypatch.setattr(alerts.urllib.request, "urlopen", fake)
        assert alerts.send_alert("evt", photo_path=photo, video_path=clip) is True
        assert any("sendVideo" in u for u in urls)
        assert "sendPhoto" in urls[-1]

    def test_network_failure_never_raises(self, tg_env, monkeypatch, capsys):
        calls = []

        def explode(req, timeout=None):
            calls.append(req)
            raise OSError("network down")

        monkeypatch.setattr(alerts.urllib.request, "urlopen", explode)
        assert alerts.send_alert("evt") is False
        assert len(calls) == alerts.MAX_ATTEMPTS
        assert "delivery failed" in capsys.readouterr().out
