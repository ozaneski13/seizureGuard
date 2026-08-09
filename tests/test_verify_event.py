import json
import sys
import types

import pytest

import verify_event as ve


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(ve.time, "sleep", lambda s: None)
    monkeypatch.delenv("SEIZUREGUARD_BACKEND", raising=False)


def _is_screen(prompt):
    return '"seen"' in prompt


def _fake_run_claude(screen_fn, confirm_fn):
    def fake(prompt, model, images, timeout=600):
        if _is_screen(prompt):
            return screen_fn(prompt)
        return confirm_fn(prompt)
    return fake


def _run_main(monkeypatch, event_dir):
    monkeypatch.setattr(sys, "argv", ["verify_event.py", str(event_dir)])
    ve.main()
    return json.loads((event_dir / "analysis.json").read_text())


def _prompt_hits_window(prompt, lo, hi):
    import re
    m = re.search(r"attached images\): ([0-9., ]+)", prompt)
    times = [float(x) for x in m.group(1).split(",") if x.strip()] if m else []
    return any(lo <= t <= hi for t in times)


class TestPureCore:
    def test_frame_time(self, tmp_path):
        assert ve.frame_time(tmp_path / "frame_003_t_12.345s.jpg") == 12.345
        assert ve.frame_time(tmp_path / "whatever.jpg") == 0.0

    def test_iter_batches(self):
        items = list(range(70))
        batches = list(ve.iter_batches(items, 30))
        assert [len(b) for b in batches] == [30, 30, 10]
        assert [x for b in batches for x in b] == items

    def test_parse_json_verdict_plain(self):
        assert ve.parse_json_verdict('{"seen": "no", "confidence": 0.1}') == {
            "seen": "no", "confidence": 0.1}

    def test_parse_json_verdict_fenced(self):
        text = 'Here you go:\n```json\n{"seen": "yes", "confidence": 0.8}\n```'
        assert ve.parse_json_verdict(text)["seen"] == "yes"

    def test_parse_json_verdict_prose_wrapped(self):
        text = 'The answer is {"abnormal_event": false, "confidence": 0.2} as requested.'
        assert ve.parse_json_verdict(text)["abnormal_event"] is False

    def test_parse_json_verdict_garbage_raises(self):
        with pytest.raises(ValueError):
            ve.parse_json_verdict("I cannot read these images")


class TestSignRules:
    def _verdict(self, present_signs, abnormal=False):
        return {
            "abnormal_event": abnormal,
            "confidence": 0.5,
            "observed_signs": [
                {"sign": s, "present": True, "body_region": "x", "sustained": True}
                for s in present_signs
            ],
        }

    def test_one_hard_sign_is_positive(self):
        assert ve.decide_signs(self._verdict(["paddling"])) is True

    def test_two_soft_signs_are_positive(self):
        assert ve.decide_signs(self._verdict(["drooling", "head_tremor"])) is True

    def test_one_soft_sign_is_negative(self):
        assert ve.decide_signs(self._verdict(["drooling"])) is False

    def test_model_abnormal_flag_alone_is_positive(self):
        assert ve.decide_signs(self._verdict([], abnormal=True)) is True

    def test_absent_signs_do_not_count(self):
        verdict = {
            "abnormal_event": False,
            "observed_signs": [
                {"sign": "paddling", "present": False},
                {"sign": "drooling", "present": False},
            ],
        }
        assert ve.decide_signs(verdict) is False

    def test_empty_verdict_is_negative(self):
        assert ve.decide_signs({}) is False


class TestScreenGate:
    def test_confident_no_skips(self):
        assert ve.should_escalate({"seen": "no", "confidence": 0.05}) is False

    def test_yes_escalates(self):
        assert ve.should_escalate({"seen": "yes", "confidence": 0.05}) is True

    def test_uncertain_no_escalates(self):
        assert ve.should_escalate({"seen": "no", "confidence": 0.4}) is True

    def test_failed_screen_escalates(self):
        assert ve.should_escalate(None) is True

    def test_missing_confidence_fails_open(self):
        assert ve.should_escalate({"seen": "no"}) is True

    def test_lateral_recumbency_escalates_despite_confident_no(self):
        assert ve.should_escalate(
            {"seen": "no", "confidence": 0.05, "posture": "lying_lateral"}) is True

    def test_other_postures_do_not_force_escalation(self):
        assert ve.should_escalate(
            {"seen": "no", "confidence": 0.05, "posture": "standing"}) is False
        assert ve.should_escalate(
            {"seen": "no", "confidence": 0.05, "posture": "lying_sternal"}) is False


class TestRunClaude:
    def _proc(self, stdout, returncode=0):
        return types.SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)

    def _stream(self, result_text, is_error=False):
        lines = [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps({"type": "assistant", "message": {"content": []}}),
            json.dumps({"type": "result", "is_error": is_error, "result": result_text}),
        ]
        return "\n".join(lines)

    def test_parses_result_event(self, monkeypatch):
        monkeypatch.setattr(ve.subprocess, "run", lambda *a, **k: self._proc(
            self._stream('{"seen": "no", "confidence": 0.1}')))
        out = ve.run_claude("prompt", "haiku", images=[])
        assert out == {"seen": "no", "confidence": 0.1}

    def test_fenced_result_is_cleaned(self, monkeypatch):
        monkeypatch.setattr(ve.subprocess, "run", lambda *a, **k: self._proc(
            self._stream('```json\n{"seen": "yes", "confidence": 0.9}\n```')))
        assert ve.run_claude("p", "m", images=[])["seen"] == "yes"

    def test_images_are_embedded_as_base64(self, monkeypatch, tmp_path):
        frame = tmp_path / "frame_000_t_1.000s.jpg"
        frame.write_bytes(b"\xff\xd8fakejpg")
        captured = {}

        def fake(cmd, **kwargs):
            captured["stdin"] = kwargs.get("input")
            return self._proc(self._stream('{"seen": "no", "confidence": 0.0}'))

        monkeypatch.setattr(ve.subprocess, "run", fake)
        ve.run_claude("p", "m", images=[frame])
        msg = json.loads(captured["stdin"])
        blocks = msg["message"]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image"
        assert blocks[1]["source"]["media_type"] == "image/jpeg"
        import base64 as b64mod
        assert b64mod.b64decode(blocks[1]["source"]["data"]) == b"\xff\xd8fakejpg"

    def test_login_error_raises_dedicated_error(self, monkeypatch):
        monkeypatch.setattr(ve.subprocess, "run", lambda *a, **k: self._proc(
            self._stream("Not logged in · Please run /login", is_error=True), 1))
        with pytest.raises(ve.ClaudeLoginError):
            ve.run_claude("p", "m", images=[])

    def test_no_result_event_raises(self, monkeypatch):
        monkeypatch.setattr(ve.subprocess, "run",
                            lambda *a, **k: self._proc("segfault or whatever", 1))
        with pytest.raises(RuntimeError):
            ve.run_claude("p", "m", images=[])


class TestClaudeBackendFlow:
    def test_confident_negative_screen_skips_confirm(self, monkeypatch, tmp_path):
        calls = []

        def fake(prompt, model, images, timeout=600):
            calls.append(model)
            assert _is_screen(prompt)
            return {"seen": "no", "confidence": 0.05}

        monkeypatch.setattr(ve, "run_claude", fake)
        r = ve.assess_batch_claude(tmp_path, [], ve.get_config(), 1)
        assert r["abnormal_event"] is False
        assert r["escalated"] is False
        assert calls == ["claude-haiku-4-5"]

    def test_positive_screen_escalates_to_confirm(self, monkeypatch, tmp_path):
        models = []

        def fake(prompt, model, images, timeout=600):
            models.append(model)
            if _is_screen(prompt):
                return {"seen": "yes", "confidence": 0.8}
            return {"abnormal_event": True, "confidence": 0.9,
                    "observed_signs": [{"sign": "paddling", "present": True,
                                        "body_region": "legs", "sustained": True}]}

        monkeypatch.setattr(ve, "run_claude", fake)
        r = ve.assess_batch_claude(tmp_path, [], ve.get_config(), 1)
        assert r["abnormal_event"] is True
        assert r["escalated"] is True
        assert models == ["claude-haiku-4-5", "claude-fable-5"]

    def test_lateral_posture_reaches_confirm_and_is_recorded(self, monkeypatch, tmp_path):
        def fake(prompt, model, images, timeout=600):
            if _is_screen(prompt):
                return {"seen": "no", "confidence": 0.05, "posture": "lying_lateral"}
            return {"abnormal_event": False, "confidence": 0.2,
                    "posture": "lying_lateral", "partially_visible": True,
                    "observed_signs": []}

        monkeypatch.setattr(ve, "run_claude", fake)
        r = ve.assess_batch_claude(tmp_path, [], ve.get_config(), 1)
        assert r["escalated"] is True
        assert r["posture"] == "lying_lateral"
        assert r["partially_visible"] is True

    def test_screen_failure_fails_open_to_confirm(self, monkeypatch, tmp_path):
        def fake(prompt, model, images, timeout=600):
            if _is_screen(prompt):
                raise RuntimeError("boom")
            return {"abnormal_event": False, "confidence": 0.1, "observed_signs": []}

        monkeypatch.setattr(ve, "run_claude", fake)
        r = ve.assess_batch_claude(tmp_path, [], ve.get_config(), 1)
        assert r["escalated"] is True
        assert r["abnormal_event"] is False

    def test_confirm_failure_marks_batch_failed(self, monkeypatch, tmp_path):
        def fake(prompt, model, images, timeout=600):
            if _is_screen(prompt):
                return {"seen": "yes", "confidence": 0.9}
            raise RuntimeError("confirm down")

        monkeypatch.setattr(ve, "run_claude", fake)
        r = ve.assess_batch_claude(tmp_path, [], ve.get_config(), 1)
        assert r["abnormal_event"] is None
        assert "error" in r

    def test_login_error_aborts_immediately(self, monkeypatch, tmp_path):
        def fake(prompt, model, images, timeout=600):
            raise ve.ClaudeLoginError("not logged in")

        monkeypatch.setattr(ve, "run_claude", fake)
        with pytest.raises(ve.ClaudeLoginError):
            ve.assess_batch_claude(tmp_path, [], ve.get_config(), 1)


class TestEndToEnd:
    def test_early_true_survives_later_quiet_batches(self, monkeypatch, synthetic_event_dir):
        def screen(prompt):
            hit = _prompt_hits_window(prompt, 20.0, 23.0)
            return {"seen": "yes" if hit else "no", "confidence": 0.9 if hit else 0.05}

        def confirm(prompt):
            return {"abnormal_event": True, "confidence": 0.9,
                    "observed_signs": [{"sign": "rhythmic_jerking", "present": True,
                                        "body_region": "whole body", "sustained": True}]}

        monkeypatch.setattr(ve, "run_claude", _fake_run_claude(screen, confirm))
        out = _run_main(monkeypatch, synthetic_event_dir)
        assert out["final_abnormal_event"] is True
        assert out["failed_batches"] == 0
        assert any(b["escalated"] for b in out["batches"])
        assert not all(b["escalated"] for b in out["batches"])

    def test_all_negative_stays_negative(self, monkeypatch, synthetic_event_dir):
        monkeypatch.setattr(ve, "run_claude", _fake_run_claude(
            lambda p: {"seen": "no", "confidence": 0.05},
            lambda p: pytest.fail("confirm must not be called"),
        ))
        out = _run_main(monkeypatch, synthetic_event_dir)
        assert out["final_abnormal_event"] is False
        assert out["failed_batches"] == 0

    def test_failed_confirm_reports_incomplete(self, monkeypatch, synthetic_event_dir):
        def confirm(prompt):
            if _prompt_hits_window(prompt, 33.0, 37.0):
                raise RuntimeError("simulated failure")
            return {"abnormal_event": False, "confidence": 0.1, "observed_signs": []}

        monkeypatch.setattr(ve, "run_claude", _fake_run_claude(
            lambda p: {"seen": "no", "confidence": 0.5}, confirm))
        out = _run_main(monkeypatch, synthetic_event_dir)
        assert out["failed_batches"] >= 1
        assert out["final_abnormal_event"] is False
        assert "incomplete" in out["final_reason"]

    def test_analysis_json_contract(self, monkeypatch, synthetic_event_dir):
        monkeypatch.setattr(ve, "run_claude", _fake_run_claude(
            lambda p: {"seen": "no", "confidence": 0.05}, lambda p: None))
        out = _run_main(monkeypatch, synthetic_event_dir)
        for key in ("final_abnormal_event", "final_confidence", "backend",
                    "screen_model", "confirm_model", "batch_size", "num_frames",
                    "base_frames", "burst_frames", "failed_batches",
                    "final_reason", "batches"):
            assert key in out
        assert out["backend"] == "claude-cli"
        for b in out["batches"]:
            for key in ("abnormal_event", "confidence", "screen_verdict",
                        "escalated", "observed_signs"):
                assert key in b

    def test_openai_backend_requires_key(self, monkeypatch, synthetic_event_dir):
        monkeypatch.setenv("SEIZUREGUARD_BACKEND", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(sys, "argv", ["verify_event.py", str(synthetic_event_dir)])
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            ve.main()
