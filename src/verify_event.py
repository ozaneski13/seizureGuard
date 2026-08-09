"""Two-tier VLM verification of a captured event directory: a cheap screen
model triages each 30-frame batch, positives escalate to a strong confirm
model that assesses specific canine seizure signs; a deterministic rule
layer decides, recall-first. Writes analysis.json into the event dir.

Usage: python src/verify_event.py <event_dir>
"""
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BATCH_SIZE = 30
SCREEN_ATTEMPTS = 1     # screen failure escalates anyway (fail-open)
CONFIRM_ATTEMPTS = 2
MAX_ATTEMPTS = CONFIRM_ATTEMPTS  # openai backend

# Recall-first: screen escalates unless it is confidently negative
SCREEN_ESCALATE_CONF = 0.15

# Canine seizure semiology vocabulary (IVETF/JVIM-informed)
HARD_SIGNS = {
    "paddling", "tonic_stiffening", "rhythmic_jerking",
    "jaw_clonus", "loss_of_posture", "fencing_posture",
}
SOFT_SIGNS = {"drooling", "head_tremor", "muscle_twitching", "disorientation"}
ALL_SIGNS = sorted(HARD_SIGNS | SOFT_SIGNS)

POSTURES = ("standing", "sitting", "lying_sternal", "lying_lateral", "unclear")

# Learned from the first owner recording: the convulsing dog sat at the frame
# edge while its mirror reflection read as a healthy standing dog.
VISION_CAUTIONS = (
    "Judge only the physical dog: rooms may contain mirrors or glass whose "
    "reflections look like a second dog — never base your verdict on a "
    "reflection. If the dog is partially out of frame, evaluate what is "
    "visible instead of assuming the hidden part moves normally.\n"
)


def get_config():
    return {
        "backend": os.environ.get("SEIZUREGUARD_BACKEND", "claude-cli"),
        "screen_model": os.environ.get("SEIZUREGUARD_SCREEN_MODEL", "claude-haiku-4-5"),
        # A/B on the owner's real seizure recording (2026-08-09): sonnet-5
        # confirm called all 6 seizure batches normal; fable-5 flagged all 6
        # with hard signs. Confirm quality is what the alert rides on.
        "confirm_model": os.environ.get("SEIZUREGUARD_CONFIRM_MODEL", "claude-fable-5"),
        "openai_model": os.environ.get("SEIZUREGUARD_MODEL", "gpt-4.1-mini"),
    }


# ---------------------------------------------------------------- pure core

def frame_time(path):
    m = re.search(r"_t_([0-9.]+)s", path.name)
    return float(m.group(1)) if m else 0.0


def iter_batches(paths, batch_size):
    for i in range(0, len(paths), batch_size):
        yield paths[i:i + batch_size]


def strip_fences(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def parse_json_verdict(text):
    cleaned = strip_fences(text)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in model output: {text[:200]!r}")
    return json.loads(m.group(0))


def decide_signs(verdict):
    """Batch-level decision from a confirm verdict (recall-first rule layer)."""
    signs = verdict.get("observed_signs") or []
    present = {s.get("sign") for s in signs if isinstance(s, dict) and s.get("present")}
    if bool(verdict.get("abnormal_event")):
        return True
    if present & HARD_SIGNS:
        return True
    return len(present) >= 2


def should_escalate(screen_verdict):
    """Screen tier gate: escalate unless confidently negative (fail-open).

    Lateral recumbency always escalates regardless of confidence: a dog flat
    on its side with any motion is the classic ictal presentation, and the
    screen tier has misread it as normal on real footage."""
    if screen_verdict is None:
        return True
    seen = str(screen_verdict.get("seen", "")).strip().lower()
    conf = float(screen_verdict.get("confidence", 1.0))
    posture = str(screen_verdict.get("posture", "")).strip().lower()
    return seen == "yes" or conf >= SCREEN_ESCALATE_CONF or posture == "lying_lateral"


# ------------------------------------------------------- claude-cli backend

class ClaudeLoginError(RuntimeError):
    pass


SYSTEM_CONTRACT = (
    "You are a JSON-only vision component inside an automated pipeline; no "
    "human reads your reply. The user message contains camera frames as "
    "attached images. Your entire reply must be exactly one JSON object "
    "matching the requested schema — no prose, no markdown, no headings. "
    "Findings that contradict the task's premise (no dog, wrong footage, "
    "synthetic imagery) are expressed INSIDE the JSON via its fields, never "
    "by abandoning the format."
)


def _timestamps(paths):
    return ", ".join(f"{frame_time(p):.1f}" for p in paths)


def screen_prompt(paths):
    return (
        "You are screening frames from a dog monitoring camera for abnormal motor "
        "events (possible epileptic seizure). This is NOT medical diagnosis.\n"
        f"Frame timestamps in seconds (same order as the attached images): {_timestamps(paths)}\n"
        "Frames are sampled at 2 fps normally and 10 fps during motion bursts, so gaps vary.\n"
        + VISION_CAUTIONS +
        "We strongly prefer false positives over misses. If unsure, lean towards yes.\n\n"
        "Reply with exactly this JSON object and nothing else:\n"
        '{"seen": "yes" or "no", "confidence": <number 0..1, likelihood an abnormal '
        'motor event is present in these frames>, '
        f'"posture": "<dominant posture, one of: {", ".join(POSTURES)}>", '
        '"note": "<optional short remark>"}\n'
        "If no dog is visible — wrong scene, empty room, synthetic/test imagery — that IS "
        'a valid result: {"seen": "no", "confidence": 0.0, "note": "no dog visible"}.'
    )


def confirm_prompt(paths):
    return (
        "You are analyzing frames from a dog monitoring camera for signs of an "
        "abnormal motor event (possible epileptic seizure). This is NOT medical diagnosis.\n"
        f"Frame timestamps in seconds (same order as the attached images): {_timestamps(paths)}\n"
        "Frames are sampled at 2 fps normally and 10 fps during motion bursts, so gaps vary.\n"
        + VISION_CAUTIONS +
        "Assess each of these specific canine seizure signs across the sequence: "
        f"{', '.join(ALL_SIGNS)}.\n"
        "We prefer false positives over misses.\n\n"
        "Reply with exactly this JSON object and nothing else:\n"
        "{\n"
        '  "abnormal_event": true or false,\n'
        '  "confidence": <number 0..1>,\n'
        f'  "posture": "<dominant posture, one of: {", ".join(POSTURES)}>",\n'
        '  "partially_visible": true or false,\n'
        '  "observed_signs": [\n'
        f'    {{"sign": "<one of: {", ".join(ALL_SIGNS)}>", "present": true or false, '
        '"body_region": "<short text>", "sustained": true or false}\n'
        "  ],\n"
        '  "note": "<optional short remark>"\n'
        "}\n"
        "Include an observed_signs entry for every sign you could evaluate.\n"
        "If no dog is visible — wrong scene, empty room, synthetic/test imagery — that IS "
        'a valid result: {"abnormal_event": false, "confidence": 0.0, "observed_signs": [], '
        '"note": "no dog visible"}.'
    )


def run_claude(prompt, model, images, timeout=300):
    """Single-turn vision call: frames go in as base64 content blocks over
    stream-json stdin — the model has no tools and no filesystem access, so
    it can only ever see this batch's frames."""
    exe = shutil.which("claude")
    if exe is None:
        raise RuntimeError("claude CLI not found on PATH")

    content = [{"type": "text", "text": prompt}]
    for p in images:
        b64 = base64.b64encode(Path(p).read_bytes()).decode("ascii")
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg",
                                   "data": b64}})
    stdin = json.dumps({"type": "user",
                        "message": {"role": "user", "content": content}}) + "\n"

    proc = subprocess.run(
        [exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
         "--verbose", "--model", model, "--max-turns", "1",
         "--system-prompt", SYSTEM_CONTRACT],
        input=stdin, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout,
    )

    result = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "result":
            result = obj
    if result is None:
        raise RuntimeError(
            f"claude CLI produced no result event (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '')[:300]}"
        )
    result_text = result.get("result") or ""
    if result.get("is_error"):
        if "not logged in" in result_text.lower():
            raise ClaudeLoginError(
                "claude CLI is not logged in — run `claude` in a terminal and use /login"
            )
        raise RuntimeError(f"claude CLI error: {result_text[:300]}")
    return parse_json_verdict(result_text)


def _with_retry(fn, batch_index, tier, attempts):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(), None
        except ClaudeLoginError:
            raise
        except Exception as e:
            last_error = str(e)
            print(f"[WARN] Batch {batch_index} {tier} attempt {attempt}/{attempts} failed: {e}",
                  flush=True)
            if attempt < attempts:
                time.sleep(2 ** attempt)
    return None, last_error


def assess_batch_claude(event_dir, paths, config, batch_index):
    """Screen with the cheap model; confirm positives with the strong model."""
    screen, screen_error = _with_retry(
        lambda: run_claude(screen_prompt(paths), config["screen_model"],
                           images=paths, timeout=300),
        batch_index, "screen", SCREEN_ATTEMPTS,
    )
    escalated = should_escalate(screen)

    if not escalated:
        return {
            "abnormal_event": False,
            "confidence": float(screen.get("confidence", 0.0)),
            "screen_verdict": screen,
            "escalated": False,
            "observed_signs": [],
            "posture": screen.get("posture"),
        }

    confirm, confirm_error = _with_retry(
        lambda: run_claude(confirm_prompt(paths), config["confirm_model"],
                           images=paths, timeout=600),
        batch_index, "confirm", CONFIRM_ATTEMPTS,
    )
    if confirm is None:
        return {
            "abnormal_event": None,
            "confidence": 0.0,
            "screen_verdict": screen if screen is not None else {"error": screen_error},
            "escalated": True,
            "observed_signs": [],
            "error": confirm_error,
        }
    return {
        "abnormal_event": decide_signs(confirm),
        "confidence": float(confirm.get("confidence", 0.0)),
        "screen_verdict": screen if screen is not None else {"error": screen_error},
        "escalated": True,
        "observed_signs": confirm.get("observed_signs") or [],
        "posture": confirm.get("posture"),
        "partially_visible": confirm.get("partially_visible"),
    }


# ---------------------------------------------------------- openai backend

def encode_images(paths):
    payload = []
    for p in paths:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})
    return payload


def ask_openai(client, event_dir, paths, model):
    times = ", ".join(f"{frame_time(p):.1f}" for p in paths)
    prompt = (
        "You are analyzing chronological frames from a dog monitoring camera for an "
        "abnormal motor event (possible seizure). This is NOT medical diagnosis.\n"
        f"Frame timestamps in seconds (same order as the images): {times}\n"
        "Frames are sampled at 2 fps normally and 10 fps during motion bursts, so gaps vary.\n"
        f"Assess these canine seizure signs: {', '.join(ALL_SIGNS)}.\n"
        "We prefer false positives over misses.\n"
        "Return JSON with fields: abnormal_event (boolean), confidence (0..1), "
        "observed_signs (array of {sign, present, body_region, sustained})."
    )
    schema = {
        "type": "object",
        "properties": {
            "abnormal_event": {"type": "boolean"},
            "confidence": {"type": "number"},
            "observed_signs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sign": {"type": "string", "enum": ALL_SIGNS},
                        "present": {"type": "boolean"},
                        "body_region": {"type": "string"},
                        "sustained": {"type": "boolean"},
                    },
                    "required": ["sign", "present", "body_region", "sustained"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["abnormal_event", "confidence", "observed_signs"],
        "additionalProperties": False,
    }
    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}, *encode_images(paths)]
        }],
        text={"format": {"type": "json_schema", "name": "event_verdict",
                         "schema": schema, "strict": True}},
        max_output_tokens=600,
    )
    return json.loads(resp.output_text.strip())


def assess_batch_openai(client, event_dir, paths, config, batch_index):
    verdict, error = _with_retry(
        lambda: ask_openai(client, event_dir, paths, config["openai_model"]),
        batch_index, "confirm", MAX_ATTEMPTS,
    )
    if verdict is None:
        return {"abnormal_event": None, "confidence": 0.0, "screen_verdict": None,
                "escalated": True, "observed_signs": [], "error": error}
    return {
        "abnormal_event": decide_signs(verdict),
        "confidence": float(verdict.get("confidence", 0.0)),
        "screen_verdict": None,
        "escalated": True,
        "observed_signs": verdict.get("observed_signs") or [],
    }


def make_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (required for SEIZUREGUARD_BACKEND=openai)")
    from openai import OpenAI
    return OpenAI(api_key=api_key)


# ------------------------------------------------------------------- main

def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if len(sys.argv) != 2:
        print("Usage: python src/verify_event.py <event_dir>")
        raise SystemExit(1)

    config = get_config()

    event_dir = Path(sys.argv[1]).resolve()
    if not event_dir.exists():
        raise RuntimeError(f"Event dir not found: {event_dir}")

    base_dir = event_dir / "base"
    burst_dir = event_dir / "burst"
    if not base_dir.exists():
        raise RuntimeError(f"Missing base folder: {base_dir}")
    if not burst_dir.exists():
        raise RuntimeError(f"Missing burst folder: {burst_dir}")

    frames = sorted(
        list(base_dir.glob("frame_*.jpg")) + list(burst_dir.glob("frame_*.jpg")),
        key=frame_time,
    )
    if not frames:
        raise RuntimeError("No frames found")

    if config["backend"] == "openai":
        client = make_openai_client()
        assess = lambda paths, bi: assess_batch_openai(client, event_dir, paths, config, bi)
    else:
        assess = lambda paths, bi: assess_batch_claude(event_dir, paths, config, bi)

    batch_results = []
    for bi, batch in enumerate(iter_batches(frames, BATCH_SIZE), start=1):
        r = assess(batch, bi)
        batch_results.append(r)
        print(f"[INFO] Batch {bi}: abnormal={r['abnormal_event']} "
              f"conf={r['confidence']:.2f} escalated={r['escalated']}", flush=True)

    # Event-level decision (recall-focused): pure OR over analyzed batches.
    # A seizure that ended is still a seizure that happened — nothing may veto a true.
    any_true = any(r.get("abnormal_event") is True for r in batch_results)
    failed_batches = sum(1 for r in batch_results if r.get("abnormal_event") is None)
    max_conf = max(
        (float(r.get("confidence", 0.0)) for r in batch_results
         if r.get("abnormal_event") is not None),
        default=0.0,
    )

    final = any_true
    if any_true:
        final_reason = "Any batch true (recall-focused OR rule)."
    elif failed_batches:
        final_reason = (
            f"No batch marked abnormal_event=true, but {failed_batches} batch(es) failed "
            "and were never analyzed — treat this result as incomplete."
        )
    else:
        final_reason = "No batch marked abnormal_event=true."

    out = {
        "final_abnormal_event": final,
        "final_confidence": max_conf,
        "backend": config["backend"],
        "screen_model": config["screen_model"] if config["backend"] == "claude-cli" else None,
        "confirm_model": (config["confirm_model"] if config["backend"] == "claude-cli"
                          else config["openai_model"]),
        "batch_size": BATCH_SIZE,
        "num_frames": len(frames),
        "base_frames": len(list(base_dir.glob("frame_*.jpg"))),
        "burst_frames": len(list(burst_dir.glob("frame_*.jpg"))),
        "failed_batches": failed_batches,
        "final_reason": final_reason,
        "batches": batch_results,
    }

    out_path = event_dir / "analysis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("✅ Analysis written to:", out_path)
    print(json.dumps({"final_abnormal_event": final, "final_confidence": max_conf}, indent=2))


if __name__ == "__main__":
    main()
