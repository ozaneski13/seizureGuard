"""VLM verification of a captured event directory: each 30-frame batch goes
to the confirm model, which assesses specific canine seizure signs; a
deterministic rule layer decides, recall-first. Writes analysis.json into
the event dir after every batch, `complete: false` until the last one; a
rerun over the same frames keeps the batches already answered and asks only
the rest (see recorded_batches).

Per-batch keys besides the verdict: a failed batch (abnormal_event None)
carries `error` and `error_kind` - "parse" (the model replied but no verdict
could be read; the error holds the first 2000 chars of the reply), "call"
(the CLI/API call failed; only these can be a backend outage) or "crash"
(an unexpected exception here). `salvaged: true` with `raw_reply` marks a
positive read out of an unparseable reply; its confidence is 0.0.
`multiple_verdicts: true` marks a reply that held more than one verdict
object (see _find_verdict).

A claude CLI login error still writes analysis.json (the remaining batches
failed, `backend_outage` set) and then exits non-zero.

Usage: python src/verify_event.py <event_dir>
"""
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BATCH_SIZE = 30
CONFIRM_ATTEMPTS = 2
MAX_ATTEMPTS = CONFIRM_ATTEMPTS  # openai backend

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
# Every sign below fired as a false positive on ordinary behaviour before it
# had a definition: a dog settling down read as "loss_of_posture", circling
# before lying down as "disorientation", motion blur as "drooling". The
# exclusions are the load-bearing half.
SIGN_DEFINITIONS = """Sign definitions - a sign is present ONLY if the movement is INVOLUNTARY:
- paddling: rhythmic swimming-like limb motion while lying down, unable to rise. NOT walking, running, digging or scratching.
- tonic_stiffening: sustained rigid extension of limbs/neck/trunk the dog cannot release. NOT stretching, bracing or standing still.
- rhythmic_jerking: repetitive involuntary whole-body or limb jerks (roughly 2-6 per second). NOT gait, shaking water off, or scratching.
- jaw_clonus: repetitive involuntary jaw chomping. NOT chewing, panting, yawning or licking.
- loss_of_posture: SUDDEN INVOLUNTARY COLLAPSE - the dog falls or its legs give way and it cannot get up. NOT lying down, settling to rest, rolling over, or sleeping. A dog that chooses to lie down has NOT lost posture.
- fencing_posture: one forelimb rigidly extended while the other is flexed, held that way.
- drooling: visible saliva strands or a clearly wet muzzle. NOT motion blur, shadow, or a dark patch.
- head_tremor: involuntary repetitive head oscillation while otherwise still. NOT looking around, sniffing, or shaking.
- muscle_twitching: localized involuntary muscle rippling. NOT normal movement or breathing.
- disorientation: post-seizure confusion - aimless pacing, bumping into objects, unresponsive staring. NOT sniffing, exploring, or circling before lying down.

Two-sided rule. Voluntary behaviour is never a sign: walking, trotting,
settling down, circling before lying down, stretching, scratching,
shaking off, sniffing and playing are normal however vigorous or blurred
they look, and blurred frames are evidence of speed, not of a sign.

Involuntary events are the opposite and MUST be flagged: a dog lying on
its side with rapid repetitive limb movement; legs giving way; thrashing
or rolling while unable to right itself; repetitive jerking at roughly
2-6 per second; rigid limbs held extended. Flag these even when
individual frames are blurred and even if the dog seems to move around
the room between episodes - seizures start and stop.

Duration matters for the rhythmic signs (paddling, rhythmic_jerking,
head_tremor, jaw_clonus, muscle_twitching): read it from the frame
timestamps. Movement seen for less than about 2 seconds is too short to
call rhythmic - a seizure's rhythmic phase keeps going. "sustained" means
it lasted at least that long. The on-screen clock is not evidence either
way: dogs lie down and roll at night too.

If you are genuinely torn between the two readings, flag it.
"""


VISION_CAUTIONS = (
    "Judge only the physical dog: rooms may contain mirrors or glass whose "
    "reflections look like a second dog — never base your verdict on a "
    "reflection. If the dog is partially out of frame, evaluate what is "
    "visible instead of assuming the hidden part moves normally.\n"
)


def get_config():
    return {
        "backend": os.environ.get("SEIZUREGUARD_BACKEND", "claude-cli"),
        # Confirm quality is what the alert rides on. 2026-08-09: sonnet-5
        # called all 6 seizure batches normal, fable-5 flagged them. 2026-09-26,
        # same prompt, owner's seizure: opus-5-5 alerted 5/5 colour + 2/2 gray
        # with 2-4 of 6 batches positive; fable-5 1/3 today (~60% over two
        # days); fable-5-1 0/5. opus-5-5 stayed silent on all 4 known false
        # alarms and alerted on 1 of 20 recent real negatives. Needs claude
        # CLI >= 2.1.283 (older versions reject the model id).
        "confirm_model": os.environ.get("SEIZUREGUARD_CONFIRM_MODEL", "claude-opus-5-5"),
        "openai_model": os.environ.get("SEIZUREGUARD_MODEL", "gpt-4.1-mini"),
    }


# ---------------------------------------------------------------- pure core

def frame_time(path):
    m = re.search(r"_t_([0-9.]+)s", path.name)
    return float(m.group(1)) if m else 0.0


def iter_batches(paths, batch_size):
    """Consecutive batches. A short remainder is never sent on its own: it
    becomes the last batch_size frames, overlapping the batch before. A
    3-frame tail (0.2 s) has no context, and the model flagged a dog's
    ordinary back-roll on one as "ambiguous" (false alarm, 2026-09-25 22:06)."""
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        if len(batch) < batch_size and i > 0:
            batch = paths[-batch_size:]
        yield batch


def strip_fences(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


RAW_EXCERPT_CHARS = 2000
SALVAGED_CONFIDENCE = 0.0


class VerdictParseError(ValueError):
    """The model replied, but no verdict could be read from the reply.
    Event-level by origin: never a backend outage, whatever the text says."""

    def __init__(self, reason, raw):
        super().__init__(f"Unparseable model reply ({reason}): {raw[:RAW_EXCERPT_CHARS]}")
        self.reason = reason


def _find_verdict(text):
    """The JSON object in text that carries a verdict, trying every '{'
    so prose around the object (even prose with braces) does not matter.
    Returns (verdict or None, first decode error or None). A lone sign
    entry is not a verdict: taking one from a broken reply reads as a
    clean negative.

    Several verdict objects (the prompt's no-dog example quoted, then the
    real answer) combine recall-first: the last positive one wins, else the
    last one, marked `multiple_verdicts`. Taking the first read a corrected
    positive as a clean negative."""
    decoder = json.JSONDecoder()
    first_error = None
    verdicts = []
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError as e:
            first_error = first_error or e
            continue
        if isinstance(obj, dict) and ("abnormal_event" in obj or "observed_signs" in obj):
            verdicts.append(obj)
    if not verdicts:
        return None, first_error
    if len(verdicts) == 1:
        return verdicts[0], None
    positive = [v for v in verdicts if decide_signs(v)]
    return dict((positive or verdicts)[-1], multiple_verdicts=True), None


def _salvage(text):
    """Recall-first last resort for a reply that does not parse: the signs
    readable in the raw text, as a verdict the rule layer sees as positive,
    or None when they would not be positive."""
    present = []
    for entry in re.findall(r"\{[^{}]*\}", text):
        sign = re.search(r'"sign"\s*:\s*"([a-z_]+)"', entry)
        if (sign and sign.group(1) in ALL_SIGNS and sign.group(1) not in present
                and re.search(r'"present"\s*:\s*true', entry)):
            present.append(sign.group(1))
    verdict = {
        "abnormal_event": bool(re.search(r'"abnormal_event"\s*:\s*true', text)),
        "confidence": SALVAGED_CONFIDENCE,
        "observed_signs": [{"sign": s, "present": True} for s in present],
        "note": "salvaged from an unparseable model reply",
        "salvaged": True,
        "raw_reply": text[:RAW_EXCERPT_CHARS],
    }
    return verdict if decide_signs(verdict) else None


def parse_json_verdict(text):
    """Verdict dict from the model's reply. Three live failures (2026-09-02,
    the grayscale run, 2026-09-25) were unescaped double quotes in the
    free-text note, the schema's last member; a strict parse of the whole
    reply threw away the decisive fields before it. So: parse leniently,
    then retry without the trailing note, then salvage a positive from the
    raw text. Raises VerdictParseError when nothing usable is left."""
    cleaned = strip_fences(text)
    verdict, error = _find_verdict(cleaned)
    if verdict is None:
        notes = list(re.finditer(r',?\s*"note"\s*:', cleaned))
        if notes:
            verdict, _ = _find_verdict(cleaned[:notes[-1].start()].rstrip() + "}")
        # A note placed before the signs cuts them off, and the rest reads
        # as a clean negative: a positive salvage wins, and a cut without
        # its signs is no verdict.
        if verdict is not None and not decide_signs(verdict):
            verdict = _salvage(text) or (verdict if "observed_signs" in verdict else None)
    elif not decide_signs(verdict):
        # A later object that did not decode (a corrected answer with the
        # usual inner quotes) may hold the positive.
        verdict = _salvage(text) or verdict
    if verdict is None:
        verdict = _salvage(text)
    if verdict is None:
        raise VerdictParseError(str(error) if error else "no JSON verdict object", text)
    # Sign evidence the rule layer cannot read is no clean negative: fail,
    # so the repair retry runs and a second odd reply alerts UNVERIFIED.
    if not decide_signs(verdict) and _signs_unreadable(verdict):
        raise VerdictParseError("observed_signs is not a list of {sign, present} objects", text)
    return verdict


# Failures that mean "the backend is unavailable right now" rather than
# "this event could not be analyzed". They repeat on every event, so the
# monitor announces them once instead of alerting per event.
BACKEND_OUTAGE_MARKERS = (
    "hit your limit",
    "usage limit",
    "rate limit",
    "too many requests",
    "quota",
    "overloaded",
    "not logged in",
    "token has expired",
    "authenticate",
)

# Batch "error_kind": where a failure came from. Only a failed call can be
# a backend outage; a reply that did not parse, or a crash in this script,
# is about this event. Batches written before the field existed have none.
ERR_PARSE, ERR_CALL, ERR_CRASH = "parse", "call", "crash"


def error_kind(exc):
    return ERR_PARSE if isinstance(exc, (VerdictParseError, json.JSONDecodeError)) else ERR_CALL


def is_backend_outage(error_text, kind=None):
    """True when an error means the whole backend is down, not this event."""
    if not error_text or kind not in (None, ERR_CALL):
        return False
    low = str(error_text).lower()
    if any(m in low for m in BACKEND_OUTAGE_MARKERS):
        return True
    # A bare "429" also matched JSON offsets ("char 1429"); trust it only
    # in the CLI's own error text.
    return low.startswith("claude cli") and re.search(r"\b429\b", low) is not None


def outage_reason(batch_results):
    """The shared outage reason across failed batches, or None."""
    reasons = []
    for r in batch_results:
        if r.get("abnormal_event") is not None:
            continue
        err = r.get("error") or (r.get("screen_verdict") or {}).get("error")
        if is_backend_outage(err, r.get("error_kind")):
            reasons.append(str(err))
    if not reasons:
        return None
    return max(set(reasons), key=reasons.count)[:200]


def coerce_confidence(value):
    """The model's confidence as a float in [0, 1]; null, text or NaN is 0.0."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(c):
        return 0.0
    return min(max(c, 0.0), 1.0)


def failed_batch(error, kind):
    return {"abnormal_event": None, "confidence": 0.0, "screen_verdict": None,
            "escalated": True, "observed_signs": [], "error": error,
            "error_kind": kind}


def _signs_unreadable(verdict):
    """observed_signs that is not a list, or a present entry without a
    string sign: the model's shape, whatever it is."""
    signs = verdict.get("observed_signs")
    if signs is None:
        return False
    if not isinstance(signs, list):
        return True
    return any(isinstance(s, dict) and s.get("present") and not isinstance(s.get("sign"), str)
               for s in signs)


def decide_signs(verdict):
    """Batch-level decision from a confirm verdict (recall-first rule layer).
    Never raises, the parser runs it on every verdict: a present entry whose
    sign is no string still counts, as an unnamed sign of its own."""
    signs = verdict.get("observed_signs")
    if not isinstance(signs, list):
        signs = []
    present = {s["sign"] if isinstance(s.get("sign"), str) else ("?", i)
               for i, s in enumerate(signs) if isinstance(s, dict) and s.get("present")}
    if bool(verdict.get("abnormal_event")):
        return True
    if present & HARD_SIGNS:
        return True
    return len(present) >= 2


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


def confirm_prompt(paths):
    return (
        "You are analyzing frames from a dog monitoring camera for signs of an "
        "abnormal motor event (possible epileptic seizure). This is NOT medical diagnosis.\n"
        f"Frame timestamps in seconds (same order as the attached images): {_timestamps(paths)}\n"
        "Frames are sampled at 2 fps normally and 10 fps during motion bursts, so gaps vary.\n"
        + VISION_CAUTIONS +
        "Assess each of these specific canine seizure signs across the sequence: "
        f"{', '.join(ALL_SIGNS)}.\n"
        + SIGN_DEFINITIONS +
        "We prefer false positives over misses, but only for genuinely "
        "involuntary movement - normal behaviour must read as normal.\n\n"
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

    # --no-session-persistence: without it every call writes a transcript
    # containing the base64 frames (~2-3 MB) under ~/.claude/projects —
    # 8 days of 24/7 monitoring filled 11 GB of a Pi SD card that way.
    proc = subprocess.run(
        [exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
         "--verbose", "--model", model, "--max-turns", "1",
         "--no-session-persistence",
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
    """fn(previous_exception) -> result; previous_exception is None on the
    first attempt. Returns (result, None) or (None, last exception)."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(last_error), None
        except ClaudeLoginError:
            raise
        except Exception as e:
            last_error = e
            print(f"[WARN] Batch {batch_index} {tier} attempt {attempt}/{attempts} failed: {e}",
                  flush=True)
            if attempt < attempts:
                time.sleep(2 ** attempt)
    return None, last_error


def repair_instruction(error):
    """Appended to a retry after an unparseable reply: resending the same
    prompt let the same habit (inner quotes in the note) fail again."""
    return ("\n\nYour previous reply to this request was not valid JSON "
            f"({error.reason}). Return only one valid JSON object. Never put "
            "double quotes inside string values; use single quotes there instead.")


def assess_batch_claude(event_dir, paths, config, batch_index):
    """Assess one batch with the confirm model.

    Until 2026-09-24 a cheap screen model looked at every batch first. It
    was measured blind to the owner's real seizure (haiku answered "no" on
    all six batches, "normal purposeful walking"), and both attempts to let
    it filter silenced that seizure, so its gate became a no-op. After that
    it only cost an extra call per batch and supplied the note the event
    viewer shows - a wrong explanation on exactly the events that matter.
    Do not reintroduce a cheap pre-filter without re-running the reference
    events in FOLLOWUPS. `screen_verdict` stays in analysis.json as None."""
    def attempt(previous_error):
        prompt = confirm_prompt(paths)
        if isinstance(previous_error, VerdictParseError):
            prompt += repair_instruction(previous_error)
        return run_claude(prompt, config["confirm_model"], images=paths, timeout=600)

    confirm, confirm_error = _with_retry(attempt, batch_index, "confirm", CONFIRM_ATTEMPTS)
    if confirm is None:
        return failed_batch(str(confirm_error), error_kind(confirm_error))
    result = {
        "abnormal_event": decide_signs(confirm),
        "confidence": coerce_confidence(confirm.get("confidence")),
        "screen_verdict": None,
        "escalated": True,
        "observed_signs": confirm.get("observed_signs") or [],
        "posture": confirm.get("posture"),
        "partially_visible": confirm.get("partially_visible"),
        "note": confirm.get("note"),
    }
    if confirm.get("salvaged"):
        result["salvaged"] = True
        result["raw_reply"] = confirm.get("raw_reply")
    if confirm.get("multiple_verdicts"):
        result["multiple_verdicts"] = True
    return result


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
        lambda _previous: ask_openai(client, event_dir, paths, config["openai_model"]),
        batch_index, "confirm", MAX_ATTEMPTS,
    )
    if verdict is None:
        return failed_batch(str(error), error_kind(error))
    return {
        "abnormal_event": decide_signs(verdict),
        "confidence": coerce_confidence(verdict.get("confidence")),
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


def recorded_batches(out_path, frame_names):
    """{batch number: result} for the batches an interrupted run over the
    same frames already answered. A finished analysis (or one over other
    frames) is not resumed: the monitor only reruns verify when it chose
    to. A failed batch is asked again; a verdict never is, since a second
    answer could silence a positive the first one found."""
    try:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if (not isinstance(prev, dict) or prev.get("complete") is not False
            or prev.get("frames") != frame_names or prev.get("batch_size") != BATCH_SIZE
            or not isinstance(prev.get("batches"), list)):
        return {}
    return {bi: r for bi, r in enumerate(prev["batches"], start=1)
            if isinstance(r, dict) and r.get("abnormal_event") is not None}


def write_analysis(out_path, out):
    # Written beside it, then renamed over it: the monitor reuses an existing
    # analysis.json, so no reader may ever see a half-written one.
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp_path, out_path)


def summarize(batch_results, login_error, complete):
    """Event-level fields over the batches answered so far."""
    # Event-level decision (recall-focused): pure OR over analyzed batches.
    # A seizure that ended is still a seizure that happened — nothing may veto a true.
    any_true = any(r.get("abnormal_event") is True for r in batch_results)
    failed_batches = sum(1 for r in batch_results if r.get("abnormal_event") is None)
    # Confidence OF THE FINDING. Taking the max over all analyzed batches
    # let a plainly normal event report 0.85 (the confirm model reports
    # confidence in its own verdict, not seizure probability), and an alert
    # quoted 0.85 while its single positive batch was only 0.45.
    positive_conf = [float(r.get("confidence", 0.0)) for r in batch_results
                     if r.get("abnormal_event") is True]
    analyzed_conf = [float(r.get("confidence", 0.0)) for r in batch_results
                     if r.get("abnormal_event") is not None]
    max_conf = max(positive_conf) if positive_conf else (
        max(analyzed_conf) if analyzed_conf else 0.0)

    if any_true:
        final_reason = "Any batch true (recall-focused OR rule)."
    elif failed_batches:
        final_reason = (
            f"No batch marked abnormal_event=true, but {failed_batches} batch(es) failed "
            "and were never analyzed — treat this result as incomplete."
        )
    elif not complete:
        final_reason = (f"No batch marked abnormal_event=true in the {len(batch_results)} "
                        "analyzed so far — the run is incomplete.")
    else:
        final_reason = "No batch marked abnormal_event=true."

    return {
        "final_abnormal_event": any_true,
        "final_confidence": max_conf,
        "complete": complete,
        "positive_batches": len(positive_conf),
        "backend_outage": (str(login_error)[:200] if login_error is not None
                           else outage_reason(batch_results)),
        "failed_batches": failed_batches,
        "final_reason": final_reason,
    }


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

    frame_names = [p.relative_to(event_dir).as_posix() for p in frames]
    batches = list(iter_batches(frames, BATCH_SIZE))
    out_path = event_dir / "analysis.json"
    recorded = recorded_batches(out_path, frame_names)
    base = {
        "backend": config["backend"],
        "confirm_model": (config["confirm_model"] if config["backend"] == "claude-cli"
                          else config["openai_model"]),
        "batch_size": BATCH_SIZE,
        "num_frames": len(frames),
        "base_frames": len(list(base_dir.glob("frame_*.jpg"))),
        "burst_frames": len(list(burst_dir.glob("frame_*.jpg"))),
        "frames": frame_names,
    }

    batch_results = []
    login_error = None
    for bi, batch in enumerate(batches, start=1):
        # One odd reply must not kill the run: analysis.json would never be
        # written and the other batches' positives would be lost with it.
        # A login error fails this batch and every later one without another
        # call; analysis.json still records it as a backend outage, so the
        # monitor announces it once instead of alerting on every event.
        if bi in recorded:
            r = recorded[bi]
            print(f"[INFO] Batch {bi}: kept from the interrupted run", flush=True)
        elif login_error is not None:
            r = failed_batch(str(login_error), ERR_CALL)
        else:
            try:
                r = assess(batch, bi)
            except ClaudeLoginError as e:
                login_error = e
                r = failed_batch(str(e), ERR_CALL)
            except Exception as e:
                print(f"[WARN] Batch {bi} crashed: {type(e).__name__}: {e}", flush=True)
                r = failed_batch(f"{type(e).__name__}: {e}", ERR_CRASH)
        batch_results.append(r)
        print(f"[INFO] Batch {bi}: abnormal={r['abnormal_event']} "
              f"conf={r['confidence']:.2f} escalated={r['escalated']}", flush=True)
        # After every batch, so a restart or the monitor's timeout loses at
        # most the batch in flight, and a positive found so far is on disk.
        out = summarize(batch_results, login_error, complete=bi == len(batches))
        write_analysis(out_path, {**out, **base, "batches": batch_results})

    print("✅ Analysis written to:", out_path)
    print(json.dumps({"final_abnormal_event": out["final_abnormal_event"],
                      "final_confidence": out["final_confidence"]}, indent=2))
    if login_error is not None:
        raise login_error


if __name__ == "__main__":
    main()
