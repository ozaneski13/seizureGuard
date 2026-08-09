# seizureGuard

Detects abnormal motor events (e.g., epileptic seizures) in pet dogs from a home monitoring camera.

> **Disclaimer:** This is a hobby prototype, not a medical or veterinary device. It does not diagnose anything. If your dog has seizures, talk to a veterinarian.

## How it works

```
camera / video file
      │
      ▼
continuous monitor ──► event capture ──► pose gate ──► VLM verify ──► alert
 (ring buffer +         2 fps base +      local YOLO     two-tier       Telegram or
  motion trigger        10 fps bursts     1.5-4.8 Hz     screen/confirm console
  FSM, 5s pre-roll)     at motion peaks   rhythm check   via Claude
```

1. **Monitor** watches the camera through a 90s ring buffer. Sustained motion (2s) arms an event; 10s of quiet (or a 60s cap) ends it. Full-frame lighting changes never trigger. Each event is saved with 5s of pre-roll as 2 fps base frames plus 10 fps bursts around the strongest motion peaks.
2. **Pose gate** (optional, local, free) runs dog pose estimation on the burst frames and measures rhythmic limb power in the 1.5–4.8 Hz band — the clinical signature of clonic jerking, capped by Nyquist at the 10 fps burst rate. Non-rhythmic events are skipped; everything ambiguous escalates (fail-open).
3. **Verify** sends frames to a vision model in chronological batches: a cheap *screen* model answers yes/no per batch, positives escalate to a *confirm* model that assesses specific canine seizure signs (paddling, tonic stiffening, rhythmic jerking, jaw clonus, loss of posture, fencing posture, plus soft signs). A deterministic rule layer decides; the event-level decision is a pure recall-first OR — nothing can veto a positive batch.
4. **Alert** goes to Telegram (with the peak frame photo) when configured, console otherwise.

## VLM backends

| Backend | Selected by | Requirements |
|---|---|---|
| `claude-cli` (default) | — | [Claude Code CLI](https://claude.com/claude-code) installed and logged in: run `claude` in a terminal once and use `/login` (works with a Claude subscription, no API key). Screen: `claude-haiku-4-5`, confirm: `claude-fable-5` — an A/B on real seizure footage showed sonnet-5 confirm missing all seizure batches that fable-5 flagged; raise `SEIZUREGUARD_SCREEN_MODEL` too for maximum recall at higher cost. |
| `openai` | `SEIZUREGUARD_BACKEND=openai` | `OPENAI_API_KEY` set; uses the Responses API with structured outputs (default model `gpt-4.1-mini`). |

The monitor auto-detects whether a backend is usable at startup (claude login or API key) and runs unverified motion alerts otherwise.

## Setup

```
pip install -r requirements.txt
```

Optional, for the local pose gate (isolated venv so the global torch stays untouched):

```
python -m venv .venv-pose
.venv-pose\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv-pose\Scripts\pip install ultralytics
.venv-pose\Scripts\yolo pose train data=dog-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640 device=0
copy runs\pose\train\weights\best.pt models\dog-pose.pt
```

> Note: install `ultralytics` **after** torch+torchvision from the cu128 index would still let pip replace them with CPU builds via the torchvision dependency — install torch and torchvision together from the cu128 index first (as above); if ultralytics later reports CPU-only torch, re-run the cu128 install line.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SEIZUREGUARD_BACKEND` | `claude-cli` | `claude-cli` or `openai` |
| `SEIZUREGUARD_SCREEN_MODEL` | `claude-haiku-4-5` | cheap per-batch screen model (claude backend) |
| `SEIZUREGUARD_CONFIRM_MODEL` | `claude-fable-5` | semiology confirm model (claude backend) |
| `SEIZUREGUARD_MODEL` | `gpt-4.1-mini` | model for the openai backend |
| `OPENAI_API_KEY` | — | required for the openai backend |
| `SEIZUREGUARD_VERIFY` | auto | `0` disables VLM verification, `1` forces it |
| `SEIZUREGUARD_POSE_PYTHON` | — | path to `.venv-pose\Scripts\python.exe`; enables the pose gate |
| `SEIZUREGUARD_POSE_MODEL` | `models/dog-pose.pt` | pose model weights |
| `SEIZUREGUARD_TG_TOKEN` / `SEIZUREGUARD_TG_CHAT` | — | Telegram bot token + chat id; unset = console alerts |

## Usage

```
# Continuous monitoring from the camera (index 0)
python src/monitor.py

# Replay a video file through the same pipeline (also used by the tests)
python src/monitor.py --source path\to\video.mp4

# Monitor an IP camera / go2rtc restream (H264 flavor decodes most reliably).
# Dropped streams reconnect automatically; a dead stream raises one
# "monitor blind" alert after 60s.
python src/monitor.py --source rtsp://127.0.0.1:8554/mi360_h264

# Print motion scores to calibrate MOTION_ON / MOTION_OFF for your room
python src/monitor.py --log-motion

# One-shot: capture a 60s event right now
python scripts/capture_event_from_camera.py

# One-shot: build an event from the first 60s of test_video.mp4
python scripts/extract_event_from_video.py

# Verify a captured event directory manually
python src/verify_event.py data/events/event_YYYYMMDD_HHMMSS_monitor

# Run the pose gate manually (inside the pose venv)
.venv-pose\Scripts\python src/pose_gate.py data/events/event_YYYYMMDD_HHMMSS_monitor
```

The verifier writes `analysis.json` into the event directory (final decision, per-batch screen/confirm verdicts, observed signs, failed batches); the pose gate writes `pose_gate.json`.

## Tests

```
python -m pytest
```

The suite is fully offline: it builds a synthetic 70s video (seizure-like jitter, a lighting flash that must never trigger, moderate motion), runs the real extraction and monitor pipeline on it, and stubs the VLM backends. No API key, no claude login, no GPU needed.

## Data & privacy

Everything under `data/` (plus `test_video.mp4`, `models/`, `runs/`, `.venv-pose/`) is gitignored — captured frames are private home footage and must never be committed. The pose gate doubles as a privacy mode: with verification disabled, no footage ever leaves the machine.

## Evaluation

First real-footage results (2026-08-09, n=11 public clips): all 4 clinical
seizure videos caught — including one documented as missed by a commercial
detector — and 6/7 normal-behavior clips correctly rejected; the single false
positive was sleep twitching. Method, per-clip table, and caveats in
[EVAL.md](EVAL.md).

## License

MIT (see [LICENSE](LICENSE)). Note: the optional pose gate and PTZ tracker
import [ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
at runtime — install it only if you use those components and mind its terms.

## Datasets & research

See [DATASETS.md](DATASETS.md) for the August 2026 survey of datasets, models, and methods relevant to vision-based canine seizure detection, with an ordered improvement roadmap.

Note: the K9-Bench false-positive probe from that roadmap is currently manual-only — the HF dataset is gated (terms acceptance required) and ships YouTube links rather than video files, so it can't be fetched automatically.
