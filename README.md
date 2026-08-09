# seizureGuard

[![tests](https://github.com/ozaneski13/seizureGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/ozaneski13/seizureGuard/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Watches your dog through a home camera, 24/7. When it sees motion that looks
like an epileptic seizure, it sends a Telegram message to your phone with a
photo — verified by a vision AI that checks for actual canine seizure signs
(paddling, rigid posture, rhythmic jerking) before alerting you.

Evaluated on real clinical seizure footage: **4/4 published veterinary seizure
videos detected** — including one documented as missed by a commercial
detector — with 6/7 normal-behavior clips correctly rejected. Details and
caveats in [EVAL.md](EVAL.md).

> **Disclaimer:** This is a hobby prototype, not a medical or veterinary
> device. It does not diagnose anything. If your dog has seizures, talk to a
> veterinarian.

---

## How it works — the image processing, step by step

```
camera ──► 1. motion detection ──► 2. frame sampling ──► 3. pose gate ──► 4. AI verify ──► 5. alert
            (is something            (save the right      (optional,       (does it look     (Telegram
             moving?)                 frames)              local, free)     like a seizure?)   + photo)
```

### Step 1 — Motion detection (cheap, runs on every frame)

Every frame is compared with the previous one:

1. Convert both frames to grayscale, take the absolute pixel difference.
2. Blur the difference (kills camera noise), threshold it, and average the
   result — a 0–255 score proportional to how much of the image changed
   (~2.5 ≈ 1% of pixels). That single number is the **motion score**.
3. If *more than 60%* of the image changed at once, it's a lighting event
   (lamp switched on, auto-exposure) — never counted as motion.

A small state machine turns scores into events:

- Motion must stay above the threshold for **2 seconds** to start an event
  (a single frame spike is ignored).
- **10 seconds of quiet** ends the event; **60 seconds** is the hard cap.
- The **5 seconds before the trigger** are included from a rolling frame
  buffer, so you never lose the onset.

### Step 2 — Frame sampling (save little, save right)

Sending every frame to an AI would be slow and expensive. Instead, each event
is saved as:

- **Base frames at 2 fps** across the whole event (context), plus
- **Burst frames at 10 fps** around the strongest motion peaks — *and* around
  the best moment of every 30-second region that shows real motion, so a long
  subtle seizure can't be starved of coverage by one big spike elsewhere.

Frames are resized so the longest side is 640 px (aspect preserved).

### Step 3 — Pose gate (optional, local, free)

If you set up the pose model, a YOLO pose network finds the dog's keypoints
in the burst frames and a frequency analysis asks: *do the limbs oscillate
rhythmically in the 1.5–4.8 Hz band* — the clinical signature of clonic
jerking? Non-rhythmic events are dropped before spending AI calls.

The gate is strictly **fail-open**: no dog found, low confidence, spotty
detections, or any error ⇒ the event escalates to verification anyway. The
gate may only ever save money, never silently drop a seizure.

### Step 4 — Two-tier AI verification

Frames go to a vision model in chronological batches of 30:

1. **Screen tier** (cheap model): "any sign of an abnormal motor event?" —
   answers JSON `{"seen": yes/no, "confidence": 0..1, "posture": ...}`.
   A batch escalates if the answer is yes, the confidence isn't near zero,
   **or the dog is lying on its side** (the classic seizure posture — learned
   from real footage where a cheap model misread a convulsion as normal).
2. **Confirm tier** (strong model): assesses ten specific canine seizure
   signs — paddling, tonic stiffening, rhythmic jerking, jaw clonus, loss of
   posture, fencing posture, drooling, head tremor, muscle twitching,
   disorientation — each with body region and whether it was sustained.

A deterministic rule layer decides per batch: **1 hard sign, or 2 any signs,
or the model's own abnormal flag ⇒ positive.** The event-level decision is a
pure OR over batches — recall first, nothing can veto a positive.

### Step 5 — Alert

A positive event sends a Telegram message with the frame from the first
motion peak attached, so you can judge in two seconds whether to run home.
Without Telegram configured, alerts print to the console/log.

---

## Setup, from zero

### Step 0 — Install

Python 3.10+ required.

```
git clone https://github.com/ozaneski13/seizureGuard.git
cd seizureGuard
pip install -r requirements.txt
```

### Step 1 — Get a camera stream

Pick whichever matches your hardware:

**A. USB webcam** — nothing to set up:

```
python src/monitor.py --source 0
```

**B. A video file** (good first test — replay any clip through the full
pipeline):

```
python src/monitor.py --source path/to/video.mp4
```

**C. Wi-Fi / IP cameras** — run [go2rtc](https://github.com/AlexxIT/go2rtc)
(a single binary) to turn almost any camera into a local RTSP URL. Example
`go2rtc.yaml` for Xiaomi cameras (no jailbreak needed; go2rtc ≥ v1.9.14):

```yaml
rtsp:
  listen: "127.0.0.1:8554"
streams:
  dogcam: xiaomi://<xiaomi_account_id>:<region>@<camera_lan_ip>?did=<device_id>&model=<model_id>
  # Low-resolution substream - ideal: the pipeline works at 640 px anyway
  dogcam_sub: xiaomi://<...same...>&subtype=1
```

go2rtc walks you through the Xiaomi login in its web UI (`:1984`) on first
run; it also supports plain `rtsp://`, ONVIF, and many other camera types.
Then:

```
python src/monitor.py --source rtsp://127.0.0.1:8554/dogcam_sub
```

Tips learned the hard way: prefer the low-res substream (`subtype=1`) —
it's the pipeline's working resolution and decodes almost for free; dropped
streams reconnect automatically; if the stream stays dead for 60 s you get
one "monitor blind" alert.

### Step 2 — Connect the AI (choose one backend)

**Claude (default — works with a Claude subscription, no API key):**

1. Install [Claude Code](https://claude.com/claude-code).
2. Run `claude` in a terminal once and log in with `/login`.

That's it — the monitor auto-detects the login at startup. Screen tier uses
`claude-haiku-4-5`, confirm tier `claude-fable-5` (an A/B test on real
seizure footage showed weaker confirm models missing it — see FOLLOWUPS.md).

**OpenAI (alternative):** set `SEIZUREGUARD_BACKEND=openai` and
`OPENAI_API_KEY=<your key>`.

No backend at all? The monitor still runs and sends *unverified* motion
alerts.

### Step 3 — Telegram alerts

1. In Telegram, message **@BotFather** → send `/newbot` → pick a name and a
   username. BotFather replies with a **bot token** (`123456789:AA...`).
   Treat it like a password.
2. Message **@userinfobot** (any text) — it replies with your numeric
   **chat id**.
3. **Open your new bot's chat and press Start.** Don't skip this: Telegram
   bots cannot message you first, and without it every send fails with
   `chat not found`.
4. Set the two environment variables:

   ```
   # Windows (new terminals pick these up)
   setx SEIZUREGUARD_TG_TOKEN "<bot token>"
   setx SEIZUREGUARD_TG_CHAT  "<chat id>"

   # Linux / systemd service: add to the unit instead
   #   Environment=SEIZUREGUARD_TG_TOKEN=...
   #   Environment=SEIZUREGUARD_TG_CHAT=...
   ```

5. Test it — this should ping your phone:

   ```
   python src/alerts.py "hello from seizureGuard"
   ```

Unset variables = alerts fall back to the console. Delivery failures never
crash the monitor.

### Step 4 — Run

```
python src/monitor.py --source <your source> --name dogcam
```

You'll see `✅ Monitoring camera ... (verify: on)`. From then on: events are
captured to `data/events/`, verified, and alerted. Each event directory
contains the saved frames plus `analysis.json` — the full per-batch verdict
with observed signs, so you can always audit *why* it alerted (or didn't).

### Step 5 (optional) — The local pose gate

Needs a CUDA GPU for training (inference can run on CPU):

```
python -m venv .venv-pose
.venv-pose/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv-pose/Scripts/pip install ultralytics
.venv-pose/Scripts/yolo pose train data=dog-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640 device=0
copy runs/pose/train/weights/best.pt models/dog-pose.pt
```

Then point the monitor at that Python:
`SEIZUREGUARD_POSE_PYTHON=.venv-pose/Scripts/python.exe`.
(Install torch+torchvision together from the cu128 index *first* — installing
ultralytics alone will replace them with CPU builds.)

### Step 6 (optional) — Run it 24/7

- **Linux / Raspberry Pi (recommended):** create a systemd unit per camera —
  this project runs in production on a Pi 5 exactly this way:

  ```ini
  [Unit]
  Description=seizureGuard monitor (dogcam)
  After=network-online.target

  [Service]
  User=<you>
  WorkingDirectory=/home/<you>/seizureGuard
  ExecStart=/usr/bin/python3 -u src/monitor.py --source rtsp://127.0.0.1:8554/dogcam_sub --name dogcam
  Restart=always
  RestartSec=10

  [Install]
  WantedBy=multi-user.target
  ```

  `sudo systemctl enable --now seizureguard-dogcam`. The whole stack
  (restreamer + monitor) survives power cuts and reboots unattended.
- **Windows:** `scripts/start-seizureguard.ps1` keeps one hidden monitor per
  camera; wire it to a Startup shortcut plus an hourly scheduled task.

---

## Tuning

Calibrate the motion thresholds for *your* room — lighting, camera distance,
and dog size all matter:

```
python src/monitor.py --source <src> --log-motion
```

Watch the printed scores during normal activity, then adjust `MOTION_ON` /
`MOTION_OFF` in `src/monitor.py` so normal life sits below the trigger.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SEIZUREGUARD_BACKEND` | `claude-cli` | `claude-cli` or `openai` |
| `SEIZUREGUARD_SCREEN_MODEL` | `claude-haiku-4-5` | cheap per-batch screen model |
| `SEIZUREGUARD_CONFIRM_MODEL` | `claude-fable-5` | semiology confirm model |
| `SEIZUREGUARD_MODEL` | `gpt-4.1-mini` | model for the openai backend |
| `OPENAI_API_KEY` | — | required for the openai backend |
| `SEIZUREGUARD_VERIFY` | auto | `0` disables AI verification, `1` forces it |
| `SEIZUREGUARD_POSE_PYTHON` | — | path to the pose venv python; enables the gate |
| `SEIZUREGUARD_POSE_MODEL` | `models/dog-pose.pt` | pose model weights |
| `SEIZUREGUARD_TG_TOKEN` / `SEIZUREGUARD_TG_CHAT` | — | Telegram bot token + chat id |
| `SEIZUREGUARD_MOVING_FLAG` | — | flag file marking self-commanded PTZ motion |
| `SEIZUREGUARD_PTZ_URL` | `http://127.0.0.1:1985` | PTZ gateway base URL (tracker; gateway not included — see `src/tracker.py`) |
| `SEIZUREGUARD_TRACK_MODEL` | `yolo11n.pt` | detection model for the PTZ tracker |

## Tests

```
python -m pytest
```

Fully offline: builds a synthetic video (seizure-like jitter, a lighting
flash that must never trigger, moderate motion), runs the real extraction and
monitor pipeline on it, and stubs the AI backends. No API key, no login, no
GPU needed. 122 tests.

## Data & privacy

Everything under `data/` (captured frames, events, eval clips) plus `models/`
and the pose venv is gitignored — camera footage is private and must never be
committed. With verification disabled the footage never leaves your machine
at all; with it enabled, only sampled event frames are sent to the AI
backend you configured.

## Results & research

- [EVAL.md](EVAL.md) — evaluation on real clinical seizure videos and hard
  negatives (play, scratching, sleep twitching), with honest caveats.
- [DATASETS.md](DATASETS.md) — survey of datasets, models, and methods for
  vision-based canine seizure detection, with an improvement roadmap.
- [FOLLOWUPS.md](FOLLOWUPS.md) — engineering log: open items, measured
  findings, rejected experiments.

## License

MIT (see [LICENSE](LICENSE)). Note: the optional pose gate and PTZ tracker
import [ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
at runtime — install it only if you use those components and mind its terms.
