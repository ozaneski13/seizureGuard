import cv2
from pathlib import Path

VIDEO_PATH = Path("test_video.mp4")  # put your video at project root
OUTPUT_DIR = Path("data/test_event_1min_2fps")

EVENT_SECONDS = 60.0
SAMPLE_FPS = 2.0
SAMPLE_INTERVAL = 1.0 / SAMPLE_FPS  # 0.5s

WIDTH, HEIGHT = 640, 360

def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print("Looking for video at:", VIDEO_PATH.resolve())
    if not VIDEO_PATH.exists():
        raise RuntimeError(f"Video file not found: {VIDEO_PATH.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Could not open video (codec or path issue)")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps is None or fps <= 0 or total_frames <= 0:
        raise RuntimeError("Could not read fps/frame_count from video")

    duration = total_frames / fps
    if duration <= 0:
        raise RuntimeError("Video duration is invalid")

    # Use first 60 seconds (or full duration if shorter)
    window = min(EVENT_SECONDS, duration)

    timestamps = [i * SAMPLE_INTERVAL for i in range(int(window * SAMPLE_FPS))]

    saved = 0
    for idx, t in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Frame read failed at t={t:.3f}s")
            continue

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        out_path = OUTPUT_DIR / f"frame_{idx:03d}_t_{t:0.3f}s.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    print("✅ Saved frames:", saved)
    print("✅ Output folder:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()
