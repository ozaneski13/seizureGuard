import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from synthetic import make_video  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "test_video.mp4"
    make_video(path)
    return path


@pytest.fixture(scope="session")
def synthetic_event_dir(synthetic_video, tmp_path_factory):
    """Run the real extract script on the synthetic video, once per session."""
    workdir = tmp_path_factory.mktemp("event")
    shutil.copy(synthetic_video, workdir / "test_video.mp4")
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "extract_event_from_video.py")],
        cwd=workdir, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    events = sorted((workdir / "data" / "test_events").glob("event_*"))
    assert len(events) == 1, f"expected one event dir, got {events}"
    return events[0]
