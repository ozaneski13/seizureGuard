"""Download the labeled evaluation corpus into data/eval/{seizure,normal}/.

All sources are openly licensed and directly downloadable (verified Aug 2026):
seizure clips are supplementary videos of peer-reviewed open-access veterinary
papers served from the PMC Open Data S3 bucket; normal-dog clips come from
Pexels (Pexels License) and Wikimedia Commons (CC BY / CC BY-SA — attribution
kept in this manifest; clips are used as local test inputs, not redistributed).
"""
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

EVAL_ROOT = Path("data/eval")
USER_AGENT = "seizureGuard-eval/1.0 (hobby research; contact: repo owner)"

# label, filename, url, license/attribution
MANIFEST = [
    # --- seizure class: real canine epileptic events from OA journal supplements
    ("seizure", "gtcs_compilation_frontiers2022.mp4",
     "https://pmc-oa-opendata.s3.amazonaws.com/PMC9097225.1/Video_1.MP4",
     "CC BY - Frontiers Vet Sci 2022 (PMC9097225) Suppl. Video 1, 4 GTCS events"),
    ("seizure", "gtcs_intense_frontiers2025.mov",
     "https://pmc-oa-opendata.s3.amazonaws.com/PMC12069445.1/Video_1.MOV",
     "CC BY - Frontiers Vet Sci 2025 (PMC12069445) Suppl. Video 1"),
    ("seizure", "gtcs_mild_missed_frontiers2025.mov",
     "https://pmc-oa-opendata.s3.amazonaws.com/PMC12069445.1/Video_2.MOV",
     "CC BY - Frontiers Vet Sci 2025 (PMC12069445) Suppl. Video 2, "
     "missed by a commercial detector"),
    ("seizure", "partridge_dog_seizure_mdpi2023.zip",
     "https://pmc-oa-opendata.s3.amazonaws.com/PMC10000155.1/animals-13-00810-s001.zip",
     "CC BY 4.0 - Animals (MDPI) 2023 (PMC10000155) Video S1 (zip)"),
    ("seizure", "lafora_myoclonus_plos2017.mp4",
     "https://pmc-oa-opendata.s3.amazonaws.com/PMC5540395.1/pone.0182024.s003.mp4",
     "CC BY - PLOS ONE 2017 (PMC5540395) S1 Movie, myoclonic epilepsy"),
    # --- normal class: everyday dog behavior incl. classic false-positive bait
    ("normal", "puppies_playing_meadow_pexels.mp4",
     "https://videos.pexels.com/video-files/3045714/3045714-hd_1920_1080_25fps.mp4",
     "Pexels License - pexels.com/video/dogs-playing-3045714"),
    ("normal", "labrador_walking_pexels.mp4",
     "https://videos.pexels.com/video-files/8746940/8746940-hd_1080_1920_25fps.mp4",
     "Pexels License - pexels.com/video/a-footage-of-a-dog-walking-8746940"),
    ("normal", "pomeranian_walking_pexels.mp4",
     "https://videos.pexels.com/video-files/5534310/5534310-hd_1080_1920_30fps.mp4",
     "Pexels License - pexels.com/video/dog-walking-outside-the-house-5534310"),
    ("normal", "scratch_reflex_commons.webm",
     "https://upload.wikimedia.org/wikipedia/commons/0/01/Scratch_reflex_demonstrated_by_Irish_Wolfhound_mix.webm",
     "CC BY-SA 3.0 - Wikimedia Commons, rhythmic limb motion (non-seizure)"),
    ("normal", "sleeping_dogs_twitching_commons.ogv",
     "https://upload.wikimedia.org/wikipedia/commons/2/23/A_domestic_dog_snoring.ogv",
     "CC BY-SA 4.0 - Wikimedia Commons, sleep twitching (hard negative)"),
    ("normal", "puppy_playing_indoors_commons.webm",
     "https://upload.wikimedia.org/wikipedia/commons/2/21/Puppy_playing.webm",
     "CC BY-SA 3.0 - Wikimedia Commons, attribution: Subhashish Panigrahi"),
    ("normal", "puppies_play_fight_commons.webm",
     "https://upload.wikimedia.org/wikipedia/commons/0/04/Goldendoodle_and_Black_Lab_puppies_play_fighting_in_slow_motion.webm",
     "CC BY-SA 3.0 - Wikimedia Commons, vigorous play"),
]


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    manifest_lines = []
    for label, name, url, license_note in MANIFEST:
        folder = EVAL_ROOT / label
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / name
        manifest_lines.append(f"{label}/{name}\n  {url}\n  {license_note}\n")
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[SKIP] {label}/{name} (already present)", flush=True)
            continue
        print(f"[GET ] {label}/{name}", flush=True)
        try:
            fetch(url, dest)
            print(f"       {dest.stat().st_size / 1e6:.1f} MB", flush=True)
        except Exception as e:
            print(f"[FAIL] {name}: {e}", flush=True)
            continue
        time.sleep(2)  # be polite, Commons rate-limits bursts

        if dest.suffix == ".zip":
            with zipfile.ZipFile(dest) as z:
                for member in z.namelist():
                    if Path(member).suffix.lower() in (".mp4", ".avi", ".mov", ".m4v"):
                        target = folder / (dest.stem + Path(member).suffix.lower())
                        target.write_bytes(z.read(member))
                        print(f"       extracted {target.name}", flush=True)
            dest.unlink()

    (EVAL_ROOT / "SOURCES.txt").write_text("\n".join(manifest_lines), encoding="utf-8")
    print("done; attribution manifest at", EVAL_ROOT / "SOURCES.txt")


if __name__ == "__main__":
    main()
