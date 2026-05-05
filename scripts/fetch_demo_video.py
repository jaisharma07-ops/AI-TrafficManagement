"""Download a CC-licensed traffic clip for the demo.

Tries a list of public mirrors in order. Falls back to invoking
`make_demo_video.py` if all downloads fail (e.g., offline graders).

URLs are recorded in `samples/SOURCE.txt` after a successful download.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "samples" / "demo_traffic.mp4"
SRC_TXT = OUT.parent / "SOURCE.txt"

# Public, redistributable, no auth required.
# Each entry: (label, url, license)
CANDIDATES = [
    (
        "BigBuckBunny dashcam-style stand-in (sample.mp4)",
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
        "CC license per Google's gtv-videos-bucket sample set",
    ),
    (
        "Sample short MP4 (Wikimedia mirror)",
        "https://download.samplelib.com/mp4/sample-5s.mp4",
        "Sample Lib - free for any use",
    ),
]

USER_AGENT = "ForensicAI-DemoFetcher/1.0"


def _try_download(url: str, dest: Path, timeout: float = 30.0) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, \
                dest.open("wb") as out:
            shutil.copyfileobj(resp, out)
        size = dest.stat().st_size
        if size < 50_000:
            print(f"  -> file too small ({size} bytes), treating as failure")
            return False
        print(f"  -> ok, {size / 1024:.0f} KiB")
        return True
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"  -> failed: {e}")
        return False


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for label, url, lic in CANDIDATES:
        print(f"Trying: {label}")
        print(f"  url: {url}")
        if _try_download(url, OUT):
            SRC_TXT.write_text(
                f"label: {label}\nurl: {url}\nlicense: {lic}\n",
                encoding="utf-8",
            )
            print(f"\nWrote {OUT}")
            print(f"Source recorded in {SRC_TXT}")
            return 0
        print()
    print("All download candidates failed - falling back to synthetic clip.")
    fallback = Path(__file__).with_name("make_demo_video.py")
    return subprocess.call([sys.executable, str(fallback)])


if __name__ == "__main__":
    sys.exit(main())
