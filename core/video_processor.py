"""Video I/O helpers.

Wraps OpenCV so the rest of the pipeline can iterate frames at a fixed
sample rate, extract incident windows, and convert frames to PIL images
for the VLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List

import cv2
import numpy as np
from PIL import Image


@dataclass
class FrameSample:
    timestamp: float          # seconds from start of video
    frame_index: int          # native (un-sampled) frame index
    image_bgr: np.ndarray     # raw OpenCV frame (BGR)

    def to_pil(self) -> Image.Image:
        rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def probe(path: str | Path) -> VideoMeta:
    """Open the file just to read metadata, then close."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cv2 could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return VideoMeta(path=path, fps=fps, frame_count=frame_count,
                     width=width, height=height)


def iter_frames(path: str | Path, sample_fps: float) -> Generator[FrameSample, None, None]:
    """Yield frames sampled at approximately `sample_fps`.

    For a video at native 30 fps and `sample_fps=5`, every 6th frame is
    yielded. Falls back to yielding every frame if native fps is unknown.
    """
    meta = probe(path)
    cap = cv2.VideoCapture(str(meta.path))
    try:
        if meta.fps and sample_fps and sample_fps < meta.fps:
            stride = max(1, int(round(meta.fps / sample_fps)))
        else:
            stride = 1
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                ts = idx / meta.fps if meta.fps else float(idx)
                yield FrameSample(timestamp=ts, frame_index=idx, image_bgr=frame)
            idx += 1
    finally:
        cap.release()


def extract_window(
    path: str | Path,
    center_sec: float,
    pre_sec: float,
    post_sec: float,
    n_keyframes: int,
) -> List[FrameSample]:
    """Pull `n_keyframes` evenly spaced samples from [center-pre, center+post].

    Used to build the VLM forensic input. Boundaries are clamped to the
    video duration.
    """
    meta = probe(path)
    if meta.fps <= 0:
        raise ValueError(f"Cannot determine fps for {path}")

    # Reserve last frame as inclusive upper bound; some codecs fail to
    # decode the very last index when seeked.
    last_safe_idx = max(0, meta.frame_count - 2)
    start = max(0.0, center_sec - pre_sec)
    end = min(last_safe_idx / meta.fps, center_sec + post_sec)
    if end <= start:
        raise ValueError(f"Window is empty for center={center_sec}")

    targets = np.linspace(start, end, n_keyframes)
    cap = cv2.VideoCapture(str(meta.path))
    try:
        out: List[FrameSample] = []
        for t in targets:
            frame_idx = min(last_safe_idx, int(round(t * meta.fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                # Walk back a few frames as a fallback
                for back in range(1, 5):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - back))
                    ok, frame = cap.read()
                    if ok:
                        break
                if not ok:
                    continue
            out.append(FrameSample(
                timestamp=float(t),
                frame_index=frame_idx,
                image_bgr=frame,
            ))
        return out
    finally:
        cap.release()


def save_frame_png(sample: FrameSample, path: str | Path) -> Path:
    """Save the frame as a PNG (BGR -> file) for the report bundle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sample.image_bgr)
    return path


def format_timestamp_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m core.video_processor <video.mp4>")
        sys.exit(1)
    target = sys.argv[1]
    m = probe(target)
    print(f"Video: {m.path}")
    print(f"  size:     {m.width}x{m.height}")
    print(f"  fps:      {m.fps:.2f}")
    print(f"  frames:   {m.frame_count}")
    print(f"  duration: {m.duration_sec:.2f}s")
    n = sum(1 for _ in iter_frames(target, sample_fps=5.0))
    print(f"  sampled @ 5fps -> {n} frames")
