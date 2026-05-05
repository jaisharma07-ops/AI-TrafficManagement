"""Synthesize an offline demo video.

Generates a 640x360, 30 fps, ~15 second clip showing two colored
rectangles representing vehicles. The "red SUV" enters from the left
and accelerates; the "blue cyclist" enters from the right at moderate
speed. Around t = 8s the SUV decelerates sharply and overlaps the
cyclist's bounding box - this triggers either the collision IoU rule
(if YOLO somehow detects them) or the frame-diff fallback.

This file is intentionally self-contained - no project deps beyond cv2
and numpy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "samples" / "demo_traffic.mp4"


def _draw_road(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    # Asphalt
    frame[:] = (40, 40, 40)
    # Lane markings
    for y in range(20, h, 60):
        for x in range(20, w, 80):
            cv2.rectangle(frame, (x, y), (x + 30, y + 4), (200, 200, 200), -1)
    # Sky band
    cv2.rectangle(frame, (0, 0), (w, 60), (110, 80, 50), -1)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fps = 30
    duration = 15
    n_frames = fps * duration
    w, h = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT), fourcc, fps, (w, h))
    if not writer.isOpened():
        print("ERROR: cv2.VideoWriter failed to open the output file.")
        return 1

    # Vehicle A: red SUV - left to right, decelerates near collision frame
    # Vehicle B: blue cyclist - right to left, steady speed
    collision_frame = int(8.0 * fps)
    for i in range(n_frames):
        frame = np.empty((h, w, 3), dtype=np.uint8)
        _draw_road(frame)

        # Vehicle A position
        if i < collision_frame:
            x_a = int(60 + (i / collision_frame) * 240)
        else:
            # Slows abruptly to ~5 px / frame
            slow_progress = (i - collision_frame)
            x_a = int(300 + slow_progress * 0.4)
        cv2.rectangle(frame, (x_a, 200), (x_a + 80, 260), (0, 0, 220), -1)
        cv2.rectangle(frame, (x_a + 5, 205), (x_a + 25, 220), (180, 180, 220), -1)
        cv2.rectangle(frame, (x_a + 50, 205), (x_a + 75, 220), (180, 180, 220), -1)

        # Vehicle B position
        x_b = int(560 - (i / n_frames) * 280)
        cv2.rectangle(frame, (x_b, 215), (x_b + 30, 245), (220, 100, 30), -1)
        cv2.circle(frame, (x_b + 6, 248), 6, (40, 40, 40), -1)
        cv2.circle(frame, (x_b + 24, 248), 6, (40, 40, 40), -1)

        # HUD timestamp
        cv2.putText(frame, f"CAM-A  t={i / fps:5.2f}s",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    print(f"Wrote synthetic demo video: {OUT}")
    print(f"  duration: {duration}s @ {fps} fps  ({n_frames} frames)")
    print(f"  size:     {w}x{h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
