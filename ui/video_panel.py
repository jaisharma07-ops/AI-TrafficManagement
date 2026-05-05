"""Video player + bounding-box overlay panel."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from core.anomaly_detector import Detection
from core.video_processor import FrameSample


def render_player(video_path: Optional[Path]) -> None:
    if video_path is None or not Path(video_path).exists():
        st.info("Upload a traffic video to begin analysis.")
        return
    st.video(str(video_path))


def annotated_keyframe(
    frame: FrameSample,
    detections: List[Detection],
    *,
    label_classes: dict[int, str] | None = None,
) -> np.ndarray:
    """Return a copy of the frame with bounding boxes drawn (BGR)."""
    img = frame.image_bgr.copy()
    label_classes = label_classes or {}
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = label_classes.get(d.cls, str(d.cls))
        cv2.putText(
            img, f"{label} {d.conf:.2f}", (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
        )
    return img


def render_keyframe_strip(keyframe_paths: List[Path]) -> None:
    """Show keyframes inline; highlights what the VLM saw."""
    if not keyframe_paths:
        return
    st.markdown("##### Keyframes analyzed by the VLM")
    cols = st.columns(min(4, len(keyframe_paths)))
    for col, p in zip(cols, keyframe_paths):
        if Path(p).exists():
            col.image(str(p), use_container_width=True)
