"""Video player and keyframe strip — Aegis Forensic Intelligence design system."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from core.anomaly_detector import Detection
from core.video_processor import FrameSample


def render_player(video_path: Optional[Path]) -> None:
    """Render the video player or an empty-state placeholder."""
    if video_path is None or not Path(video_path).exists():
        st.markdown(
            "<div style='"
            "display:flex;flex-direction:column;align-items:center;"
            "justify-content:center;"
            "min-height:220px;"
            "border:1.5px dashed #1a2c42;"
            "border-radius:10px;"
            "text-align:center;padding:32px;"
            "'>"
            "<div style='font-size:40px;opacity:0.12;margin-bottom:14px;'>🎬</div>"
            "<div style='font-size:13px;color:#2a3d56;font-family:Inter,sans-serif;"
            "font-weight:500;letter-spacing:0.04em;'>"
            "Upload traffic footage to begin"
            "</div>"
            "<div style='font-size:11px;color:#1e2d45;margin-top:6px;"
            "font-family:Inter,sans-serif;'>"
            "MP4 · AVI · MOV &nbsp;·&nbsp; up to 200 MB"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
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
        cv2.rectangle(img, (x1, y1), (x2, y2), (59, 130, 246), 2)
        label = label_classes.get(d.cls, str(d.cls))
        cv2.putText(
            img, f"{label} {d.conf:.2f}", (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (59, 130, 246), 1, cv2.LINE_AA,
        )
    return img


def render_keyframe_strip(keyframe_paths: List[Path]) -> None:
    """Show keyframes inline with labels — highlights what the VLM analyzed."""
    if not keyframe_paths:
        return

    # Section label
    st.markdown(
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:14px;'>"
        "<div style='height:1px;flex:1;background:#1a2c42;'></div>"
        "<p style='font-size:9px;font-weight:700;letter-spacing:0.12em;"
        "color:#3a5070;text-transform:uppercase;margin:0;white-space:nowrap;"
        "font-family:Inter,sans-serif;'>Keyframes Analyzed by VLM</p>"
        "<div style='height:1px;flex:1;background:#1a2c42;'></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    n = min(4, len(keyframe_paths))
    cols = st.columns(n, gap="small")
    for i, (col, p) in enumerate(zip(cols, keyframe_paths[:n])):
        p = Path(p)
        if p.exists():
            col.image(str(p), use_column_width=True)
            col.markdown(
                f"<div style='text-align:center;margin-top:4px;'>"
                f"<span style='"
                f"font-size:9px;font-weight:600;letter-spacing:0.08em;"
                f"color:#2a3d56;font-family:Inter,sans-serif;text-transform:uppercase;"
                f"'>Frame {i + 1}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
