"""Central configuration for Forensic AI.

All tunable thresholds, model names, and paths live here so other modules
stay declarative.
"""

import os
from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT_DIR / "samples"
PROMPTS_DIR = ROOT_DIR / "prompts"
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DEMO_VIDEO_PATH = SAMPLES_DIR / "demo_traffic.mp4"

# --- Hardware auto-detection ---
# Probe CUDA at import time so every module sees the same device.
# FORENSIC_DEVICE env var overrides ("cuda", "cpu") for CI / debugging.
def _detect_device() -> tuple[str, str]:
    override = os.environ.get("FORENSIC_DEVICE", "").lower()
    if override in {"cpu", "cuda"}:
        if override == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda", "float16"
            except Exception:
                pass
            return "cpu", "float32"
        return "cpu", "float32"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "float32"


DEVICE, TORCH_DTYPE = _detect_device()
ON_GPU = DEVICE == "cuda"

# --- Model identifiers ---
YOLO_MODEL = "yolov8n.pt"  # auto-downloaded by ultralytics
VLM_MODEL = "vikhyatk/moondream2"
VLM_REVISION = "2024-08-26"  # pin for reproducibility

# --- Video sampling ---
# On GPU we can afford to scan denser; CPU stays at 5 fps for responsiveness.
SCAN_FPS = 8 if ON_GPU else 5
WINDOW_PRE_SEC = 5.0    # seconds before anomaly to include in window
WINDOW_POST_SEC = 2.0   # seconds after anomaly to include in window
KEYFRAMES_PER_WINDOW = 4

# --- Anomaly thresholds (PRD §4.1) ---
IOU_OVERLAP_THRESHOLD = 0.30
IOU_CONSECUTIVE_FRAMES = 2
DECEL_DROP_RATIO = 0.70           # >70% velocity drop
DECEL_WINDOW_SEC = 0.5
DISAPPEAR_GRACE_FRAMES = 3        # frames a track may be missing before flagged

# Vehicle classes from COCO (used by YOLOv8n)
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
PERSON_CLASS_ID = 0

# --- VLM ---
VLM_MAX_NEW_TOKENS = 512
# YOLO inference batch size (GPU only — CPU runs frame-by-frame anyway).
YOLO_BATCH = 8 if ON_GPU else 1
YOLO_IMGSZ = 640

# --- Chat / description sampling ---
# More chat descriptions are cheap on GPU; cap higher to enrich retrieval.
DESCRIPTION_INTERVAL_SEC = 2.0   # one VLM micro-description per N sec for chat
DESCRIPTION_MAX_CALLS = 16 if ON_GPU else 8

# --- App ---
APP_TITLE = "Forensic AI - Traffic Incident Analyst"
MAX_VIDEO_LENGTH_SEC = 120
