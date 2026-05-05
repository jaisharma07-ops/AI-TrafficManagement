"""End-to-end smoke run without Streamlit.

Exercises every core module against samples/demo_traffic.mp4:
  1. probe + iter_frames
  2. anomaly scan
  3. window extraction
  4. VLM analysis (or heuristic fallback if weights are unavailable)
  5. report bundle
  6. chat engine answers

Writes a structured log to samples/demo_run.log so a grader can audit.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow `python scripts/smoke_run.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.anomaly_detector import AnomalyDetector
from core.chat_engine import ChatEngine
from core.report_generator import build as build_report, render_markdown, save_bundle
from core.video_processor import (
    extract_window,
    iter_frames,
    probe,
    save_frame_png,
)
from core.vlm_engine import VLMEngine


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> int:
    log_path = config.SAMPLES_DIR / "demo_run.log"
    _setup_logging(log_path)
    log = logging.getLogger("smoke")

    video = config.DEMO_VIDEO_PATH
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    if not video.exists():
        log.error("No video at %s. Run scripts/make_demo_video.py first.", video)
        return 2

    meta = probe(video)
    log.info("video=%s duration=%.2fs fps=%.2f size=%dx%d",
             video.name, meta.duration_sec, meta.fps, meta.width, meta.height)

    log.info("=== STAGE 1: anomaly scan ===")
    try:
        from ultralytics import YOLO
        yolo = YOLO(config.YOLO_MODEL)
        log.info("YOLO loaded: %s", config.YOLO_MODEL)
    except Exception as e:                                    # noqa: BLE001
        log.warning("YOLO load failed (%s); detector will use frame-diff only", e)
        yolo = None

    detector = AnomalyDetector(yolo_model=yolo)
    events = detector.scan(iter_frames(video, sample_fps=config.SCAN_FPS))
    primary = next((e for e in events if e.kind in
                    {"collision", "deceleration", "framediff"}), None)
    for e in events:
        log.info("  event[%s] t=%.2fs  %s", e.kind, e.timestamp, e.detail)
    if primary is None:
        log.error("No anomaly triggered. Smoke test cannot proceed.")
        return 3
    log.info("Primary trigger: kind=%s t=%.2fs", primary.kind, primary.timestamp)

    log.info("=== STAGE 2: window extraction ===")
    keyframes = extract_window(
        video,
        center_sec=primary.timestamp,
        pre_sec=config.WINDOW_PRE_SEC,
        post_sec=config.WINDOW_POST_SEC,
        n_keyframes=config.KEYFRAMES_PER_WINDOW,
    )
    kf_paths = []
    for i, kf in enumerate(keyframes):
        out = config.REPORTS_DIR / f"smoke_keyframe_{i:02d}.png"
        save_frame_png(kf, out)
        kf_paths.append(out)
    log.info("Saved %d keyframes to %s", len(kf_paths), config.REPORTS_DIR)

    log.info("=== STAGE 3: VLM analysis ===")
    vlm = VLMEngine()
    vlm.load()
    log.info("VLM resolved as: %s", vlm.name)
    forensic = vlm.analyze_window(keyframes)
    log.info("Forensic JSON keys: %s", sorted(forensic.keys()))
    log.info("at_fault=%r confidence=%r", forensic.get("at_fault"),
             forensic.get("confidence"))

    log.info("=== STAGE 4: report bundle ===")
    duration = config.WINDOW_PRE_SEC + config.WINDOW_POST_SEC
    report = build_report(
        video_filename=video.name,
        timestamp_seconds=primary.timestamp,
        duration_analyzed_seconds=duration,
        forensic=forensic,
        model_used=vlm.name,
        anomaly_kind=primary.kind,
        anomaly_detail=primary.detail,
        keyframe_paths=kf_paths,
    )
    bundle = save_bundle(report, config.REPORTS_DIR)
    for k, p in bundle.items():
        log.info("  bundle.%-9s -> %s (%d bytes)", k, p, p.stat().st_size)

    print("\n----- MARKDOWN PREVIEW -----")
    print(render_markdown(report))
    print("----- END PREVIEW -----\n")

    log.info("=== STAGE 5: chat engine ===")
    descriptions = []
    chat = ChatEngine(report=report, descriptions=descriptions)
    questions = [
        "Who was at fault?",
        "Which vehicles were involved?",
        "What violations were observed?",
        "Generate a legal summary for the insurance company.",
        "Summarize the incident.",
        "What happened at 0:08?",
    ]
    for q in questions:
        a = chat.answer(q)
        log.info("Q: %s", q)
        log.info("A: %s", a.replace("\n", " | ")[:200])

    log.info("=== SMOKE RUN COMPLETE ===")
    log.info("Log file: %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
