"""Forensic AI - Streamlit dashboard.

3-pane layout per PRD §5:
    [Video player + bbox]   [Live log + Incident report]
    [Chat with video                                    ]

Run:
    streamlit run app.py

Pipeline steps wired from this file:
    1. User uploads .mp4 / .avi
    2. Anomaly detector scans at SCAN_FPS - logs to live panel
    3. On first trigger: extract window, run VLM forensic analysis
    4. Build report, save bundle, render right pane
    5. Generate per-2s descriptions for chat retrieval (best-effort)
    6. Chat answers grounded in report + descriptions
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Streamlit's auto-reloader chokes on torch.classes; force polling watcher.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")

import streamlit as st

import config
from core.anomaly_detector import AnomalyDetector, AnomalyEvent
from core.chat_engine import ChatEngine, FrameDescription
from core.report_generator import build as build_report, render_markdown, save_bundle
from core.video_processor import (
    FrameSample,
    extract_window,
    format_timestamp_hms,
    iter_frames,
    probe,
    save_frame_png,
)
from core.vlm_engine import VLMEngine
from ui import chat_panel, live_log, video_panel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("forensic-ai.app")


# --- Cached resources ---

@st.cache_resource(show_spinner=False)
def get_yolo():
    """Load YOLOv8 once per session."""
    from ultralytics import YOLO
    log.info("Loading YOLO model %s", config.YOLO_MODEL)
    return YOLO(config.YOLO_MODEL)


@st.cache_resource(show_spinner=False)
def get_vlm() -> VLMEngine:
    """Build the VLM facade (does not load weights yet - lazy)."""
    return VLMEngine()


# --- Session state ---

def _init_state() -> None:
    st.session_state.setdefault("uploaded_path", None)
    st.session_state.setdefault("report", None)
    st.session_state.setdefault("descriptions", [])
    st.session_state.setdefault("chat_engine", ChatEngine())
    st.session_state.setdefault("bundle_paths", None)
    st.session_state.setdefault("anomaly_event", None)
    st.session_state.setdefault("keyframe_paths", [])
    live_log.init_state()
    chat_panel.init_state()


# --- Pipeline driver ---

def _save_uploaded_file(uploaded_file) -> Path:
    target = config.SAMPLES_DIR / "uploaded.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def _scan_for_anomaly(video_path: Path) -> tuple[Optional[AnomalyEvent], List[AnomalyEvent]]:
    yolo = get_yolo()
    detector = AnomalyDetector(yolo_model=yolo)
    live_log.append(0.0, "info", "Loaded YOLOv8 - scanning at "
                                 f"{config.SCAN_FPS} fps...")
    n_frames = 0
    last_logged_sec = -1
    events: List[AnomalyEvent] = []
    for sample in iter_frames(video_path, sample_fps=config.SCAN_FPS):
        n_frames += 1
        # Per-second progress log (one entry per integer second)
        cur_sec = int(sample.timestamp)
        if cur_sec != last_logged_sec and cur_sec % 2 == 0:
            live_log.append(sample.timestamp, "info",
                            f"Scanning... t={sample.timestamp:.1f}s")
            last_logged_sec = cur_sec
        new_events = detector.scan([sample])
        for e in new_events:
            events.append(e)
            if e.kind == "deceleration":
                live_log.append(e.timestamp, "warn",
                                f"⚠ Sudden deceleration: {e.detail}")
            elif e.kind == "collision":
                live_log.append(e.timestamp, "crit",
                                f"🚨 COLLISION EVENT: {e.detail}")
            elif e.kind == "framediff":
                live_log.append(e.timestamp, "crit",
                                f"🚨 Anomaly (motion fallback): {e.detail}")
            elif e.kind == "error":
                live_log.append(e.timestamp, "warn", f"YOLO error: {e.detail}")

        # Stop early once we have a primary trigger
        primary = next((x for x in events if x.kind in
                        {"collision", "deceleration", "framediff"}), None)
        if primary is not None:
            break

    primary = next((x for x in events if x.kind in
                    {"collision", "deceleration", "framediff"}), None)
    if primary is None:
        live_log.append(0.0, "info",
                        f"Scan complete - {n_frames} frames analyzed, "
                        "no anomalies triggered.")
    else:
        live_log.append(primary.timestamp, "info",
                        f"Scan halted at first anomaly (t={primary.timestamp:.2f}s).")
    return primary, events


def _generate_descriptions(video_path: Path,
                           vlm: VLMEngine,
                           interval_sec: float) -> List[FrameDescription]:
    """Per-N-second one-line description for the chat retriever.

    Capped at 8 calls so a CPU demo doesn't take forever; expands the
    chat surface enough to answer time-based questions.
    """
    meta = probe(video_path)
    if meta.duration_sec <= 0:
        return []
    descs: List[FrameDescription] = []
    n_calls = min(8, max(1, int(meta.duration_sec / interval_sec)))
    if n_calls == 0:
        return descs
    timestamps = [meta.duration_sec * (i + 0.5) / n_calls for i in range(n_calls)]
    for ts in timestamps:
        try:
            frames = extract_window(
                video_path,
                center_sec=ts, pre_sec=0.05, post_sec=0.05,
                n_keyframes=1,
            )
        except Exception as e:                                 # noqa: BLE001
            log.warning("description extract failed at %.2fs: %s", ts, e)
            continue
        if not frames:
            continue
        text = vlm.describe_frame(frames[0])
        if text:
            descs.append(FrameDescription(timestamp=ts, description=text))
            live_log.append(ts, "info", f"Indexed clip @ t={ts:.1f}s")
    return descs


def run_pipeline(video_path: Path) -> None:
    """Full end-to-end run: scan -> trigger -> VLM -> report -> descriptions."""
    live_log.clear()
    live_log.append(0.0, "info", f"Loaded video: {video_path.name}")
    meta = probe(video_path)
    live_log.append(0.0, "info",
                    f"  duration={meta.duration_sec:.1f}s "
                    f"@ {meta.fps:.1f}fps  ({meta.width}x{meta.height})")

    if meta.duration_sec > config.MAX_VIDEO_LENGTH_SEC:
        live_log.append(0.0, "warn",
                        f"Video exceeds {config.MAX_VIDEO_LENGTH_SEC}s; "
                        "MVP analyzes prefix only.")

    primary, _ = _scan_for_anomaly(video_path)
    if primary is None:
        st.session_state.report = None
        return

    live_log.append(primary.timestamp, "info",
                    f"Extracting window: -{config.WINDOW_PRE_SEC}s..+{config.WINDOW_POST_SEC}s")
    keyframes = extract_window(
        video_path,
        center_sec=primary.timestamp,
        pre_sec=config.WINDOW_PRE_SEC,
        post_sec=config.WINDOW_POST_SEC,
        n_keyframes=config.KEYFRAMES_PER_WINDOW,
    )
    keyframe_paths: List[Path] = []
    for i, kf in enumerate(keyframes):
        p = config.REPORTS_DIR / f"keyframe_{i:02d}_t{kf.timestamp:.2f}.png"
        save_frame_png(kf, p)
        keyframe_paths.append(p)
    st.session_state.keyframe_paths = keyframe_paths

    vlm = get_vlm()
    live_log.append(primary.timestamp, "info",
                    f"Loading VLM ({vlm.model_name})...")
    vlm.load()
    live_log.append(primary.timestamp, "info",
                    f"VLM ready ({vlm.name}) - running forensic analysis...")
    forensic = vlm.analyze_window(keyframes)

    duration = config.WINDOW_PRE_SEC + config.WINDOW_POST_SEC
    report = build_report(
        video_filename=video_path.name,
        timestamp_seconds=primary.timestamp,
        duration_analyzed_seconds=duration,
        forensic=forensic,
        model_used=vlm.name,
        anomaly_kind=primary.kind,
        anomaly_detail=primary.detail,
        keyframe_paths=keyframe_paths,
    )
    st.session_state.report = report
    live_log.append(primary.timestamp, "ok",
                    f"Incident report generated (id {report.incident_id[:8]})")

    bundle = save_bundle(report, config.REPORTS_DIR)
    st.session_state.bundle_paths = bundle
    st.session_state.anomaly_event = primary
    live_log.append(primary.timestamp, "info",
                    f"Bundle saved: {bundle['json'].name}, "
                    f"{bundle['markdown'].name}, {bundle['legal'].name}")

    # Description index for chat
    live_log.append(primary.timestamp, "info",
                    "Generating description index for chat...")
    descs = _generate_descriptions(
        video_path, vlm, config.DESCRIPTION_INTERVAL_SEC,
    )
    st.session_state.descriptions = descs
    chat_engine = ChatEngine(report=report, descriptions=descs)
    st.session_state.chat_engine = chat_engine
    live_log.append(primary.timestamp, "ok",
                    f"Indexed {len(descs)} chat descriptions. Ready.")


# --- Layout ---

DARK_CSS = """
<style>
  .stApp { background-color: #0e1117; color: #fafafa; }
  .stMarkdown { color: #fafafa; }
  .block-container { padding-top: 1.5rem; }
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title=config.APP_TITLE,
        layout="wide",
        page_icon="🚨",
        initial_sidebar_state="collapsed",
    )
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    _init_state()

    st.markdown(f"### 🚨 {config.APP_TITLE}")
    st.caption(
        "Multimodal AI: anomaly detection + vision-language reasoning + grounded chat. "
        "Same class of model behind GPT-4V / Gemini, applied to forensic traffic analysis."
    )

    # Top row: upload + run
    upcol, runcol = st.columns([3, 1])
    uploaded = upcol.file_uploader(
        "Upload traffic footage (.mp4 / .avi) - or use samples/demo_traffic.mp4",
        type=["mp4", "avi", "mov"],
    )
    use_demo = runcol.button("Use demo video", use_container_width=True)

    if uploaded is not None:
        path = _save_uploaded_file(uploaded)
        st.session_state.uploaded_path = path
    elif use_demo and config.DEMO_VIDEO_PATH.exists():
        st.session_state.uploaded_path = config.DEMO_VIDEO_PATH
    elif use_demo:
        st.error(
            f"No demo video at {config.DEMO_VIDEO_PATH}. "
            "Run `python scripts/fetch_demo_video.py` first."
        )

    run_btn_disabled = st.session_state.uploaded_path is None
    if st.button("▶ Analyze video", type="primary", disabled=run_btn_disabled,
                 use_container_width=True):
        with st.status("Running forensic analysis...", expanded=True) as status:
            try:
                run_pipeline(st.session_state.uploaded_path)
                status.update(label="Analysis complete", state="complete")
            except Exception as e:                              # noqa: BLE001
                log.exception("Pipeline failed")
                status.update(label=f"Analysis failed: {e}", state="error")

    # Three-pane main area
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("#### VIDEO")
        video_panel.render_player(st.session_state.uploaded_path)
        if st.session_state.keyframe_paths:
            video_panel.render_keyframe_strip(st.session_state.keyframe_paths)

    with right:
        live_log.render()
        st.divider()
        st.markdown("##### INCIDENT REPORT")
        report = st.session_state.report
        if report is None:
            st.markdown(
                "<div style='opacity: 0.5;'>(awaiting incident detection)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(render_markdown(report))
            bundle = st.session_state.bundle_paths or {}
            dl_cols = st.columns(3)
            for col, key, label in [
                (dl_cols[0], "markdown", "📄 Markdown"),
                (dl_cols[1], "legal", "⚖ Legal"),
                (dl_cols[2], "json", "🧾 JSON"),
            ]:
                p = bundle.get(key)
                if p and Path(p).exists():
                    col.download_button(
                        label=label,
                        data=Path(p).read_bytes(),
                        file_name=Path(p).name,
                        use_container_width=True,
                    )

    st.divider()
    chat_panel.render(st.session_state.chat_engine)


if __name__ == "__main__":
    main()
