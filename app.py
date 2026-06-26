"""Forensic AI — Streamlit dashboard.

Aegis Forensic Intelligence design system (Stitch-generated).
3-pane layout per PRD §5:
    [Video player + keyframes]   [Live log + Incident report]
    [Chat with video — full width                            ]

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


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

def _save_uploaded_file(uploaded_file) -> Path:
    target = config.SAMPLES_DIR / "uploaded.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def _scan_for_anomaly(video_path: Path) -> tuple[Optional[AnomalyEvent], List[AnomalyEvent]]:
    yolo = get_yolo()
    detector = AnomalyDetector(yolo_model=yolo)
    live_log.append(0.0, "info", f"YOLOv8 loaded · scanning @ {config.SCAN_FPS} fps")
    n_frames = 0
    last_logged_sec = -1
    events: List[AnomalyEvent] = []
    for sample in iter_frames(video_path, sample_fps=config.SCAN_FPS):
        n_frames += 1
        cur_sec = int(sample.timestamp)
        if cur_sec != last_logged_sec and cur_sec % 2 == 0:
            live_log.append(sample.timestamp, "info",
                            f"Scanning… t = {sample.timestamp:.1f}s")
            last_logged_sec = cur_sec
        new_events = detector.scan([sample])
        for e in new_events:
            events.append(e)
            if e.kind == "deceleration":
                live_log.append(e.timestamp, "warn",
                                f"Sudden deceleration detected · {e.detail}")
            elif e.kind == "collision":
                live_log.append(e.timestamp, "crit",
                                f"COLLISION EVENT · {e.detail}")
            elif e.kind == "framediff":
                live_log.append(e.timestamp, "crit",
                                f"Anomaly (motion fallback) · {e.detail}")
            elif e.kind == "error":
                live_log.append(e.timestamp, "warn", f"YOLO error · {e.detail}")

        primary = next((x for x in events if x.kind in
                        {"collision", "deceleration", "framediff"}), None)
        if primary is not None:
            break

    primary = next((x for x in events if x.kind in
                    {"collision", "deceleration", "framediff"}), None)
    if primary is None:
        live_log.append(0.0, "info",
                        f"Scan complete · {n_frames} frames · no anomalies triggered")
    else:
        live_log.append(primary.timestamp, "info",
                        f"Scan halted at first anomaly (t = {primary.timestamp:.2f}s)")
    return primary, events


def _generate_descriptions(video_path: Path,
                           vlm: VLMEngine,
                           interval_sec: float) -> List[FrameDescription]:
    """Per-N-second description for the chat retriever."""
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
            live_log.append(ts, "info", f"Indexed clip @ t = {ts:.1f}s")
    return descs


def run_pipeline(video_path: Path) -> None:
    """Full end-to-end run: scan → trigger → VLM → report → descriptions."""
    live_log.clear()
    live_log.append(0.0, "info", f"Loaded · {video_path.name}")
    meta = probe(video_path)
    live_log.append(0.0, "info",
                    f"Duration {meta.duration_sec:.1f}s · "
                    f"{meta.fps:.1f} fps · {meta.width}×{meta.height}")

    if meta.duration_sec > config.MAX_VIDEO_LENGTH_SEC:
        live_log.append(0.0, "warn",
                        f"Video exceeds {config.MAX_VIDEO_LENGTH_SEC}s — analyzing prefix only")

    primary, _ = _scan_for_anomaly(video_path)
    if primary is None:
        st.session_state.report = None
        return

    live_log.append(primary.timestamp, "info",
                    f"Extracting window: −{config.WINDOW_PRE_SEC}s … +{config.WINDOW_POST_SEC}s")
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
                    f"Loading VLM ({vlm.model_name})…")
    vlm.load()
    live_log.append(primary.timestamp, "info",
                    f"VLM ready ({vlm.name}) · running forensic analysis…")
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
                    f"Incident report generated · ID {report.incident_id[:8]}")

    bundle = save_bundle(report, config.REPORTS_DIR)
    st.session_state.bundle_paths = bundle
    st.session_state.anomaly_event = primary
    live_log.append(primary.timestamp, "info",
                    f"Bundle saved · {bundle['json'].name}")

    live_log.append(primary.timestamp, "info",
                    "Generating description index for chat…")
    descs = _generate_descriptions(
        video_path, vlm, config.DESCRIPTION_INTERVAL_SEC,
    )
    st.session_state.descriptions = descs
    chat_engine = ChatEngine(report=report, descriptions=descs)
    st.session_state.chat_engine = chat_engine
    live_log.append(primary.timestamp, "ok",
                    f"Indexed {len(descs)} chat descriptions · Ready")


# ---------------------------------------------------------------------------
# Aegis Forensic Intelligence — Design System CSS
# ---------------------------------------------------------------------------

_AEGIS_CSS = """
<style>
  /* --- Google Fonts: Inter + JetBrains Mono --- */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Design Tokens ───────────────────────────────────────── */
  :root {
    --bg:            #080d1a;
    --bg-subtle:     #0a1020;
    --surface:       #0d1526;
    --surface-2:     #101d30;
    --surface-3:     #132035;
    --border:        #1e2d45;
    --border-soft:   #15243a;
    --border-bright: #283d58;
    --text-1:        #e2e8f0;
    --text-2:        #94a3b8;
    --text-3:        #4a6180;
    --text-4:        #2a3d56;
    --accent-blue:   #3b82f6;
    --accent-blue-dim: rgba(59,130,246,0.12);
    --accent-red:    #ef4444;
    --accent-red-dim:  rgba(239,68,68,0.12);
    --accent-amber:  #f59e0b;
    --accent-amber-dim: rgba(245,158,11,0.12);
    --accent-green:  #22c55e;
    --accent-green-dim: rgba(34,197,94,0.12);
    --radius-xl:     18px;
    --radius-lg:     14px;
    --radius-md:     10px;
    --radius-sm:     7px;
    --radius-xs:     4px;
    --shadow-card:   0 4px 40px rgba(0,0,0,0.55), 0 1px 0 rgba(255,255,255,0.03) inset;
    --shadow-glow-red:  0 0 32px rgba(239,68,68,0.18);
    --shadow-glow-blue: 0 0 24px rgba(59,130,246,0.15);
  }

  /* ── Global Reset ────────────────────────────────────────── */
  *, *::before, *::after { box-sizing: border-box; }

  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stApp"] {
    background-color: var(--bg) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: var(--text-1) !important;
  }

  /* Remove default Streamlit container padding */
  .block-container {
    padding: 0 !important;
    max-width: 100% !important;
  }
  [data-testid="stAppViewContainer"] > .main {
    padding: 0 !important;
  }
  [data-testid="stVerticalBlock"] {
    gap: 0 !important;
  }

  /* Ensure main content area allows overflow for cards */
  section[data-testid="stSidebar"] { display: none !important; }
  .main .block-container { overflow: visible !important; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden !important; }
  [data-testid="stDecoration"] { display: none !important; }

  /* Fix column overflow for card labels */
  [data-testid="stHorizontalBlock"] { overflow: visible !important; }
  [data-testid="stVerticalBlockBorderWrapper"] { overflow: visible !important; }

  /* ── Scrollbar ───────────────────────────────────────────── */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 999px;
  }
  ::-webkit-scrollbar-thumb:hover { background: var(--border-bright); }

  /* ── Typography ──────────────────────────────────────────── */
  h1, h2, h3, h4, h5, h6,
  [data-testid="stMarkdownContainer"] h1,
  [data-testid="stMarkdownContainer"] h2,
  [data-testid="stMarkdownContainer"] h3,
  [data-testid="stMarkdownContainer"] h4 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-1) !important;
    letter-spacing: -0.025em !important;
    margin-top: 0 !important;
  }
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li {
    color: var(--text-2) !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    font-family: 'Inter', sans-serif !important;
  }
  [data-testid="stMarkdownContainer"] strong {
    color: var(--text-1) !important;
    font-weight: 600 !important;
  }
  [data-testid="stMarkdownContainer"] code {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-xs) !important;
    color: var(--accent-blue) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    padding: 1px 5px !important;
  }

  /* ── File Uploader ───────────────────────────────────────── */
  [data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border-bright) !important;
    border-radius: var(--radius-md) !important;
    padding: 4px 12px !important;
    transition: border-color 0.2s, background 0.2s !important;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: var(--accent-blue) !important;
    background: var(--surface-2) !important;
  }
  [data-testid="stFileUploader"] label,
  [data-testid="stFileUploader"] p,
  [data-testid="stFileUploader"] span,
  [data-testid="stFileUploaderDropzoneInstructions"] span {
    color: var(--text-3) !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
  }
  [data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
  }

  /* ── Buttons ─────────────────────────────────────────────── */
  [data-testid="stButton"] > button,
  [data-testid="stDownloadButton"] > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.01em !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.18s cubic-bezier(0.4,0,0.2,1) !important;
    padding: 9px 20px !important;
  }

  /* Primary — red gradient */
  [data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(239,68,68,0.4), 0 1px 0 rgba(255,255,255,0.12) inset !important;
  }
  [data-testid="stButton"] > button[kind="primary"]:hover:not(:disabled) {
    box-shadow: 0 6px 32px rgba(239,68,68,0.55), 0 1px 0 rgba(255,255,255,0.12) inset !important;
    transform: translateY(-1px) !important;
  }
  [data-testid="stButton"] > button[kind="primary"]:active:not(:disabled) {
    transform: translateY(0) !important;
    box-shadow: 0 2px 12px rgba(239,68,68,0.4) !important;
  }
  [data-testid="stButton"] > button[kind="primary"]:disabled {
    background: var(--surface-3) !important;
    color: var(--text-4) !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
  }

  /* Secondary / ghost */
  [data-testid="stButton"] > button[kind="secondary"],
  [data-testid="stButton"] > button:not([kind]) {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-2) !important;
  }
  [data-testid="stButton"] > button[kind="secondary"]:hover,
  [data-testid="stButton"] > button:not([kind]):hover {
    border-color: var(--accent-blue) !important;
    color: var(--accent-blue) !important;
    background: var(--accent-blue-dim) !important;
  }

  /* Download buttons */
  [data-testid="stDownloadButton"] > button {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-3) !important;
    font-size: 12px !important;
    padding: 7px 12px !important;
  }
  [data-testid="stDownloadButton"] > button:hover {
    border-color: var(--accent-blue) !important;
    color: var(--accent-blue) !important;
    background: var(--accent-blue-dim) !important;
  }

  /* ── Chat ────────────────────────────────────────────────── */
  [data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: 10px !important;
    padding: 12px 16px !important;
  }
  [data-testid="stChatMessage"][data-testid*="user"] {
    border-color: var(--border-bright) !important;
  }
  [data-testid="stChatInput"] {
    background: var(--surface-2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color 0.2s !important;
  }
  [data-testid="stChatInput"]:focus-within {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
  }
  [data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-1) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
  }
  [data-testid="stChatInput"] textarea::placeholder { color: var(--text-3) !important; }

  /* ── Status widget ───────────────────────────────────────── */
  [data-testid="stStatus"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-1) !important;
  }
  [data-testid="stStatus"] p { color: var(--text-2) !important; }

  /* ── Alert / info boxes ──────────────────────────────────── */
  [data-testid="stAlert"] {
    background: var(--accent-blue-dim) !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-1) !important;
  }
  [data-testid="stAlert"][data-baseweb="notification"][kind="error"] {
    background: var(--accent-red-dim) !important;
    border-color: rgba(239,68,68,0.3) !important;
  }
  [data-testid="stException"] {
    background: var(--accent-red-dim) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: var(--radius-sm) !important;
  }

  /* ── Dividers ────────────────────────────────────────────── */
  hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 24px 0 !important;
  }

  /* ── Column spacing ──────────────────────────────────────── */
  [data-testid="column"] {
    padding: 0 10px !important;
    overflow: visible !important;
  }
  [data-testid="column"]:first-child { padding-left: 0 !important; }
  [data-testid="column"]:last-child  { padding-right: 0 !important; }
  [data-testid="stHorizontalBlock"]  { overflow: visible !important; }
  [data-testid="stVerticalBlockBorderWrapper"] { overflow: visible !important; }

  /* ── Media ───────────────────────────────────────────────── */
  video {
    border-radius: var(--radius-md) !important;
    background: #000 !important;
    width: 100% !important;
  }
  [data-testid="stImage"] img {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
  }

  /* ── Spinner ─────────────────────────────────────────────── */
  [data-testid="stSpinner"] { color: var(--accent-blue) !important; }

  /* ── Caption ─────────────────────────────────────────────── */
  [data-testid="stCaptionContainer"] p {
    color: var(--text-3) !important;
    font-size: 11px !important;
    line-height: 1.5 !important;
  }

  /* ── Metric cards ────────────────────────────────────────── */
  [data-testid="stMetric"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 14px 16px !important;
  }
  [data-testid="stMetricLabel"] p { color: var(--text-3) !important; font-size: 11px !important; }
  [data-testid="stMetricValue"] {
    color: var(--text-1) !important;
    font-weight: 700 !important;
    font-size: 22px !important;
  }

  /* ── Focus ring ──────────────────────────────────────────── */
  :focus-visible {
    outline: 2px solid var(--accent-blue) !important;
    outline-offset: 2px !important;
  }
</style>
"""


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _card_open(extra_style: str = "") -> None:
    """Open a glassmorphism card div."""
    st.markdown(
        f"<div style='"
        f"background:#0d1526;"
        f"border:1px solid #1e2d45;"
        f"border-top:1px solid rgba(255,255,255,0.06);"
        f"border-radius:14px;"
        f"padding:24px 26px 26px 26px;"
        f"box-shadow:0 4px 40px rgba(0,0,0,0.55),0 1px 0 rgba(255,255,255,0.03) inset;"
        f"overflow:visible;"
        f"{extra_style}"
        f"'>",
        unsafe_allow_html=True,
    )


def _card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _section_label(text: str, color: str = "#3a5070") -> None:
    st.markdown(
        f"<p style='"
        f"font-size:10px;"
        f"font-weight:700;"
        f"letter-spacing:0.12em;"
        f"color:{color};"
        f"text-transform:uppercase;"
        f"margin:0 0 16px 0;"
        f"font-family:Inter,sans-serif;"
        f"'>{text}</p>",
        unsafe_allow_html=True,
    )


def _spacer(px: int = 20) -> None:
    st.markdown(f"<div style='height:{px}px;'></div>", unsafe_allow_html=True)


def _render_report(report) -> None:
    """Styled incident report — Aegis design system."""
    from core.video_processor import format_timestamp_hms as _hms

    confidence_color = {
        "high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"
    }.get(report.confidence.lower(), "#94a3b8")
    confidence_bg = {
        "high": "rgba(34,197,94,0.10)", "medium": "rgba(245,158,11,0.10)",
        "low": "rgba(239,68,68,0.10)"
    }.get(report.confidence.lower(), "rgba(148,163,184,0.08)")

    kind_label = report.anomaly_kind.upper() if report.anomaly_kind else "ANOMALY"
    ts_hms = _hms(report.timestamp_seconds)

    # ── Badge + timestamp row ──────────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex;align-items:center;justify-content:space-between;"
        f"margin-bottom:20px;'>"
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<span style='display:inline-flex;align-items:center;gap:5px;"
        f"background:rgba(239,68,68,0.12);color:#ef4444;"
        f"border:1px solid rgba(239,68,68,0.3);border-radius:999px;"
        f"padding:4px 12px;font-size:10px;font-weight:700;letter-spacing:0.12em;"
        f"font-family:Inter,sans-serif;'>"
        f"<span style='width:5px;height:5px;background:#ef4444;border-radius:50%;"
        f"display:inline-block;animation:pulse 2s infinite;'></span>"
        f"{kind_label}</span>"
        f"</div>"
        f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:12px;"
        f"color:#3a5070;letter-spacing:0.05em;'>T+ {ts_hms}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Key metrics row ────────────────────────────────────────────────────
    metrics = [
        ("Confidence",  report.confidence.capitalize(), confidence_color, confidence_bg),
        ("Fault",       (report.at_fault or "unclear")[:18], "#e2e8f0",    "#0a1525"),
        ("Report ID",   report.incident_id[:8] + "…",        "#3b82f6",    "rgba(59,130,246,0.08)"),
    ]
    cols = st.columns(3, gap="small")
    for col, (label, value, color, bg) in zip(cols, metrics):
        col.markdown(
            f"<div style='background:{bg};border:1px solid #1a2c42;"
            f"border-radius:10px;padding:12px 14px;'>"
            f"<div style='font-size:9px;font-weight:700;letter-spacing:0.12em;"
            f"color:#3a5070;text-transform:uppercase;margin-bottom:5px;"
            f"font-family:Inter,sans-serif;'>{label}</div>"
            f"<div style='font-size:14px;font-weight:600;color:{color};"
            f"font-family:Inter,sans-serif;line-height:1.2;'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _spacer(16)

    # ── Scene description ──────────────────────────────────────────────────
    if report.scene_description:
        st.markdown(
            f"<div style='background:#080e1c;border:1px solid #1a2c42;"
            f"border-radius:10px;padding:14px 16px;margin-bottom:12px;'>"
            f"<div style='font-size:9px;font-weight:700;letter-spacing:0.12em;"
            f"color:#3a5070;text-transform:uppercase;margin-bottom:8px;"
            f"font-family:Inter,sans-serif;'>Scene Description</div>"
            f"<div style='font-size:13px;color:#94a3b8;line-height:1.65;"
            f"font-family:Inter,sans-serif;'>{report.scene_description}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Sequence of events ─────────────────────────────────────────────────
    if report.sequence_of_events:
        items_html = "".join(
            f"<div style='display:flex;gap:12px;align-items:flex-start;"
            f"padding:7px 0;border-bottom:1px solid #0f1e30;'>"
            f"<span style='width:6px;height:6px;border-radius:50%;"
            f"background:#3b82f6;flex-shrink:0;margin-top:5px;'></span>"
            f"<span style='font-size:12px;color:#94a3b8;line-height:1.55;"
            f"font-family:Inter,sans-serif;'>{e}</span>"
            f"</div>"
            for e in report.sequence_of_events
        )
        st.markdown(
            f"<div style='background:#080e1c;border:1px solid #1a2c42;"
            f"border-radius:10px;padding:14px 16px;margin-bottom:12px;'>"
            f"<div style='font-size:9px;font-weight:700;letter-spacing:0.12em;"
            f"color:#3a5070;text-transform:uppercase;margin-bottom:8px;"
            f"font-family:Inter,sans-serif;'>Sequence of Events</div>"
            f"{items_html}</div>",
            unsafe_allow_html=True,
        )

    # ── Probable cause ─────────────────────────────────────────────────────
    if report.probable_cause and report.probable_cause.lower() not in {"unclear", ""}:
        st.markdown(
            f"<div style='background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.18);"
            f"border-radius:10px;padding:12px 16px;margin-bottom:12px;'>"
            f"<div style='font-size:9px;font-weight:700;letter-spacing:0.12em;"
            f"color:#f59e0b;text-transform:uppercase;margin-bottom:5px;"
            f"font-family:Inter,sans-serif;'>Probable Cause</div>"
            f"<div style='font-size:13px;color:#d4a017;font-family:Inter,sans-serif;line-height:1.5;'>"
            f"{report.probable_cause}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Violations ─────────────────────────────────────────────────────────
    if report.violations_observed:
        violations_html = "".join(
            f"<span style='display:inline-block;background:rgba(239,68,68,0.08);"
            f"border:1px solid rgba(239,68,68,0.2);border-radius:999px;"
            f"padding:3px 10px;font-size:11px;color:#ef4444;"
            f"font-family:Inter,sans-serif;margin:2px;'>{v}</span>"
            for v in report.violations_observed
        )
        st.markdown(
            f"<div style='background:#080e1c;border:1px solid #1a2c42;"
            f"border-radius:10px;padding:12px 16px;margin-bottom:12px;'>"
            f"<div style='font-size:9px;font-weight:700;letter-spacing:0.12em;"
            f"color:#3a5070;text-transform:uppercase;margin-bottom:8px;"
            f"font-family:Inter,sans-serif;'>Violations Observed</div>"
            f"{violations_html}</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title=config.APP_TITLE,
        layout="wide",
        page_icon="🛡️",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_AEGIS_CSS, unsafe_allow_html=True)
    _init_state()

    # ── HEADER ───────────────────────────────────────────────────────────
    st.markdown(
        "<div style='padding:40px 48px 0 48px;'>"

        # Top bar: logo + title + version pill
        "<div style='display:flex;align-items:center;justify-content:space-between;"
        "margin-bottom:10px;'>"

        # Left: icon + titles
        "<div style='display:flex;align-items:center;gap:18px;'>"
        "<div style='"
        "width:46px;height:46px;"
        "background:linear-gradient(145deg,#ef4444 0%,#b91c1c 100%);"
        "border-radius:12px;"
        "display:flex;align-items:center;justify-content:center;"
        "font-size:22px;"
        "box-shadow:0 4px 20px rgba(239,68,68,0.45),0 1px 0 rgba(255,255,255,0.1) inset;"
        "flex-shrink:0;'>🛡️</div>"

        "<div>"
        "<h1 style='"
        "font-size:24px;font-weight:800;letter-spacing:-0.04em;"
        "color:#e2e8f0;margin:0 0 3px 0;line-height:1;font-family:Inter,sans-serif;"
        "'>Forensic AI</h1>"
        "<p style='"
        "font-size:12px;color:#3a5070;margin:0;font-family:Inter,sans-serif;"
        "letter-spacing:0.03em;font-weight:500;"
        "'>Traffic Incident Analyst &nbsp;·&nbsp; Computer Vision + VLM</p>"
        "</div>"
        "</div>"

        # Right: system status pill
        "<div style='"
        "display:flex;align-items:center;gap:6px;"
        "background:#0a1525;border:1px solid #1a2c42;border-radius:999px;"
        "padding:6px 14px;'>"
        "<span style='width:6px;height:6px;background:#22c55e;border-radius:50%;"
        "display:inline-block;'></span>"
        "<span style='font-size:11px;font-weight:600;color:#3a5070;"
        "font-family:Inter,sans-serif;letter-spacing:0.05em;'>SYSTEM READY</span>"
        "</div>"

        "</div>"  # end top bar

        # Accent rule
        "<div style='"
        "height:1px;"
        "background:linear-gradient(90deg,#ef4444 0%,rgba(239,68,68,0.15) 40%,transparent 70%);"
        "margin-top:18px;"
        "'></div>"

        "</div>",
        unsafe_allow_html=True,
    )

    # ── CONTROLS ROW ─────────────────────────────────────────────────────
    _spacer(28)
    st.markdown("<div style='padding:0 48px;'>", unsafe_allow_html=True)

    _card_open()
    _section_label("VIDEO INPUT")

    upload_col, demo_col = st.columns([5, 1], gap="medium")
    with upload_col:
        uploaded = st.file_uploader(
            "Upload traffic footage (.mp4 / .avi / .mov)",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed",
        )
    with demo_col:
        use_demo = st.button("Use Demo", use_container_width=True)

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

    _spacer(16)

    run_btn_disabled = st.session_state.uploaded_path is None
    label = "▶   Analyze Video" if not run_btn_disabled else "Upload or select a video to begin"
    if st.button(label, type="primary", disabled=run_btn_disabled, use_container_width=True):
        with st.status("Running forensic analysis pipeline…", expanded=True) as status:
            try:
                run_pipeline(st.session_state.uploaded_path)
                status.update(label="✅  Analysis complete", state="complete")
            except Exception as e:                              # noqa: BLE001
                log.exception("Pipeline failed")
                status.update(label=f"❌  Analysis failed: {e}", state="error")

    _card_close()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── MAIN 3-PANE AREA ─────────────────────────────────────────────────
    _spacer(24)
    st.markdown("<div style='padding:0 48px;'>", unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")

    # ── LEFT: Video + keyframes ───────────────────────────────────────────
    with left:
        _card_open()
        _section_label("VIDEO FEED")
        video_panel.render_player(st.session_state.uploaded_path)
        if st.session_state.keyframe_paths:
            _spacer(20)
            video_panel.render_keyframe_strip(st.session_state.keyframe_paths)
        _card_close()

    # ── RIGHT: Live log + Incident report ────────────────────────────────
    with right:
        # Live log card
        _card_open("margin-bottom:20px;")
        live_log.render()
        _card_close()

        # Incident report card
        _card_open()
        _section_label("INCIDENT REPORT")
        report = st.session_state.report
        if report is None:
            st.markdown(
                "<div style='text-align:center;padding:32px 0;'>"
                "<div style='font-size:36px;margin-bottom:12px;opacity:0.2;'>📋</div>"
                "<div style='font-size:12px;color:#2a3d56;font-family:Inter,sans-serif;"
                "font-weight:500;letter-spacing:0.05em;'>AWAITING INCIDENT DETECTION</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            _render_report(report)

            # Download buttons
            bundle = st.session_state.bundle_paths or {}
            _spacer(16)
            _section_label("EXPORT BUNDLE")
            dl_cols = st.columns(3, gap="small")
            for col, key, label in [
                (dl_cols[0], "markdown", "📄 Markdown"),
                (dl_cols[1], "legal",    "⚖️ Legal"),
                (dl_cols[2], "json",     "🧾 JSON"),
            ]:
                p = bundle.get(key)
                if p and Path(p).exists():
                    col.download_button(
                        label=label,
                        data=Path(p).read_bytes(),
                        file_name=Path(p).name,
                        use_container_width=True,
                    )
        _card_close()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── CHAT SECTION ─────────────────────────────────────────────────────
    _spacer(24)
    st.markdown("<div style='padding:0 48px 56px 48px;'>", unsafe_allow_html=True)
    _card_open()
    chat_panel.render(st.session_state.chat_engine)
    _card_close()
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
