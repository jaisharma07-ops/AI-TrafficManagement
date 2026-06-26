"""Live log component — Aegis Forensic Intelligence design system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import streamlit as st


@dataclass
class LogEntry:
    timestamp_sec: float
    level: str         # "info" | "warn" | "crit" | "ok"
    message: str

    @property
    def dot_color(self) -> str:
        return {
            "info": "#3b82f6",
            "warn": "#f59e0b",
            "crit": "#ef4444",
            "ok":   "#22c55e",
        }.get(self.level, "#4a6180")

    @property
    def text_color(self) -> str:
        return {
            "info": "#94a3b8",
            "warn": "#d4a017",
            "crit": "#f87171",
            "ok":   "#4ade80",
        }.get(self.level, "#94a3b8")

    @property
    def bg_color(self) -> str:
        return {
            "info": "transparent",
            "warn": "rgba(245,158,11,0.05)",
            "crit": "rgba(239,68,68,0.06)",
            "ok":   "rgba(34,197,94,0.05)",
        }.get(self.level, "transparent")

    @property
    def border_color(self) -> str:
        return {
            "info": "transparent",
            "warn": "rgba(245,158,11,0.15)",
            "crit": "rgba(239,68,68,0.2)",
            "ok":   "rgba(34,197,94,0.15)",
        }.get(self.level, "transparent")


def init_state() -> None:
    if "log_entries" not in st.session_state:
        st.session_state.log_entries: List[LogEntry] = []


def append(ts: float, level: str, message: str) -> None:
    init_state()
    st.session_state.log_entries.append(LogEntry(ts, level, message))


def clear() -> None:
    st.session_state.log_entries = []


def _fmt_ts(ts: float) -> str:
    if ts < 0:
        return "  ·  "
    s = int(ts)
    return f"{s // 60:02d}:{s % 60:02d}"


def render(container=None) -> None:
    """Render the live activity log."""
    init_state()
    target = container or st

    # Section label
    target.markdown(
        "<div style='display:flex;align-items:center;justify-content:space-between;"
        "margin-bottom:16px;'>"
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.12em;"
        "color:#3a5070;text-transform:uppercase;margin:0;"
        "font-family:Inter,sans-serif;'>Live Activity Log</p>"
        f"<span style='font-size:10px;color:#2a3d56;font-family:\"JetBrains Mono\",monospace;"
        f"'>{len(st.session_state.log_entries)} events</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    entries = st.session_state.log_entries

    if not entries:
        target.markdown(
            "<div style='"
            "display:flex;flex-direction:column;align-items:center;justify-content:center;"
            "padding:28px 0;"
            "'>"
            "<div style='font-size:28px;opacity:0.12;margin-bottom:10px;'>📡</div>"
            "<div style='font-size:11px;color:#2a3d56;font-family:Inter,sans-serif;"
            "font-weight:500;letter-spacing:0.06em;'>AWAITING UPLOAD</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Log container with max-height scroll
    rows_html = ""
    for e in entries:
        ts_str = _fmt_ts(e.timestamp_sec)
        # Escape HTML in message
        safe_msg = (e.message
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))
        rows_html += (
            f"<div style='"
            f"display:flex;align-items:baseline;gap:10px;"
            f"padding:6px 10px;"
            f"border-radius:6px;"
            f"background:{e.bg_color};"
            f"border:1px solid {e.border_color};"
            f"margin-bottom:3px;"
            f"'>"
            # Dot
            f"<span style='"
            f"width:5px;height:5px;border-radius:50%;"
            f"background:{e.dot_color};"
            f"flex-shrink:0;margin-top:4px;"
            f"display:inline-block;"
            f"'></span>"
            # Timestamp
            f"<span style='"
            f"font-family:\"JetBrains Mono\",monospace;"
            f"font-size:10px;color:#2a3d56;"
            f"flex-shrink:0;min-width:34px;"
            f"'>{ts_str}</span>"
            # Message
            f"<span style='"
            f"font-family:\"JetBrains Mono\",monospace;"
            f"font-size:11px;color:{e.text_color};"
            f"line-height:1.5;word-break:break-word;"
            f"'>{safe_msg}</span>"
            f"</div>"
        )

    target.markdown(
        f"<div style='"
        f"max-height:220px;"
        f"overflow-y:auto;"
        f"background:#060c18;"
        f"border:1px solid #0f1e30;"
        f"border-radius:10px;"
        f"padding:10px;"
        f"'>"
        f"{rows_html}"
        f"</div>",
        unsafe_allow_html=True,
    )
