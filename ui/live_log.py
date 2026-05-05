"""Streaming live-log component (PRD §5)."""

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
    def emoji(self) -> str:
        return {"info": "🟢", "warn": "🟡", "crit": "🔴", "ok": "✅"}.get(self.level, "·")


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
        return "  --  "
    s = int(ts)
    return f"{s // 60:02d}:{s % 60:02d}"


def render(container=None) -> None:
    """Render the log inside the given container (or current scope)."""
    init_state()
    target = container or st
    target.markdown("##### LIVE LOG")
    if not st.session_state.log_entries:
        target.markdown(
            "<div style='font-family: monospace; opacity: 0.5;'>"
            "(awaiting upload)</div>",
            unsafe_allow_html=True,
        )
        return
    rows = []
    for e in st.session_state.log_entries:
        rows.append(
            f"<div style='font-family: monospace; font-size: 0.85em;'>"
            f"{_fmt_ts(e.timestamp_sec)} {e.emoji} {e.message}"
            f"</div>"
        )
    target.markdown("\n".join(rows), unsafe_allow_html=True)
