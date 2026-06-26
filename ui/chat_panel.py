"""Chat panel — Aegis Forensic Intelligence design system."""

from __future__ import annotations

from typing import List

import streamlit as st

from core.chat_engine import ChatEngine


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[dict] = []


def render(engine: ChatEngine) -> None:
    init_state()

    # ── Section header ────────────────────────────────────────────────────
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:space-between;"
        "margin-bottom:20px;'>"

        "<div style='display:flex;align-items:center;gap:10px;'>"
        "<div style='width:30px;height:30px;"
        "background:linear-gradient(135deg,#3b82f6,#1d4ed8);"
        "border-radius:8px;"
        "display:flex;align-items:center;justify-content:center;"
        "font-size:14px;'>"
        "💬</div>"
        "<div>"
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.12em;"
        "color:#3a5070;text-transform:uppercase;margin:0 0 2px 0;"
        "font-family:Inter,sans-serif;'>Chat with Video</p>"
        "<p style='font-size:11px;color:#2a3d56;margin:0;"
        "font-family:Inter,sans-serif;'>"
        "Ask questions about the incident · grounded in VLM analysis"
        "</p>"
        "</div>"
        "</div>"

        # Message count badge
        f"<span style='"
        f"background:#0a1525;border:1px solid #1a2c42;border-radius:999px;"
        f"padding:4px 12px;font-size:10px;font-weight:600;"
        f"color:#2a3d56;font-family:Inter,sans-serif;letter-spacing:0.05em;"
        f"'>{len(st.session_state.chat_history)} messages</span>"

        "</div>",
        unsafe_allow_html=True,
    )

    # ── Message history ───────────────────────────────────────────────────
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    # ── Gate: no report yet ───────────────────────────────────────────────
    if engine.report is None:
        st.markdown(
            "<div style='text-align:center;padding:24px 0;"
            "border:1px dashed #1a2c42;border-radius:10px;margin-top:8px;'>"
            "<div style='font-size:24px;opacity:0.15;margin-bottom:8px;'>🔒</div>"
            "<div style='font-size:12px;color:#2a3d56;font-family:Inter,sans-serif;"
            "font-weight:500;letter-spacing:0.04em;'>"
            "Run the analysis pipeline first to enable chat"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Quick-action pills ────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:10px;font-weight:600;letter-spacing:0.1em;"
        "color:#3a5070;text-transform:uppercase;margin:0 0 10px 0;"
        "font-family:Inter,sans-serif;'>Quick Actions</p>",
        unsafe_allow_html=True,
    )
    q_cols = st.columns(4, gap="small")
    quick_actions = [
        ("Who was at fault?",           "Who was at fault?"),
        ("Legal summary",               "Generate a legal summary for the insurance company."),
        ("Summarize incident",          "Summarize what happened."),
        ("Violations observed?",        "What violations were observed?"),
    ]
    triggered_query: str | None = None
    for col, (label, q) in zip(q_cols, quick_actions):
        if col.button(label, use_container_width=True, key=f"qa_{label}"):
            triggered_query = q

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about the incident, a timestamp, fault, or violations…")
    query = triggered_query or user_input

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        answer = engine.answer(query)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
