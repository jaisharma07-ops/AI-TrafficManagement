"""Streamlit chat panel."""

from __future__ import annotations

from typing import List

import streamlit as st

from core.chat_engine import ChatEngine


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[dict] = []


def render(engine: ChatEngine) -> None:
    init_state()
    st.markdown("### CHAT WITH VIDEO")

    # History
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    if engine.report is None:
        st.caption("Run the analysis pipeline first to enable chat.")
        return

    # Quick-actions
    cols = st.columns(3)
    quick = [
        ("Who was at fault?", "Who was at fault?"),
        ("Generate legal summary", "Generate a legal summary for the insurance company."),
        ("Summarize", "Summarize what happened."),
    ]
    triggered_query = None
    for col, (label, q) in zip(cols, quick):
        if col.button(label, use_container_width=True):
            triggered_query = q

    user_input = st.chat_input("Type your question...")
    query = triggered_query or user_input
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        answer = engine.answer(query)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
