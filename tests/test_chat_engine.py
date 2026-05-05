"""Tests for the deterministic chat engine."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.chat_engine import ChatEngine, FrameDescription
from core.report_generator import IncidentReport


@pytest.fixture
def report() -> IncidentReport:
    return IncidentReport(
        incident_id="abc12345-67-test",
        video_filename="demo.mp4",
        timestamp_seconds=8.0,
        duration_analyzed_seconds=7.0,
        scene_description="A red SUV approaches an intersection at speed.",
        vehicles_involved=["Red SUV", "Blue cyclist"],
        sequence_of_events=[
            "Red SUV enters the intersection without stopping",
            "Blue cyclist proceeds with the green light",
            "SUV strikes the cyclist",
        ],
        probable_cause="The SUV failed to yield at a marked intersection.",
        violations_observed=["Failure to stop at intersection"],
        at_fault="Red SUV",
        confidence="high",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_used="test-model",
    )


@pytest.fixture
def descriptions():
    return [
        FrameDescription(timestamp=2.0, description=(
            "A red SUV is approaching an intersection from the left at moderate speed."
        )),
        FrameDescription(timestamp=6.0, description=(
            "A blue cyclist is visible in the bike lane on the right."
        )),
        FrameDescription(timestamp=8.0, description=(
            "The red SUV enters the intersection while the cyclist is mid-crossing."
        )),
        FrameDescription(timestamp=10.0, description=(
            "Vehicles are stopped after the collision."
        )),
    ]


def test_fault_intent(report, descriptions):
    eng = ChatEngine(report=report, descriptions=descriptions)
    out = eng.answer("Who was at fault?")
    assert "Red SUV" in out
    assert "high" in out.lower()


def test_legal_summary_intent(report, descriptions):
    eng = ChatEngine(report=report, descriptions=descriptions)
    out = eng.answer("Generate a legal summary for the insurance company.")
    assert "FORMAL INCIDENT SUMMARY" in out
    assert "Red SUV" in out
    assert report.incident_id in out


def test_vehicles_intent(report, descriptions):
    eng = ChatEngine(report=report, descriptions=descriptions)
    out = eng.answer("Which vehicles were involved?")
    assert "Red SUV" in out and "Blue cyclist" in out


def test_violations_intent(report, descriptions):
    eng = ChatEngine(report=report, descriptions=descriptions)
    out = eng.answer("What violations were observed?")
    assert "Failure to stop at intersection" in out


def test_summarize_intent(report, descriptions):
    eng = ChatEngine(report=report, descriptions=descriptions)
    out = eng.answer("Summarize what happened.")
    assert "00:08" in out
    assert "Red SUV" in out


def test_time_query(report, descriptions):
    eng = ChatEngine(report=report, descriptions=descriptions)
    out = eng.answer("What happened at 0:08?")
    assert "00:08" in out
    assert "intersection" in out.lower()


def test_retrieval_fallback(report, descriptions):
    eng = ChatEngine(report=report, descriptions=descriptions)
    out = eng.answer("Tell me about the cyclist")
    # Should pick at least one description mentioning the cyclist
    assert "cyclist" in out.lower()


def test_no_report_no_legal():
    eng = ChatEngine()
    out = eng.answer("Generate a legal summary please.")
    assert "no incident" in out.lower()
