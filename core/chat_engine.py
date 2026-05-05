"""Template-based chat over the description list.

PRD §4.3 Mode A. No LLM dependency - keyword overlap + intent rules
+ deterministic templates against the IncidentReport. This is plenty
to satisfy the §11 success criterion of "answers at least 3 distinct
questions correctly" while remaining cold-start fast on CPU.

Intents handled:
  - "who/what was at fault"             -> report.at_fault + confidence
  - "what color/type of vehicle ..."    -> retrieve top-matching descriptions
  - "what happened at <timestamp>"      -> nearest description
  - "generate legal summary"            -> render_legal_summary()
  - "summarize / what happened"         -> condensed report
  - everything else                     -> top-3 keyword-overlap descriptions
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from core.report_generator import IncidentReport, render_legal_summary
from core.video_processor import format_timestamp_hms


@dataclass
class FrameDescription:
    """A timestamped natural-language description of a clip."""
    timestamp: float
    description: str


# --- Intent detection ---

_FAULT_RX = re.compile(r"\b(at fault|who.*responsib|whose fault|who.*caused)\b", re.I)
_LEGAL_RX = re.compile(r"\b(legal|insurance|police|formal).*\b(summary|report|statement)\b", re.I)
_TIME_RX = re.compile(r"\b(?:at|near|around)\s*(\d{1,2}:\d{2}|\d+(?:\.\d+)?\s*s?)\b", re.I)
_VIOLATION_RX = re.compile(r"\b(violations?|broke|breaking|illegal|unlawful|run.*light|stop.*sign)\b", re.I)
_SUMMARIZE_RX = re.compile(r"\b(summari[sz]e|brief|overview|what happened|describe.*incident)\b", re.I)
_VEHICLES_RX = re.compile(r"\b(vehicles?|cars?|trucks?|cyclist|pedestrian|involved|how many)\b", re.I)


def _parse_time(text: str) -> Optional[float]:
    m = _TIME_RX.search(text)
    if not m:
        return None
    raw = m.group(1)
    if ":" in raw:
        mm, ss = raw.split(":")
        return int(mm) * 60 + float(ss)
    return float(raw.rstrip("s "))


def _word_set(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2}


_STOPWORDS = {
    "the", "and", "was", "were", "are", "what", "who", "did", "you", "there",
    "this", "that", "with", "for", "from", "have", "has", "had", "but", "any",
}


def _overlap_score(query: str, doc: str) -> int:
    q = _word_set(query) - _STOPWORDS
    d = _word_set(doc) - _STOPWORDS
    return len(q & d)


# --- The engine ---

class ChatEngine:
    def __init__(
        self,
        report: Optional[IncidentReport] = None,
        descriptions: Optional[Iterable[FrameDescription]] = None,
    ):
        self.report = report
        self.descriptions: List[FrameDescription] = list(descriptions or [])

    # -- mutation --

    def set_report(self, report: IncidentReport) -> None:
        self.report = report

    def add_description(self, desc: FrameDescription) -> None:
        self.descriptions.append(desc)

    def set_descriptions(self, items: Iterable[FrameDescription]) -> None:
        self.descriptions = list(items)

    # -- retrieval --

    def _top_k(self, query: str, k: int = 3) -> List[FrameDescription]:
        scored: List[Tuple[int, FrameDescription]] = [
            (_overlap_score(query, d.description), d) for d in self.descriptions
        ]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda x: (-x[0], x[1].timestamp))
        return [d for _, d in scored[:k]]

    def _nearest_in_time(self, t: float) -> Optional[FrameDescription]:
        if not self.descriptions:
            return None
        return min(self.descriptions, key=lambda d: abs(d.timestamp - t))

    # -- intents --

    def _ans_fault(self) -> str:
        if not self.report:
            return "No incident report has been generated yet."
        if not self.report.at_fault or self.report.at_fault.lower() == "unclear":
            return ("The model could not assign clear fault. "
                    f"Confidence in this is {self.report.confidence}.")
        ts = format_timestamp_hms(self.report.timestamp_seconds)
        return (f"At {ts}, fault is attributed to **{self.report.at_fault}** "
                f"(confidence: {self.report.confidence}). "
                f"Probable cause: {self.report.probable_cause}")

    def _ans_legal(self) -> str:
        if not self.report:
            return "No incident report has been generated yet, so a legal summary cannot be produced."
        return render_legal_summary(self.report)

    def _ans_summary(self) -> str:
        if not self.report:
            return "No incident has been analyzed yet."
        ts = format_timestamp_hms(self.report.timestamp_seconds)
        events = self.report.sequence_of_events
        events_md = "\n".join(f"  - {e}" for e in events) or "  - (no discrete events recorded)"
        return (
            f"At {ts}, the system flagged an incident.\n\n"
            f"**Scene:** {self.report.scene_description}\n\n"
            f"**Sequence of events:**\n{events_md}\n\n"
            f"**Probable cause:** {self.report.probable_cause}\n"
            f"**Fault:** {self.report.at_fault} (confidence {self.report.confidence})"
        )

    def _ans_vehicles(self) -> str:
        if not self.report:
            return "No incident report has been generated yet."
        if not self.report.vehicles_involved:
            return "The model did not identify specific vehicles involved."
        listed = ", ".join(self.report.vehicles_involved)
        return f"Vehicles involved in the incident: {listed}."

    def _ans_violations(self) -> str:
        if not self.report:
            return "No incident report has been generated yet."
        if not self.report.violations_observed:
            return "No specific violations were recorded for this incident."
        items = "\n".join(f"- {v}" for v in self.report.violations_observed)
        return f"Violations observed:\n{items}"

    def _ans_at_time(self, t: float) -> str:
        d = self._nearest_in_time(t)
        if d is None:
            return f"No description was indexed near {format_timestamp_hms(t)}."
        return (f"Around {format_timestamp_hms(d.timestamp)} "
                f"(t = {d.timestamp:.1f}s): {d.description}")

    def _ans_retrieval(self, query: str) -> str:
        hits = self._top_k(query, k=3)
        if not hits:
            return ("I couldn't find any indexed frame descriptions matching that question. "
                    "Try asking about fault, vehicles, violations, or a specific timestamp.")
        lines = [
            f"- `{format_timestamp_hms(h.timestamp)}` (t = {h.timestamp:.1f}s): {h.description}"
            for h in hits
        ]
        return "Most relevant moments:\n" + "\n".join(lines)

    # -- public entry --

    def answer(self, query: str) -> str:
        q = (query or "").strip()
        if not q:
            return "Please ask a question about the incident."

        if _LEGAL_RX.search(q):
            return self._ans_legal()
        if _FAULT_RX.search(q):
            return self._ans_fault()
        if _VIOLATION_RX.search(q):
            return self._ans_violations()
        if _VEHICLES_RX.search(q):
            return self._ans_vehicles()
        t = _parse_time(q)
        if t is not None:
            return self._ans_at_time(t)
        if _SUMMARIZE_RX.search(q):
            return self._ans_summary()
        return self._ans_retrieval(q)
