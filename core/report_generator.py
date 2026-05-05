"""Build the structured incident report.

Produces three artifacts from a forensic VLM dict:
  1. JSON record (PRD §6.1 schema).
  2. Markdown human-readable report.
  3. Formal legal/insurance summary text.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from core.video_processor import format_timestamp_hms


_LEGAL_TEMPLATE = (config.PROMPTS_DIR / "legal_summary.txt").read_text(encoding="utf-8")


@dataclass
class IncidentReport:
    """In-memory structured report (PRD §6.1)."""
    incident_id: str
    video_filename: str
    timestamp_seconds: float
    duration_analyzed_seconds: float
    scene_description: str
    vehicles_involved: List[str]
    sequence_of_events: List[str]
    probable_cause: str
    violations_observed: List[str]
    at_fault: str
    confidence: str
    generated_at: str
    model_used: str
    anomaly_kind: str = ""
    anomaly_detail: str = ""
    keyframe_paths: List[str] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "video_filename": self.video_filename,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "duration_analyzed_seconds": round(self.duration_analyzed_seconds, 3),
            "scene_description": self.scene_description,
            "vehicles_involved": list(self.vehicles_involved),
            "sequence_of_events": list(self.sequence_of_events),
            "probable_cause": self.probable_cause,
            "violations_observed": list(self.violations_observed),
            "at_fault": self.at_fault,
            "confidence": self.confidence,
            "generated_at": self.generated_at,
            "model_used": self.model_used,
            "anomaly_kind": self.anomaly_kind,
            "anomaly_detail": self.anomaly_detail,
            "keyframe_paths": list(self.keyframe_paths),
        }


def build(
    *,
    video_filename: str,
    timestamp_seconds: float,
    duration_analyzed_seconds: float,
    forensic: Dict[str, Any],
    model_used: str,
    anomaly_kind: str = "",
    anomaly_detail: str = "",
    keyframe_paths: Optional[List[Path]] = None,
) -> IncidentReport:
    """Assemble an IncidentReport from VLM output + metadata."""
    return IncidentReport(
        incident_id=str(uuid.uuid4()),
        video_filename=video_filename,
        timestamp_seconds=timestamp_seconds,
        duration_analyzed_seconds=duration_analyzed_seconds,
        scene_description=str(forensic.get("scene_description", "")),
        vehicles_involved=list(forensic.get("vehicles_involved", []) or []),
        sequence_of_events=list(forensic.get("sequence_of_events", []) or []),
        probable_cause=str(forensic.get("probable_cause", "")),
        violations_observed=list(forensic.get("violations_observed", []) or []),
        at_fault=str(forensic.get("at_fault", "unclear")),
        confidence=str(forensic.get("confidence", "low")),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_used=model_used,
        anomaly_kind=anomaly_kind,
        anomaly_detail=anomaly_detail,
        keyframe_paths=[str(p) for p in (keyframe_paths or [])],
    )


# --- Markdown ---

def render_markdown(r: IncidentReport) -> str:
    def _bullet_list(items: List[str]) -> str:
        if not items:
            return "_(none)_"
        return "\n".join(f"- {x}" for x in items)

    return (
        f"# Incident Report\n\n"
        f"**Incident ID:** `{r.incident_id}`  \n"
        f"**Source video:** `{r.video_filename}`  \n"
        f"**Time of event:** `{format_timestamp_hms(r.timestamp_seconds)}` "
        f"(t = {r.timestamp_seconds:.2f}s)  \n"
        f"**Window analyzed:** {r.duration_analyzed_seconds:.1f}s  \n"
        f"**Trigger:** {r.anomaly_kind or '_unspecified_'} - {r.anomaly_detail}  \n"
        f"**Generated:** {r.generated_at}  \n"
        f"**Model:** `{r.model_used}`  \n\n"
        f"## Scene Description\n{r.scene_description}\n\n"
        f"## Vehicles Involved\n{_bullet_list(r.vehicles_involved)}\n\n"
        f"## Sequence of Events\n{_bullet_list(r.sequence_of_events)}\n\n"
        f"## Probable Cause\n{r.probable_cause}\n\n"
        f"## Violations Observed\n{_bullet_list(r.violations_observed)}\n\n"
        f"## Determination of Fault\n"
        f"**{r.at_fault}** (confidence: **{r.confidence}**)\n"
    )


# --- Legal summary ---

def render_legal_summary(r: IncidentReport) -> str:
    vehicles_block = "\n".join(f"  - {v}" for v in r.vehicles_involved) or "  - (not identified)"
    events_block = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(r.sequence_of_events)) \
                   or "  (no discrete events recorded)"
    violations_block = "\n".join(f"  - {v}" for v in r.violations_observed) \
                       or "  - None observed."
    return _LEGAL_TEMPLATE.format(
        incident_id=r.incident_id,
        video_filename=r.video_filename,
        timestamp_hms=format_timestamp_hms(r.timestamp_seconds),
        timestamp_seconds=r.timestamp_seconds,
        duration_analyzed_seconds=r.duration_analyzed_seconds,
        generated_at=r.generated_at,
        model_used=r.model_used,
        scene_description=r.scene_description,
        vehicles_block=vehicles_block,
        events_block=events_block,
        probable_cause=r.probable_cause,
        violations_block=violations_block,
        at_fault=r.at_fault,
        confidence=r.confidence,
    )


# --- IO ---

def save_bundle(r: IncidentReport, out_dir: Path) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"incident_{r.incident_id[:8]}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    legal_path = base.with_name(base.name + "_legal.txt")
    json_path.write_text(json.dumps(r.to_json_dict(), indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(r), encoding="utf-8")
    legal_path.write_text(render_legal_summary(r), encoding="utf-8")
    return {"json": json_path, "markdown": md_path, "legal": legal_path}
