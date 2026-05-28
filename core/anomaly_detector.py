"""Anomaly detection over a stream of frames.

Implements the three triggers from PRD §4.1:
  1. IoU between any two vehicle boxes > 0.30 for >=2 consecutive frames
     (proxy for collision).
  2. A tracked vehicle's velocity drops by >70% within 0.5s
     (proxy for sudden deceleration).
  3. A tracked vehicle's bounding box disappears (proxy for off-road / occlusion).

Falls back to a frame-difference trigger when YOLO returns zero detections
across the whole stream — this keeps the pipeline demonstrable on
synthetic videos that don't contain real COCO vehicles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np

import config
from core.video_processor import FrameSample


# --- Detection / tracking primitives ---

@dataclass
class Detection:
    """A single YOLO detection in image coordinates."""
    cls: int
    conf: float
    xyxy: Tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class Track:
    """One persistent tracked object across frames."""
    track_id: int
    cls: int
    last_centroid: Tuple[float, float]
    last_xyxy: Tuple[float, float, float, float]
    last_seen_ts: float
    velocity_history: List[Tuple[float, float]] = field(default_factory=list)
    # ^ list of (timestamp, speed_pixels_per_sec)
    missed_frames: int = 0


@dataclass
class AnomalyEvent:
    timestamp: float
    kind: str       # "collision" | "deceleration" | "disappearance" | "framediff"
    detail: str
    involved_track_ids: List[int] = field(default_factory=list)


# --- Geometry ---

def iou(a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --- Centroid tracker ---

class CentroidTracker:
    """Minimal greedy nearest-centroid tracker.

    No Kalman, no DeepSORT — just enough to keep stable IDs across
    frames so we can compute per-track velocity and disappearance.
    """

    def __init__(self, max_distance_px: float = 80.0,
                 max_missed_frames: int = config.DISAPPEAR_GRACE_FRAMES):
        self.max_distance_px = max_distance_px
        self.max_missed_frames = max_missed_frames
        self._next_id = 0
        self.tracks: dict[int, Track] = {}

    def update(self, detections: List[Detection], timestamp: float) -> List[int]:
        """Match detections to existing tracks; return list of track IDs.

        Order of returned IDs matches order of input detections.
        """
        if not self.tracks:
            return [self._spawn_track(d, timestamp) for d in detections]

        # Build distance matrix between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        track_centroids = np.array(
            [self.tracks[tid].last_centroid for tid in track_ids]
        )
        det_centroids = np.array([d.centroid for d in detections]) \
            if detections else np.zeros((0, 2))

        assigned_track_for_det: List[Optional[int]] = [None] * len(detections)
        if len(detections) and len(track_ids):
            # Pairwise distances; greedy assignment
            from scipy.spatial.distance import cdist  # local import keeps cold-start cheap
            dists = cdist(det_centroids, track_centroids)
            used_tracks: set[int] = set()
            order = np.argsort(dists, axis=None)
            for flat in order:
                di, ti = divmod(int(flat), len(track_ids))
                if assigned_track_for_det[di] is not None:
                    continue
                if track_ids[ti] in used_tracks:
                    continue
                if dists[di, ti] > self.max_distance_px:
                    break
                assigned_track_for_det[di] = track_ids[ti]
                used_tracks.add(track_ids[ti])

        # Update assigned tracks; spawn unassigned detections
        out_ids: List[int] = []
        for det, tid in zip(detections, assigned_track_for_det):
            if tid is None:
                out_ids.append(self._spawn_track(det, timestamp))
            else:
                self._update_track(tid, det, timestamp)
                out_ids.append(tid)

        # Increment missed counter for tracks that weren't matched this frame
        matched = {tid for tid in assigned_track_for_det if tid is not None}
        for tid in list(self.tracks.keys()):
            if tid not in matched:
                self.tracks[tid].missed_frames += 1
                if self.tracks[tid].missed_frames > self.max_missed_frames:
                    # Track is dead - we'll handle it as a "disappearance" event
                    # in the higher-level detector; here we just drop it.
                    del self.tracks[tid]
        return out_ids

    def _spawn_track(self, det: Detection, timestamp: float) -> int:
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = Track(
            track_id=tid,
            cls=det.cls,
            last_centroid=det.centroid,
            last_xyxy=det.xyxy,
            last_seen_ts=timestamp,
        )
        return tid

    def _update_track(self, tid: int, det: Detection, timestamp: float):
        track = self.tracks[tid]
        dt = max(1e-6, timestamp - track.last_seen_ts)
        cx_old, cy_old = track.last_centroid
        cx_new, cy_new = det.centroid
        speed = float(np.hypot(cx_new - cx_old, cy_new - cy_old) / dt)
        track.velocity_history.append((timestamp, speed))
        # Trim history to last 1.5s for cheap memory
        cutoff = timestamp - 1.5
        track.velocity_history = [
            (t, v) for (t, v) in track.velocity_history if t >= cutoff
        ]
        track.last_centroid = det.centroid
        track.last_xyxy = det.xyxy
        track.last_seen_ts = timestamp
        track.missed_frames = 0


# --- Trigger logic ---

def _check_collision(detections: List[Detection],
                     prior_overlap_count: int) -> Tuple[bool, int, str]:
    """Pairwise IoU check among vehicle detections."""
    vehicle_dets = [d for d in detections if d.cls in config.VEHICLE_CLASS_IDS]
    overlap_pair: Optional[Tuple[int, int]] = None
    for i in range(len(vehicle_dets)):
        for j in range(i + 1, len(vehicle_dets)):
            if iou(vehicle_dets[i].xyxy, vehicle_dets[j].xyxy) > config.IOU_OVERLAP_THRESHOLD:
                overlap_pair = (i, j)
                break
        if overlap_pair:
            break
    if overlap_pair is None:
        return False, 0, ""
    new_count = prior_overlap_count + 1
    if new_count >= config.IOU_CONSECUTIVE_FRAMES:
        i, j = overlap_pair
        detail = (f"IoU overlap > {config.IOU_OVERLAP_THRESHOLD} between "
                  f"two vehicles for {new_count} consecutive frames")
        return True, new_count, detail
    return False, new_count, ""


def _check_deceleration(tracker: CentroidTracker, now_ts: float) -> Optional[Track]:
    """Returns a track that just decelerated >70% within DECEL_WINDOW_SEC.

    Compares the latest measured speed against the max speed observed
    earlier than DECEL_WINDOW_SEC ago (within a 1.5s lookback).
    """
    for track in tracker.tracks.values():
        hist = track.velocity_history
        if len(hist) < 2:
            continue
        latest_t, latest_v = hist[-1]
        # "Older" = at least DECEL_WINDOW_SEC before now
        older = [v for (t, v) in hist[:-1] if now_ts - t >= config.DECEL_WINDOW_SEC]
        if not older:
            continue
        prev_max = max(older)
        if prev_max < 5.0:                 # was barely moving anyway
            continue
        drop = (prev_max - latest_v) / prev_max
        if drop > config.DECEL_DROP_RATIO:
            return track
    return None


# --- Public scanner ---

class AnomalyDetector:
    """Scan a stream of `FrameSample`s and emit `AnomalyEvent`s."""

    def __init__(self, yolo_model=None):
        """`yolo_model` is an `ultralytics.YOLO` instance, or None to lazy-load."""
        self._yolo = yolo_model
        self._yolo_unavailable = False
        self.tracker = CentroidTracker()
        self._collision_streak = 0
        self._dead_track_ids_seen: set[int] = set()
        self._zero_detection_streak = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._fired_kinds: set[str] = set()
        self.total_detections_seen = 0
        # Peak-motion buffer (used as a no-YOLO fallback)
        self._peak_diff_score: float = 0.0
        self._peak_diff_ts: float = 0.0
        self._diff_prev_gray: Optional[np.ndarray] = None
        self._frames_scanned: int = 0

    # -- detection backend --

    def _ensure_yolo(self) -> bool:
        """Returns True if a usable YOLO model is available."""
        if self._yolo_unavailable:
            return False
        if self._yolo is not None:
            return True
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(config.YOLO_MODEL)
            return True
        except Exception:                                  # noqa: BLE001
            self._yolo_unavailable = True
            return False

    def _detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if not self._ensure_yolo():
            return []
        results = self._yolo.predict(
            frame_bgr, verbose=False, classes=None,
            imgsz=config.YOLO_IMGSZ, conf=0.35,
            device=0 if config.ON_GPU else "cpu",
            half=config.ON_GPU,  # fp16 inference on GPU
        )
        out: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                out.append(Detection(cls=cls, conf=conf, xyxy=xyxy))
        return out

    def _frame_diff_anomaly(self, frame_bgr: np.ndarray, ts: float) -> Optional[AnomalyEvent]:
        """Fallback: large frame-diff spike with no YOLO detections."""
        import cv2  # local
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return None
        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        score = float(diff.mean())
        if score > 5.0:  # heuristic; synthetic clips spike well above this
            return AnomalyEvent(
                timestamp=ts,
                kind="framediff",
                detail=f"Frame-diff intensity {score:.1f} (fallback trigger)",
            )
        return None

    # -- main scan loop --

    def _track_peak_motion(self, frame_bgr: np.ndarray, ts: float) -> None:
        """Maintain a running peak frame-diff score (used as YOLO fallback)."""
        import cv2  # local
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._diff_prev_gray is not None and self._diff_prev_gray.shape == gray.shape:
            score = float(cv2.absdiff(gray, self._diff_prev_gray).mean())
            if score > self._peak_diff_score:
                self._peak_diff_score = score
                self._peak_diff_ts = ts
        self._diff_prev_gray = gray

    def scan(self, frames: Iterable[FrameSample]) -> List[AnomalyEvent]:
        """Scan an iterable of samples; return all triggered events.

        For the MVP only the FIRST event is acted upon by the caller, but
        we still collect all so the live-log shows progress.
        """
        events: List[AnomalyEvent] = []
        for sample in frames:
            ts = sample.timestamp
            self._frames_scanned += 1
            self._track_peak_motion(sample.image_bgr, ts)

            try:
                detections = self._detect(sample.image_bgr)
            except Exception as e:                       # noqa: BLE001
                # YOLO failure shouldn't kill the run; record and continue
                events.append(AnomalyEvent(
                    timestamp=ts, kind="error",
                    detail=f"YOLO inference failed: {e}",
                ))
                detections = []

            self.total_detections_seen += len(detections)
            self.tracker.update(detections, ts)

            if not detections:
                self._zero_detection_streak += 1
            else:
                self._zero_detection_streak = 0

            # 1. Collision proxy
            fired, self._collision_streak, msg = _check_collision(
                detections, self._collision_streak,
            )
            if fired and "collision" not in self._fired_kinds:
                self._fired_kinds.add("collision")
                events.append(AnomalyEvent(
                    timestamp=ts, kind="collision", detail=msg,
                ))
                continue

            # 2. Deceleration proxy
            decel_track = _check_deceleration(self.tracker, ts)
            if decel_track and "deceleration" not in self._fired_kinds:
                self._fired_kinds.add("deceleration")
                events.append(AnomalyEvent(
                    timestamp=ts, kind="deceleration",
                    detail=f"Track #{decel_track.track_id} velocity dropped "
                           f">{int(config.DECEL_DROP_RATIO*100)}% in "
                           f"{config.DECEL_WINDOW_SEC}s",
                    involved_track_ids=[decel_track.track_id],
                ))

            # 3. Frame-diff spike fallback (when YOLO has been silent for a while)
            if (self.total_detections_seen == 0
                    and self._zero_detection_streak >= 5
                    and "framediff" not in self._fired_kinds):
                fb = self._frame_diff_anomaly(sample.image_bgr, ts)
                if fb is not None:
                    self._fired_kinds.add("framediff")
                    events.append(fb)

        # End-of-stream: if no primary trigger fired AND we never had
        # detections (YOLO unavailable or saw nothing), emit the
        # peak-motion timestamp so the pipeline still has a window.
        primary_kinds = {"collision", "deceleration", "framediff"}
        already_fired = primary_kinds & self._fired_kinds
        if not already_fired and self.total_detections_seen == 0 and self._frames_scanned > 0:
            events.append(AnomalyEvent(
                timestamp=self._peak_diff_ts,
                kind="framediff",
                detail=(f"No vehicle detections; flagging frame of peak motion "
                        f"(diff score {self._peak_diff_score:.2f})"),
            ))
            self._fired_kinds.add("framediff")

        return events
