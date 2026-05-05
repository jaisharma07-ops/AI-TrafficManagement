"""Unit tests for anomaly logic.

YOLO is bypassed by injecting a stub model — these tests exercise the
tracker, IoU geometry, and trigger thresholds purely.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from core.anomaly_detector import (
    AnomalyDetector,
    CentroidTracker,
    Detection,
    iou,
    _check_collision,
)
from core.video_processor import FrameSample


def _box(cx, cy, w=20, h=20, cls=2, conf=0.9):
    return Detection(
        cls=cls, conf=conf,
        xyxy=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
    )


# --- IoU ---

def test_iou_disjoint():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_identical():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_partial():
    val = iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.1 < val < 0.2  # 25 / 175


# --- Tracker ---

def test_tracker_keeps_id_for_moving_box():
    tracker = CentroidTracker()
    ids_t0 = tracker.update([_box(50, 50)], timestamp=0.0)
    ids_t1 = tracker.update([_box(54, 50)], timestamp=0.1)
    ids_t2 = tracker.update([_box(60, 51)], timestamp=0.2)
    assert ids_t0 == ids_t1 == ids_t2


def test_tracker_assigns_new_id_when_far():
    tracker = CentroidTracker(max_distance_px=20)
    ids_t0 = tracker.update([_box(50, 50)], timestamp=0.0)
    # Jump way outside the matching radius
    ids_t1 = tracker.update([_box(500, 500)], timestamp=0.1)
    assert ids_t0[0] != ids_t1[0]


def test_tracker_drops_track_after_grace():
    tracker = CentroidTracker(max_missed_frames=2)
    tracker.update([_box(50, 50)], timestamp=0.0)
    tracker.update([], timestamp=0.1)
    tracker.update([], timestamp=0.2)
    tracker.update([], timestamp=0.3)
    assert len(tracker.tracks) == 0


# --- Collision streak ---

def test_check_collision_requires_consecutive_frames():
    overlapping = [_box(50, 50, w=40, h=40), _box(60, 50, w=40, h=40)]
    fired, streak, _ = _check_collision(overlapping, prior_overlap_count=0)
    assert not fired and streak == 1
    fired, streak, _ = _check_collision(overlapping, prior_overlap_count=1)
    assert fired and streak == 2


def test_check_collision_no_vehicles_no_fire():
    # Class 0 = person; should be ignored by collision check
    persons = [_box(50, 50, cls=0), _box(55, 50, cls=0)]
    fired, streak, _ = _check_collision(persons, prior_overlap_count=5)
    assert not fired
    assert streak == 0


# --- End-to-end with stubbed YOLO ---

class _StubYolo:
    """Returns scripted detections per call to .predict."""

    def __init__(self, frame_dets: List[List[Detection]]):
        self._frame_dets = frame_dets
        self._idx = 0

    def predict(self, *_args, **_kwargs):
        i = min(self._idx, len(self._frame_dets) - 1)
        self._idx += 1
        dets = self._frame_dets[i]

        class _Box:
            def __init__(self, d: Detection):
                self.cls = np.array([d.cls])
                self.conf = np.array([d.conf])
                self.xyxy = np.array([list(d.xyxy)])

        class _Result:
            def __init__(self, ds: List[Detection]):
                self.boxes = [_Box(d) for d in ds]

        return [_Result(dets)]


def _blank(idx: int, ts: float) -> FrameSample:
    return FrameSample(
        timestamp=ts, frame_index=idx,
        image_bgr=np.zeros((64, 64, 3), dtype=np.uint8),
    )


def test_detector_fires_collision_event():
    # Two vehicles overlapping for 3 consecutive frames -> trigger on frame 2
    overlap = [_box(50, 50, w=40, h=40), _box(60, 50, w=40, h=40)]
    script = [overlap, overlap, overlap]
    det = AnomalyDetector(yolo_model=_StubYolo(script))
    events = det.scan([_blank(i, ts=i * 0.2) for i in range(3)])
    kinds = [e.kind for e in events]
    assert "collision" in kinds
    coll = next(e for e in events if e.kind == "collision")
    assert coll.timestamp == pytest.approx(0.2)


def test_detector_fires_deceleration_event():
    # Fast-moving vehicle that suddenly stops
    script = [
        [_box(10, 50)],
        [_box(40, 50)],   # high speed: ~150 px/s at 0.2s spacing
        [_box(70, 50)],
        [_box(72, 50)],   # decelerated to ~10 px/s
        [_box(73, 50)],
    ]
    det = AnomalyDetector(yolo_model=_StubYolo(script))
    events = det.scan([_blank(i, ts=i * 0.2) for i in range(5)])
    kinds = [e.kind for e in events]
    assert "deceleration" in kinds


def test_detector_falls_back_to_framediff_on_no_detections():
    # YOLO sees nothing - feed frames with a moving white square
    script = [[]] * 10
    det = AnomalyDetector(yolo_model=_StubYolo(script))

    frames: List[FrameSample] = []
    for i in range(10):
        f = np.zeros((64, 64, 3), dtype=np.uint8)
        # Big bright square that moves a lot frame-to-frame -> high diff
        x = (i * 20) % 60
        f[20:60, x:x + 4] = 255
        frames.append(FrameSample(timestamp=i * 0.2, frame_index=i, image_bgr=f))

    events = det.scan(frames)
    kinds = [e.kind for e in events]
    assert "framediff" in kinds
