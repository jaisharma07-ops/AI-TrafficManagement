"""Tests for core.video_processor.

These tests synthesize a tiny video on the fly so they don't depend on
external assets or model downloads.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.video_processor import (
    extract_window,
    format_timestamp_hms,
    iter_frames,
    probe,
)


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    """A 4-second 30-fps 64x64 clip with a moving white square."""
    out = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    duration_sec = 4
    n_frames = fps * duration_sec
    writer = cv2.VideoWriter(str(out), fourcc, fps, (64, 64))
    assert writer.isOpened(), "VideoWriter failed to open - check codec availability"
    for i in range(n_frames):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        x = (i * 1) % 60
        cv2.rectangle(frame, (x, 20), (x + 4, 24), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return out


def test_probe_returns_correct_metadata(tiny_video: Path):
    meta = probe(tiny_video)
    assert meta.fps == pytest.approx(30, abs=1)
    assert meta.frame_count == 120
    assert meta.width == 64 and meta.height == 64
    assert meta.duration_sec == pytest.approx(4.0, abs=0.1)


def test_iter_frames_respects_sample_fps(tiny_video: Path):
    # Sampling at 5 fps from a 30 fps source -> stride 6 -> 20 frames
    samples = list(iter_frames(tiny_video, sample_fps=5.0))
    assert 18 <= len(samples) <= 22
    # Timestamps should be monotonic
    for a, b in zip(samples, samples[1:]):
        assert b.timestamp >= a.timestamp


def test_extract_window_returns_n_keyframes(tiny_video: Path):
    frames = extract_window(
        tiny_video,
        center_sec=2.0,
        pre_sec=1.0,
        post_sec=1.0,
        n_keyframes=4,
    )
    assert len(frames) == 4
    timestamps = [f.timestamp for f in frames]
    # Should span [1.0, 3.0] inclusive
    assert min(timestamps) >= 0.99
    assert max(timestamps) <= 3.01
    # Strictly increasing
    assert timestamps == sorted(timestamps)


def test_extract_window_clamps_to_video_bounds(tiny_video: Path):
    # Asking for window centered near end should clamp upper bound
    frames = extract_window(
        tiny_video,
        center_sec=3.5,
        pre_sec=2.0,
        post_sec=5.0,  # past end of 4s video
        n_keyframes=3,
    )
    assert len(frames) == 3
    assert max(f.timestamp for f in frames) <= 4.01


def test_format_timestamp_hms():
    assert format_timestamp_hms(0) == "00:00"
    assert format_timestamp_hms(5.6) == "00:06"
    assert format_timestamp_hms(75) == "01:15"
    assert format_timestamp_hms(3661) == "01:01:01"
