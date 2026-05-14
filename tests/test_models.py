"""Tests for StageResult and core domain models."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_analyzer.types.models import (
    AnalysisSummary,
    OutputFormat,
    StageResult,
    StageStatus,
    TimelineEntry,
    VideoMetadata,
)


class TestStageResult:
    def test_success_sets_ok_true_and_data(self):
        result = StageResult.success(stage="test", data=[1, 2, 3], duration_ms=50.0)
        assert result.ok is True
        assert result.data == [1, 2, 3]
        assert result.status == StageStatus.SUCCESS
        assert result.error is None
        assert result.duration_ms == 50.0

    def test_fail_sets_ok_false_and_error(self):
        result = StageResult.fail(stage="test", error="something broke", duration_ms=10.0)
        assert result.ok is False
        assert result.status == StageStatus.ERROR
        assert result.error == "something broke"
        assert result.data is None

    def test_skipped_sets_status_and_reason(self):
        result = StageResult.skipped(stage="test", reason="audio-only mode")
        assert result.ok is False
        assert result.status == StageStatus.SKIPPED
        assert result.error == "audio-only mode"
        assert result.data is None

    def test_success_with_zero_duration(self):
        result = StageResult.success(stage="test", data="value")
        assert result.duration_ms == 0
        assert result.data == "value"

    def test_fail_with_none_data_is_not_ok(self):
        result = StageResult.fail(stage="test", error="oops")
        assert result.ok is False
        assert result.data is None


class TestVideoMetadata:
    def _make(self, **kwargs) -> VideoMetadata:
        defaults = dict(
            path=Path("/tmp/video.mp4"),
            duration_seconds=120.0,
            width=1920,
            height=1080,
            fps=30.0,
            has_audio=True,
            codec="h264",
            file_size_bytes=1024 * 1024,
        )
        return VideoMetadata(**{**defaults, **kwargs})

    def test_audio_only_file_has_zero_width(self):
        meta = self._make(width=0, height=0, fps=0.0, codec="mp3")
        assert meta.width == 0
        assert meta.height == 0
        assert meta.has_audio is True

    def test_normal_video_has_positive_dimensions(self):
        meta = self._make()
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.fps == 30.0

    def test_no_audio_video(self):
        meta = self._make(has_audio=False)
        assert meta.has_audio is False


class TestAnalysisSummary:
    def _make_meta(self) -> VideoMetadata:
        return VideoMetadata(
            path=Path("/tmp/v.mp4"),
            duration_seconds=60.0,
            width=1280,
            height=720,
            fps=25.0,
            has_audio=True,
            codec="h264",
            file_size_bytes=512000,
        )

    def test_default_format_is_markdown(self):
        summary = AnalysisSummary(
            video=self._make_meta(),
            timeline=[],
            summary_text="# Summary",
            transcript_segments=5,
            keyframes_analyzed=3,
        )
        assert summary.format == OutputFormat.MARKDOWN

    def test_explicit_json_format(self):
        summary = AnalysisSummary(
            video=self._make_meta(),
            timeline=[],
            summary_text='{"key": "value"}',
            transcript_segments=0,
            keyframes_analyzed=0,
            format=OutputFormat.JSON,
        )
        assert summary.format == OutputFormat.JSON

    def test_empty_timeline_is_valid(self):
        summary = AnalysisSummary(
            video=self._make_meta(),
            timeline=[],
            summary_text="",
            transcript_segments=0,
            keyframes_analyzed=0,
        )
        assert summary.timeline == []
        assert summary.transcript_segments == 0
