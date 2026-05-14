"""Tests for merge_timeline and _find_nearest_visual."""

from __future__ import annotations

from video_analyzer.stages.merge import MATCH_WINDOW_SECONDS, _find_nearest_visual, merge_timeline
from video_analyzer.types.models import (
    StageResult,
    TimelineEntry,
    TranscriptSegment,
    VisualDescription,
)


def _seg(start: float, end: float, text: str = "hello") -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def _vis(ts: float, desc: str = "a frame", idx: int = 0) -> VisualDescription:
    return VisualDescription(timestamp=ts, keyframe_index=idx, description=desc, objects=["obj"])


class TestFindNearestVisual:
    def test_returns_closest_visual_within_window(self):
        visual_map = {0.0: _vis(0.0, "first"), 5.0: _vis(5.0, "second")}
        result = _find_nearest_visual(start=3.0, end=5.0, visual_map=visual_map)
        assert result is not None
        assert result.timestamp == 5.0

    def test_returns_none_when_no_visuals_in_window(self):
        visual_map = {100.0: _vis(100.0)}
        # midpoint is 0.5, window is MATCH_WINDOW_SECONDS — 100 is outside
        result = _find_nearest_visual(start=0.0, end=1.0, visual_map=visual_map)
        assert result is None

    def test_returns_none_for_empty_map(self):
        result = _find_nearest_visual(start=0.0, end=2.0, visual_map={})
        assert result is None

    def test_exactly_at_window_boundary_is_included(self):
        ts = MATCH_WINDOW_SECONDS  # midpoint=0, dist=10 → dist <= window
        visual_map = {ts: _vis(ts)}
        result = _find_nearest_visual(start=0.0, end=0.0, visual_map=visual_map)
        assert result is not None
        assert result.timestamp == ts


class TestMergeTimeline:
    def test_merges_transcript_with_nearest_visual(self):
        transcript = StageResult.success("t", [_seg(0.0, 2.0, "words")])
        visuals = StageResult.success("v", [_vis(1.0, "scene")])

        result = merge_timeline(transcript, visuals)

        assert result.ok is True
        assert len(result.data) == 1
        entry = result.data[0]
        assert entry.transcript == "words"
        assert entry.visual == "scene"
        assert entry.objects == ["obj"]

    def test_transcript_only_produces_entries_without_visual(self):
        transcript = StageResult.success("t", [_seg(0.0, 1.0, "text")])
        visuals = StageResult.skipped("v", "audio-only")

        result = merge_timeline(transcript, visuals)

        assert result.ok is True
        assert len(result.data) == 1
        assert result.data[0].transcript == "text"
        assert result.data[0].visual is None

    def test_visuals_only_produces_entries_without_transcript(self):
        transcript = StageResult.skipped("t", "no audio")
        visuals = StageResult.success("v", [_vis(2.0, "frame")])

        result = merge_timeline(transcript, visuals)

        assert result.ok is True
        assert len(result.data) == 1
        assert result.data[0].visual == "frame"
        assert result.data[0].transcript is None

    def test_both_failed_returns_fail(self):
        transcript = StageResult.fail("t", "whisper crashed")
        visuals = StageResult.fail("v", "claude unreachable")

        result = merge_timeline(transcript, visuals)

        assert result.ok is False
        assert "Both transcript and vision" in result.error

    def test_entries_sorted_by_timestamp(self):
        segs = [_seg(5.0, 6.0, "late"), _seg(0.0, 1.0, "early")]
        transcript = StageResult.success("t", segs)
        visuals = StageResult.skipped("v", "none")

        result = merge_timeline(transcript, visuals)

        assert result.ok is True
        timestamps = [e.timestamp for e in result.data]
        assert timestamps == sorted(timestamps)

    def test_unmatched_visuals_appended_as_visual_only_entries(self):
        # transcript at t=0, visual at t=50 (outside match window)
        transcript = StageResult.success("t", [_seg(0.0, 1.0)])
        visuals = StageResult.success("v", [_vis(0.5, "near"), _vis(50.0, "far")])

        result = merge_timeline(transcript, visuals)

        assert result.ok is True
        # Should have transcript entry (matched with near) + unmatched far visual
        assert len(result.data) == 2
        visual_only = [e for e in result.data if e.transcript is None]
        assert len(visual_only) == 1
        assert visual_only[0].visual == "far"
