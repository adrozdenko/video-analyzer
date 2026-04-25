"""Timeline merger — aligns transcript segments with visual descriptions by timestamp."""

from __future__ import annotations

import time

from video_analyzer.types import (
    StageResult,
    TimelineEntry,
    TranscriptSegment,
    VisualDescription,
)

STAGE = "timeline_merger"
MATCH_WINDOW_SECONDS = 10.0


def merge_timeline(
    transcript: StageResult[list[TranscriptSegment]],
    visuals: StageResult[list[VisualDescription]],
) -> StageResult[list[TimelineEntry]]:
    start = time.perf_counter()

    has_transcript = transcript.ok and transcript.data
    has_visuals = visuals.ok and visuals.data

    if not has_transcript and not has_visuals:
        return StageResult.fail(
            stage=STAGE,
            error="Both transcript and vision stages failed or produced no data",
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    visual_map: dict[float, VisualDescription] = (
        {v.timestamp: v for v in visuals.data} if has_visuals else {}
    )

    entries: list[TimelineEntry] = []
    matched_timestamps: set[float] = set()

    if has_transcript:
        for seg in transcript.data:
            matched = _find_nearest_visual(seg.start, seg.end, visual_map)
            if matched:
                matched_timestamps.add(matched.timestamp)
            entries.append(
                TimelineEntry(
                    timestamp=seg.start,
                    end_timestamp=seg.end,
                    transcript=seg.text,
                    visual=matched.description if matched else None,
                    objects=matched.objects if matched else [],
                    text_detected=matched.text_detected if matched else None,
                )
            )

    for ts, v in visual_map.items():
        if ts in matched_timestamps:
            continue
        entries.append(
            TimelineEntry(
                timestamp=v.timestamp,
                visual=v.description,
                objects=v.objects,
                text_detected=v.text_detected,
            )
        )

    entries.sort(key=lambda e: e.timestamp)

    return StageResult.success(
        stage=STAGE,
        data=entries,
        duration_ms=(time.perf_counter() - start) * 1000,
    )


def _find_nearest_visual(
    start: float,
    end: float,
    visual_map: dict[float, VisualDescription],
) -> VisualDescription | None:
    mid = (start + end) / 2
    best: VisualDescription | None = None
    best_dist = float("inf")
    for ts, v in visual_map.items():
        dist = abs(ts - mid)
        if dist <= MATCH_WINDOW_SECONDS and dist < best_dist:
            best = v
            best_dist = dist
    return best
