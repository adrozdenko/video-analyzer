"""Timeline merger — aligns transcript segments with visual descriptions by timestamp."""

from __future__ import annotations

import time

from video_analyzer.types import (
    StageResult,
    TimelineEntry,
    TranscriptSegment,
    VisualDescription,
)


class TimelineMerger:
    """Merges transcript and visual analysis into a unified timeline."""

    def merge(
        self,
        transcript: StageResult[list[TranscriptSegment]],
        visuals: StageResult[list[VisualDescription]],
    ) -> StageResult[list[TimelineEntry]]:
        start = time.perf_counter()

        has_transcript = transcript.ok and transcript.data
        has_visuals = visuals.ok and visuals.data

        if not has_transcript and not has_visuals:
            return StageResult.fail(
                stage="timeline_merger",
                error="Both transcript and vision stages failed or produced no data",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        entries: list[TimelineEntry] = []

        # Index visuals by nearest timestamp for fast lookup
        visual_map: dict[float, VisualDescription] = {}
        if has_visuals:
            for v in visuals.data:
                visual_map[v.timestamp] = v

        if has_transcript:
            for seg in transcript.data:
                # Find the closest visual description within this segment's time range
                matched_visual = self._find_nearest_visual(
                    seg.start, seg.end, visual_map
                )

                entries.append(
                    TimelineEntry(
                        timestamp=seg.start,
                        end_timestamp=seg.end,
                        transcript=seg.text,
                        visual=matched_visual.description if matched_visual else None,
                        objects=matched_visual.objects if matched_visual else [],
                    )
                )

            # Add any unmatched visuals as visual-only entries
            matched_timestamps = set()
            for seg in transcript.data:
                v = self._find_nearest_visual(seg.start, seg.end, visual_map)
                if v:
                    matched_timestamps.add(v.timestamp)

            for ts, v in visual_map.items():
                if ts not in matched_timestamps:
                    entries.append(
                        TimelineEntry(
                            timestamp=v.timestamp,
                            visual=v.description,
                            objects=v.objects,
                        )
                    )
        else:
            # Vision-only timeline
            for v in visuals.data:
                entries.append(
                    TimelineEntry(
                        timestamp=v.timestamp,
                        visual=v.description,
                        objects=v.objects,
                    )
                )

        entries.sort(key=lambda e: e.timestamp)

        duration_ms = (time.perf_counter() - start) * 1000
        return StageResult.success(
            stage="timeline_merger", data=entries, duration_ms=duration_ms
        )

    def _find_nearest_visual(
        self,
        start: float,
        end: float,
        visual_map: dict[float, VisualDescription],
    ) -> VisualDescription | None:
        """Find the visual description closest to a transcript segment's time range."""
        best: VisualDescription | None = None
        best_dist = float("inf")

        mid = (start + end) / 2
        for ts, v in visual_map.items():
            # Visual must be within 10 seconds of the segment
            if abs(ts - mid) < best_dist and abs(ts - mid) <= 10.0:
                best = v
                best_dist = abs(ts - mid)

        return best
