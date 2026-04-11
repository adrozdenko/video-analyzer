"""Summarizer — generates final output from unified timeline using Claude."""

from __future__ import annotations

import json
import time

from video_analyzer.config.settings import DetailMode
from video_analyzer.types import (
    AnalysisSummary,
    OutputFormat,
    StageResult,
    TimelineEntry,
    VideoMetadata,
)
from video_analyzer.utils.claude_cli import call_claude_sync


class Summarizer:
    """Generates a structured summary from the unified timeline."""

    def summarize(
        self,
        video: VideoMetadata,
        timeline: list[TimelineEntry],
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        detail_mode: DetailMode = DetailMode.SUMMARY,
    ) -> StageResult[AnalysisSummary]:
        start = time.perf_counter()

        if not timeline:
            return StageResult.fail(
                stage="summarizer",
                error="No timeline entries to summarize",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        transcript_count = sum(1 for e in timeline if e.transcript)
        visual_count = sum(1 for e in timeline if e.visual)

        # Build timeline text for the prompt
        timeline_text = self._format_timeline_for_prompt(timeline)

        try:
            summary_text = self._call_claude(video, timeline_text, output_format, detail_mode)
        except Exception as e:
            # Fallback: produce raw timeline as output
            summary_text = self._fallback_summary(video, timeline, output_format)
            if not summary_text:
                return StageResult.fail(
                    stage="summarizer",
                    error=f"Claude API failed and fallback produced no output: {e}",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

        summary_text = self._append_timeline_appendix(summary_text, timeline, output_format)

        result = AnalysisSummary(
            video=video,
            timeline=timeline,
            summary_text=summary_text,
            transcript_segments=transcript_count,
            keyframes_analyzed=visual_count,
            format=output_format,
        )

        duration_ms = (time.perf_counter() - start) * 1000
        return StageResult.success(stage="summarizer", data=result, duration_ms=duration_ms)

    def _format_timeline_for_prompt(self, timeline: list[TimelineEntry]) -> str:
        lines: list[str] = []
        for entry in timeline:
            ts = _fmt_time(entry.timestamp)
            parts: list[str] = [f"[{ts}]"]
            if entry.transcript:
                parts.append(f'Speech: "{entry.transcript}"')
            if entry.visual:
                parts.append(f"Visual: {entry.visual}")
            if entry.objects:
                parts.append(f"Objects: {', '.join(entry.objects)}")
            if entry.text_detected:
                parts.append(f"On-screen text: {entry.text_detected}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def _append_timeline_appendix(
        self,
        summary_text: str,
        timeline: list[TimelineEntry],
        output_format: OutputFormat,
    ) -> str:
        """Append raw timeline evidence so visual analysis is always surfaced."""
        if output_format == OutputFormat.JSON:
            return self._append_json_timeline(summary_text, timeline)
        if output_format == OutputFormat.TEXT:
            return summary_text.rstrip() + "\n\n" + self._render_text_timeline_appendix(timeline)
        return summary_text.rstrip() + "\n\n" + self._render_markdown_timeline_appendix(timeline)

    def _append_json_timeline(self, summary_text: str, timeline: list[TimelineEntry]) -> str:
        try:
            payload = json.loads(summary_text)
        except json.JSONDecodeError:
            payload = {"summary": summary_text}

        payload["timeline_evidence"] = [
            {
                "timestamp": _fmt_time(entry.timestamp),
                "end_timestamp": _fmt_time(entry.end_timestamp) if entry.end_timestamp is not None else None,
                "transcript": entry.transcript,
                "visual": entry.visual,
                "objects": entry.objects,
                "text_detected": entry.text_detected,
            }
            for entry in timeline
        ]
        return json.dumps(payload, indent=2)

    def _render_markdown_timeline_appendix(self, timeline: list[TimelineEntry]) -> str:
        lines = ["## Timeline Evidence", ""]
        for entry in timeline:
            ts = _fmt_time(entry.timestamp)
            end_ts = _fmt_time(entry.end_timestamp) if entry.end_timestamp is not None else None
            heading = f"### [{ts}-{end_ts}]" if end_ts else f"### [{ts}]"
            lines.append(heading)
            if entry.transcript:
                lines.append(f"**Speech:** {entry.transcript}")
            if entry.visual:
                lines.append(f"**Visual:** {entry.visual}")
            if entry.objects:
                lines.append(f"**Objects:** {', '.join(entry.objects)}")
            if entry.text_detected:
                lines.append(f"**On-screen text:** {entry.text_detected}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _render_text_timeline_appendix(self, timeline: list[TimelineEntry]) -> str:
        lines = ["Timeline Evidence", "=================", ""]
        for entry in timeline:
            ts = _fmt_time(entry.timestamp)
            end_ts = _fmt_time(entry.end_timestamp) if entry.end_timestamp is not None else None
            heading = f"[{ts}-{end_ts}]" if end_ts else f"[{ts}]"
            lines.append(heading)
            if entry.transcript:
                lines.append(f"Speech: {entry.transcript}")
            if entry.visual:
                lines.append(f"Visual: {entry.visual}")
            if entry.objects:
                lines.append(f"Objects: {', '.join(entry.objects)}")
            if entry.text_detected:
                lines.append(f"On-screen text: {entry.text_detected}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _call_claude(
        self,
        video: VideoMetadata,
        timeline_text: str,
        output_format: OutputFormat,
        detail_mode: DetailMode = DetailMode.SUMMARY,
    ) -> str:
        format_instruction = {
            OutputFormat.MARKDOWN: "Format your response as clean Markdown with headers.",
            OutputFormat.JSON: "Format your response as a JSON object with keys: overview, key_moments (array), visual_content, transcript_highlights.",
            OutputFormat.TEXT: "Format your response as plain text with clear sections.",
        }[output_format]

        video_info = f"Video info: {video.duration_seconds:.0f}s duration, {video.width}x{video.height}, {video.codec}"

        if detail_mode == DetailMode.DETAILED:
            prompt = f"""Analyze this video timeline and extract ALL knowledge in exhaustive detail.

{video_info}

Timeline (timestamp | speech/visual content):
{timeline_text}

Extract EVERYTHING from this video with maximum detail. Include:

1. **Overview** — What is this video about, who is presenting, what is the context
2. **Full Transcript** — Complete reconstructed transcript organized by timestamp, preserving the speaker's exact words and arguments as closely as possible
3. **Key Arguments & Ideas** — Every distinct argument, claim, or idea presented, with supporting details and reasoning
4. **Technical Details** — Any specific technologies, frameworks, architectures, patterns, or concepts mentioned, with the presenter's opinion on each
5. **Visual Content** — Detailed description of everything shown visually: diagrams, whiteboard drawings, code, slides, UI elements, gestures
6. **Actionable Takeaways** — Concrete advice, recommendations, or principles the presenter communicates
7. **Timeline Breakdown** — Second-by-second breakdown of what happens and what is said at each point
8. **Quotes** — All notable direct quotes with timestamps

Be exhaustive. Include every detail. Do NOT summarize or condense — extract ALL information.

{format_instruction}"""
        else:
            prompt = f"""Analyze this video timeline and produce a comprehensive summary.

{video_info}

Timeline (timestamp | speech/visual content):
{timeline_text}

Create a summary with these sections:
1. **Overview** — What is this video about? (2-3 sentences)
2. **Key Moments** — Important timestamps with descriptions
3. **Visual Content** — What was shown visually
4. **Transcript Highlights** — Key spoken content

Be concise. No extra commentary beyond what is asked.

{format_instruction}"""

        return call_claude_sync(prompt, model="haiku")

    def _fallback_summary(
        self,
        video: VideoMetadata,
        timeline: list[TimelineEntry],
        output_format: OutputFormat,
    ) -> str:
        """Produce a raw timeline dump when Claude API is unavailable."""
        if output_format == OutputFormat.JSON:
            data = {
                "overview": f"Video analysis of {video.path.name} ({video.duration_seconds:.0f}s)",
                "key_moments": [
                    {
                        "time": _fmt_time(e.timestamp),
                        "transcript": e.transcript,
                        "visual": e.visual,
                    }
                    for e in timeline
                ],
                "note": "AI summarization unavailable — raw timeline data",
            }
            return json.dumps(data, indent=2)

        lines: list[str] = []
        lines.append(f"# Video Analysis: {video.path.name}")
        lines.append(f"\nDuration: {video.duration_seconds:.0f}s | {video.width}x{video.height}")
        lines.append("\n## Timeline\n")

        for entry in timeline:
            ts = _fmt_time(entry.timestamp)
            lines.append(f"### [{ts}]")
            if entry.transcript:
                lines.append(f"**Speech:** {entry.transcript}")
            if entry.visual:
                lines.append(f"**Visual:** {entry.visual}")
            if entry.objects:
                lines.append(f"**Objects:** {', '.join(entry.objects)}")
            if entry.text_detected:
                lines.append(f"**On-screen text:** {entry.text_detected}")
            lines.append("")

        lines.append("\n---\n*AI summarization unavailable — raw timeline data*")
        return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
