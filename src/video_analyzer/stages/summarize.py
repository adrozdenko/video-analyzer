"""Summarizer — generates final output from unified timeline using Claude."""

from __future__ import annotations

import json
import time

import anthropic

from video_analyzer.types import (
    AnalysisSummary,
    OutputFormat,
    StageResult,
    TimelineEntry,
    VideoMetadata,
)
from video_analyzer.utils.claude_client import create_claude_client


class Summarizer:
    """Generates a structured summary from the unified timeline."""

    def __init__(self):
        self._client = create_claude_client()

    def summarize(
        self,
        video: VideoMetadata,
        timeline: list[TimelineEntry],
        output_format: OutputFormat = OutputFormat.MARKDOWN,
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
            summary_text = self._call_claude(video, timeline_text, output_format)
        except Exception as e:
            # Fallback: produce raw timeline as output
            summary_text = self._fallback_summary(video, timeline, output_format)
            if not summary_text:
                return StageResult.fail(
                    stage="summarizer",
                    error=f"Claude API failed and fallback produced no output: {e}",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

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
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def _call_claude(
        self,
        video: VideoMetadata,
        timeline_text: str,
        output_format: OutputFormat,
    ) -> str:
        format_instruction = {
            OutputFormat.MARKDOWN: "Format your response as clean Markdown with headers.",
            OutputFormat.JSON: "Format your response as a JSON object with keys: overview, key_moments (array), visual_content, transcript_highlights.",
            OutputFormat.TEXT: "Format your response as plain text with clear sections.",
        }[output_format]

        prompt = f"""Analyze this video timeline and produce a comprehensive summary.

Video info: {video.duration_seconds:.0f}s duration, {video.width}x{video.height}, {video.codec}

Timeline (timestamp | speech/visual content):
{timeline_text}

Create a summary with these sections:
1. **Overview** — What is this video about? (2-3 sentences)
2. **Key Moments** — Important timestamps with descriptions
3. **Visual Content** — What was shown visually
4. **Transcript Highlights** — Key spoken content

{format_instruction}"""

        message = self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text

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
            lines.append("")

        lines.append("\n---\n*AI summarization unavailable — raw timeline data*")
        return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
