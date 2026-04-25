"""Analysis stages: local Whisper transcription and Claude vision analysis."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import whisper

from video_analyzer.types import (
    Keyframe,
    StageResult,
    TranscriptSegment,
    VisualDescription,
)
from video_analyzer.utils.claude_cli import call_claude_async, get_claude_cli_path

logger = logging.getLogger(__name__)

STAGE_TRANSCRIBE = "transcribe"
STAGE_VISION = "vision_analysis"

VISION_PROMPT = (
    "Describe this video frame concisely. Include:\n"
    "1. The main visual content and scene\n"
    "2. Notable objects present (list them)\n"
    "3. Any text visible in the frame\n\n"
    "Respond in JSON with keys: description, objects (array of strings), text_detected (string or null)."
)


def transcribe(
    audio_path: Path,
    model_name: str = "medium",
) -> StageResult[list[TranscriptSegment]]:
    """Transcribe an audio file using OpenAI Whisper."""
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        return StageResult.skipped(
            stage=STAGE_TRANSCRIBE,
            reason=f"Audio file missing or empty: {audio_path}",
        )

    start = time.perf_counter()
    try:
        logger.info("Loading Whisper model '%s'", model_name)
        model = whisper.load_model(model_name)

        logger.info("Transcribing %s", audio_path)
        result = model.transcribe(str(audio_path))

        segments = [
            TranscriptSegment(start=seg["start"], end=seg["end"], text=seg["text"].strip())
            for seg in result.get("segments", [])
            if seg.get("text", "").strip()
        ]

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Transcription complete: %d segments in %.1f ms", len(segments), duration_ms
        )
        return StageResult.success(
            stage=STAGE_TRANSCRIBE, data=segments, duration_ms=duration_ms
        )

    except (RuntimeError, OSError, ValueError) as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception("Transcription failed for %s", audio_path)
        return StageResult.fail(
            stage=STAGE_TRANSCRIBE, error=str(exc), duration_ms=duration_ms
        )


async def analyze_keyframes(
    keyframes: list[Keyframe],
    concurrency: int = 3,
) -> StageResult[list[VisualDescription]]:
    """Analyze keyframes using Claude vision via the claude CLI."""
    if not keyframes:
        return StageResult.skipped(stage=STAGE_VISION, reason="No keyframes to analyze")

    start = time.perf_counter()
    if not get_claude_cli_path():
        return StageResult.fail(
            stage=STAGE_VISION,
            error="Claude CLI not found. Install: npm install -g @anthropic-ai/claude-code",
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    semaphore = asyncio.Semaphore(concurrency)

    async def describe(kf: Keyframe) -> VisualDescription | None:
        async with semaphore:
            return await _describe_keyframe(kf)

    raw_results = await asyncio.gather(
        *(describe(kf) for kf in keyframes), return_exceptions=True
    )

    descriptions: list[VisualDescription] = []
    errors: list[str] = []
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            errors.append(f"Keyframe {keyframes[i].index} failed: {result}")
        elif result is None:
            errors.append(f"Keyframe {keyframes[i].index}: no result returned")
        else:
            descriptions.append(result)

    for err in errors:
        logger.warning(err)

    duration_ms = (time.perf_counter() - start) * 1000

    if not descriptions:
        return StageResult.fail(
            stage=STAGE_VISION,
            error=f"All keyframes failed: {'; '.join(errors)}",
            duration_ms=duration_ms,
        )

    if errors:
        logger.warning(
            "%d/%d keyframes failed, continuing with %d results",
            len(errors), len(keyframes), len(descriptions),
        )

    logger.info(
        "Vision analysis complete: %d/%d keyframes in %.1f ms",
        len(descriptions), len(keyframes), duration_ms,
    )
    return StageResult.success(
        stage=STAGE_VISION, data=descriptions, duration_ms=duration_ms
    )


async def _describe_keyframe(keyframe: Keyframe) -> VisualDescription | None:
    if not keyframe.path.exists():
        logger.warning("Keyframe image not found: %s", keyframe.path)
        return None
    if keyframe.path.stat().st_size == 0:
        logger.warning("Keyframe image is empty: %s", keyframe.path)
        return None

    prompt = f"Use the Read tool to view the image at {keyframe.path}. Then analyze it."
    raw_text = await call_claude_async(
        prompt, system_prompt=VISION_PROMPT, model="haiku", tools="Read",
    )
    return _parse_vision_response(raw_text, keyframe)


def _parse_vision_response(raw_text: str, keyframe: Keyframe) -> VisualDescription:
    """Parse Claude's JSON response, falling back to raw text if JSON parsing fails."""
    try:
        data = json.loads(raw_text)
        return VisualDescription(
            timestamp=keyframe.timestamp,
            keyframe_index=keyframe.index,
            description=data.get("description", raw_text),
            objects=data.get("objects", []),
            text_detected=data.get("text_detected"),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        logger.debug("Could not parse vision JSON (%s), using raw text", exc)
        return VisualDescription(
            timestamp=keyframe.timestamp,
            keyframe_index=keyframe.index,
            description=raw_text.strip(),
            objects=[],
            text_detected=None,
        )
