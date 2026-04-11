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


class Transcriber:
    """Local Whisper-based audio transcription."""

    def transcribe(
        self,
        audio_path: Path,
        model_name: str = "medium",
    ) -> StageResult[list[TranscriptSegment]]:
        """Transcribe an audio file using OpenAI Whisper.

        Args:
            audio_path: Path to the audio file.
            model_name: Whisper model size (tiny, base, small, medium, large).

        Returns:
            StageResult wrapping a list of TranscriptSegment on success.
        """
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

            segments: list[TranscriptSegment] = [
                TranscriptSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                )
                for seg in result.get("segments", [])
                if seg.get("text", "").strip()
            ]

            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Transcription complete: %d segments in %.1f ms",
                len(segments),
                duration_ms,
            )
            return StageResult.success(
                stage=STAGE_TRANSCRIBE,
                data=segments,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception("Transcription failed for %s", audio_path)
            return StageResult.fail(
                stage=STAGE_TRANSCRIBE,
                error=str(exc),
                duration_ms=duration_ms,
            )


class VisionAnalyzer:
    """Claude-powered keyframe visual analysis."""

    async def analyze(
        self,
        keyframes: list[Keyframe],
        concurrency: int = 3,
    ) -> StageResult[list[VisualDescription]]:
        """Analyze keyframes using Claude vision via OAuth (Max subscription).

        Args:
            keyframes: Extracted video keyframes with image paths.
            concurrency: Maximum parallel API calls.

        Returns:
            StageResult wrapping a list of VisualDescription on success.
        """
        if not keyframes:
            return StageResult.skipped(
                stage=STAGE_VISION,
                reason="No keyframes to analyze",
            )

        start = time.perf_counter()
        if not get_claude_cli_path():
            return StageResult.fail(
                stage=STAGE_VISION,
                error="Claude CLI not found. Install: npm install -g @anthropic-ai/claude-code",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        semaphore = asyncio.Semaphore(concurrency)

        async def _analyze_one(keyframe: Keyframe) -> VisualDescription | None:
            async with semaphore:
                return await self._describe_keyframe(keyframe)

        tasks = [_analyze_one(kf) for kf in keyframes]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        descriptions: list[VisualDescription] = []
        errors: list[str] = []

        for i, result in enumerate(raw_results):
            if isinstance(result, Exception):
                msg = f"Keyframe {keyframes[i].index} failed: {result}"
                logger.warning(msg)
                errors.append(msg)
            elif result is None:
                msg = f"Keyframe {keyframes[i].index}: no result returned"
                logger.warning(msg)
                errors.append(msg)
            else:
                descriptions.append(result)

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
                len(errors),
                len(keyframes),
                len(descriptions),
            )

        logger.info(
            "Vision analysis complete: %d/%d keyframes in %.1f ms",
            len(descriptions),
            len(keyframes),
            duration_ms,
        )
        return StageResult.success(
            stage=STAGE_VISION,
            data=descriptions,
            duration_ms=duration_ms,
        )

    async def _describe_keyframe(
        self,
        keyframe: Keyframe,
    ) -> VisualDescription | None:
        """Send a single keyframe to Claude for description via CLI.

        Returns None if the image file is unreadable.
        """
        if not keyframe.path.exists():
            logger.warning("Keyframe image not found: %s", keyframe.path)
            return None

        if keyframe.path.stat().st_size == 0:
            logger.warning("Keyframe image is empty: %s", keyframe.path)
            return None

        prompt = f"Use the Read tool to view the image at {keyframe.path}. Then analyze it."

        raw_text = await call_claude_async(
            prompt,
            system_prompt=VISION_PROMPT,
            model="haiku",
            tools="Read",
        )

        return self._parse_vision_response(raw_text, keyframe)

    @staticmethod
    def _parse_vision_response(
        raw_text: str,
        keyframe: Keyframe,
    ) -> VisualDescription:
        """Parse Claude's JSON response into a VisualDescription.

        Falls back to using the raw text as the description if JSON parsing fails.
        """
        try:
            # Strip markdown code fences if present
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                first_newline = cleaned.index("\n")
                last_fence = cleaned.rfind("```")
                cleaned = cleaned[first_newline + 1 : last_fence].strip()

            data = json.loads(cleaned)
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
