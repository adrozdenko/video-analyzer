"""Analysis stages: local Whisper transcription and Claude vision analysis."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from pathlib import Path

import anthropic
import whisper

from video_analyzer.types import (
    Keyframe,
    StageResult,
    TranscriptSegment,
    VisualDescription,
)

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
        api_key: str,
        concurrency: int = 3,
    ) -> StageResult[list[VisualDescription]]:
        """Analyze keyframes using Claude vision.

        Args:
            keyframes: Extracted video keyframes with image paths.
            api_key: Anthropic API key.
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
        client = anthropic.AsyncAnthropic(api_key=api_key)
        semaphore = asyncio.Semaphore(concurrency)

        async def _analyze_one(keyframe: Keyframe) -> VisualDescription | None:
            async with semaphore:
                return await self._describe_keyframe(client, keyframe)

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
        client: anthropic.AsyncAnthropic,
        keyframe: Keyframe,
    ) -> VisualDescription | None:
        """Send a single keyframe to Claude for description.

        Returns None if the image file is unreadable.
        """
        if not keyframe.path.exists():
            logger.warning("Keyframe image not found: %s", keyframe.path)
            return None

        image_bytes = keyframe.path.read_bytes()
        if not image_bytes:
            logger.warning("Keyframe image is empty: %s", keyframe.path)
            return None

        base64_data = base64.b64encode(image_bytes).decode("utf-8")

        # Detect media type from extension
        suffix = keyframe.path.suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")

        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": VISION_PROMPT,
                        },
                    ],
                }
            ],
        )

        raw_text = message.content[0].text
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
