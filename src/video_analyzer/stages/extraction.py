"""Extraction stages: audio, keyframes, and video probing via ffmpeg."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

from video_analyzer.types import Keyframe, StageResult, VideoMetadata

logger = logging.getLogger(__name__)

STAGE_AUDIO = "audio_extraction"
STAGE_KEYFRAME = "keyframe_extraction"
STAGE_PROBE = "video_probe"


def _require_ffmpeg() -> None:
    """Raise early if ffmpeg/ffprobe are not on PATH."""
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise EnvironmentError(
                f"'{binary}' not found on PATH. Install ffmpeg: https://ffmpeg.org/download.html"
            )


def _run(cmd: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with standard safety defaults."""
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Video probing
# ---------------------------------------------------------------------------


def probe_video(video_path: Path) -> StageResult[VideoMetadata]:
    """Validate a video file and return its metadata via ffprobe."""
    t0 = time.perf_counter()

    try:
        _require_ffmpeg()

        if not video_path.exists():
            return StageResult.fail(STAGE_PROBE, f"File not found: {video_path}")

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        result = _run(cmd)

        if result.returncode != 0:
            return StageResult.fail(
                STAGE_PROBE,
                f"ffprobe failed (rc={result.returncode}): {result.stderr.strip()}",
                duration_ms=_elapsed(t0),
            )

        info = json.loads(result.stdout)
        streams = info.get("streams", [])
        fmt = info.get("format", {})

        video_stream = next(
            (s for s in streams if s.get("codec_type") == "video"), None
        )
        if video_stream is None:
            return StageResult.fail(
                STAGE_PROBE, "No video stream found", duration_ms=_elapsed(t0)
            )

        has_audio = any(s.get("codec_type") == "audio" for s in streams)

        # Duration: prefer format-level, fall back to stream-level.
        duration = float(
            fmt.get("duration")
            or video_stream.get("duration")
            or 0
        )

        # FPS: parse r_frame_rate (e.g. "30/1" or "30000/1001").
        fps = _parse_frame_rate(video_stream.get("r_frame_rate", "0/1"))

        metadata = VideoMetadata(
            path=video_path,
            duration_seconds=duration,
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            fps=fps,
            has_audio=has_audio,
            codec=video_stream.get("codec_name", "unknown"),
            file_size_bytes=int(fmt.get("size", 0)),
        )

        return StageResult.success(STAGE_PROBE, metadata, duration_ms=_elapsed(t0))

    except EnvironmentError as exc:
        return StageResult.fail(STAGE_PROBE, str(exc), duration_ms=_elapsed(t0))
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        return StageResult.fail(
            STAGE_PROBE,
            f"Failed to parse ffprobe output: {exc}",
            duration_ms=_elapsed(t0),
        )
    except subprocess.TimeoutExpired:
        return StageResult.fail(
            STAGE_PROBE, "ffprobe timed out", duration_ms=_elapsed(t0)
        )


def _parse_frame_rate(rate_str: str) -> float:
    """Parse an ffprobe r_frame_rate string like '30/1' into a float."""
    try:
        num, den = rate_str.split("/")
        denominator = int(den)
        if denominator == 0:
            return 0.0
        return round(int(num) / denominator, 3)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _elapsed(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


class AudioExtractor:
    """Extracts audio from a video file as an MP3."""

    STAGE = STAGE_AUDIO

    def extract(self, video_path: Path, output_dir: Path) -> StageResult[Path]:
        """Extract audio track to MP3.

        Returns StageResult.skipped() when the video has no audio stream.
        """
        t0 = time.perf_counter()

        try:
            _require_ffmpeg()

            # Probe to check for audio presence first.
            probe = probe_video(video_path)
            if not probe.ok or probe.data is None:
                return StageResult.fail(
                    self.STAGE,
                    f"Cannot probe video: {probe.error}",
                    duration_ms=_elapsed(t0),
                )

            if not probe.data.has_audio:
                return StageResult.skipped(
                    self.STAGE, "Video contains no audio stream"
                )

            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{video_path.stem}.mp3"

            cmd = [
                "ffmpeg",
                "-y",               # overwrite without asking
                "-i", str(video_path),
                "-vn",              # drop video
                "-acodec", "libmp3lame",
                "-q:a", "4",       # VBR quality ~165 kbps — good balance
                str(output_path),
            ]
            result = _run(cmd)

            if result.returncode != 0:
                return StageResult.fail(
                    self.STAGE,
                    f"ffmpeg audio extraction failed (rc={result.returncode}): "
                    f"{result.stderr.strip()[-500:]}",
                    duration_ms=_elapsed(t0),
                )

            if not output_path.exists() or output_path.stat().st_size == 0:
                return StageResult.fail(
                    self.STAGE,
                    "Audio file was not created or is empty",
                    duration_ms=_elapsed(t0),
                )

            logger.info(
                "Extracted audio: %s (%.1f KB)",
                output_path.name,
                output_path.stat().st_size / 1024,
            )
            return StageResult.success(self.STAGE, output_path, duration_ms=_elapsed(t0))

        except EnvironmentError as exc:
            return StageResult.fail(self.STAGE, str(exc), duration_ms=_elapsed(t0))
        except subprocess.TimeoutExpired:
            return StageResult.fail(
                self.STAGE, "ffmpeg timed out during audio extraction", duration_ms=_elapsed(t0)
            )


# ---------------------------------------------------------------------------
# Keyframe extraction
# ---------------------------------------------------------------------------


class KeyframeExtractor:
    """Extracts keyframes from a video using ffmpeg scene-change detection."""

    STAGE = STAGE_KEYFRAME

    _DEFAULT_THRESHOLD = 0.3
    _DEFAULT_MAX_KEYFRAMES = 20
    _DEFAULT_MAX_WIDTH = 1280
    _DEFAULT_FALLBACK_INTERVAL = 5.0

    def extract(
        self,
        video_path: Path,
        output_dir: Path,
        threshold: float = _DEFAULT_THRESHOLD,
        max_keyframes: int = _DEFAULT_MAX_KEYFRAMES,
        max_width: int = _DEFAULT_MAX_WIDTH,
        fallback_interval: float = _DEFAULT_FALLBACK_INTERVAL,
    ) -> StageResult[list[Keyframe]]:
        """Extract keyframes via scene detection with a fixed-interval fallback.

        Parameters
        ----------
        video_path:
            Source video file.
        output_dir:
            Directory to write keyframe JPGs into.
        threshold:
            Scene-change sensitivity (0.0–1.0).  Lower = more keyframes.
        max_keyframes:
            Maximum number of keyframes to keep (evenly spaced if exceeded).
        max_width:
            Maximum horizontal resolution for saved frames.
        fallback_interval:
            Seconds between frames when scene detection yields nothing.
        """
        t0 = time.perf_counter()

        try:
            _require_ffmpeg()

            probe = probe_video(video_path)
            if not probe.ok or probe.data is None:
                return StageResult.fail(
                    self.STAGE,
                    f"Cannot probe video: {probe.error}",
                    duration_ms=_elapsed(t0),
                )

            output_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: try scene detection
            keyframes = self._extract_scene_keyframes(
                video_path, output_dir, threshold, max_width
            )

            # Step 2: fallback to fixed interval if scene detection returned nothing
            if not keyframes:
                logger.info(
                    "Scene detection found 0 keyframes — falling back to %.1fs interval",
                    fallback_interval,
                )
                keyframes = self._extract_interval_keyframes(
                    video_path,
                    output_dir,
                    probe.data.duration_seconds,
                    fallback_interval,
                    max_width,
                )

            if not keyframes:
                return StageResult.fail(
                    self.STAGE,
                    "No keyframes could be extracted",
                    duration_ms=_elapsed(t0),
                )

            # Step 3: cap at max_keyframes, keeping evenly spaced subset
            if len(keyframes) > max_keyframes:
                keyframes = self._evenly_sample(keyframes, max_keyframes)

            # Re-index after sampling
            for i, kf in enumerate(keyframes):
                kf.index = i

            logger.info("Extracted %d keyframes from %s", len(keyframes), video_path.name)
            return StageResult.success(self.STAGE, keyframes, duration_ms=_elapsed(t0))

        except EnvironmentError as exc:
            return StageResult.fail(self.STAGE, str(exc), duration_ms=_elapsed(t0))
        except subprocess.TimeoutExpired:
            return StageResult.fail(
                self.STAGE, "ffmpeg timed out during keyframe extraction", duration_ms=_elapsed(t0)
            )

    # -- private helpers -----------------------------------------------------

    def _extract_scene_keyframes(
        self,
        video_path: Path,
        output_dir: Path,
        threshold: float,
        max_width: int,
    ) -> list[Keyframe]:
        """Use ffmpeg scene detection to extract keyframes with timestamps."""
        # Use select filter for scene detection + showinfo to capture timestamps,
        # and scale to limit width while preserving aspect ratio.
        vf = (
            f"select='gt(scene\\,{threshold})',"
            f"scale='min({max_width}\\,iw):-2',"
            f"showinfo"
        )

        frame_pattern = output_dir / "scene_%04d.jpg"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-vsync", "vfr",
            "-q:v", "3",  # JPEG quality (2=best, 31=worst)
            str(frame_pattern),
        ]
        result = _run(cmd, timeout=600)

        if result.returncode != 0:
            logger.warning(
                "Scene detection ffmpeg failed (rc=%d): %s",
                result.returncode,
                result.stderr.strip()[-300:],
            )
            return []

        # Parse timestamps from showinfo output in stderr.
        timestamps = self._parse_showinfo_timestamps(result.stderr)

        # Collect written frame files.
        frames = sorted(output_dir.glob("scene_*.jpg"))

        keyframes: list[Keyframe] = []
        for i, frame_path in enumerate(frames):
            if frame_path.stat().st_size == 0:
                frame_path.unlink(missing_ok=True)
                continue
            ts = timestamps[i] if i < len(timestamps) else 0.0
            keyframes.append(Keyframe(path=frame_path, timestamp=ts, index=i))

        return keyframes

    def _extract_interval_keyframes(
        self,
        video_path: Path,
        output_dir: Path,
        duration: float,
        interval: float,
        max_width: int,
    ) -> list[Keyframe]:
        """Extract one frame every *interval* seconds as a fallback."""
        if duration <= 0:
            return []

        vf = (
            f"fps=1/{interval},"
            f"scale='min({max_width}\\,iw):-2'"
        )

        frame_pattern = output_dir / "interval_%04d.jpg"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-q:v", "3",
            str(frame_pattern),
        ]
        result = _run(cmd, timeout=600)

        if result.returncode != 0:
            logger.warning(
                "Interval extraction failed (rc=%d): %s",
                result.returncode,
                result.stderr.strip()[-300:],
            )
            return []

        frames = sorted(output_dir.glob("interval_*.jpg"))

        keyframes: list[Keyframe] = []
        for i, frame_path in enumerate(frames):
            if frame_path.stat().st_size == 0:
                frame_path.unlink(missing_ok=True)
                continue
            ts = round(i * interval, 3)
            keyframes.append(Keyframe(path=frame_path, timestamp=ts, index=i))

        return keyframes

    @staticmethod
    def _parse_showinfo_timestamps(stderr: str) -> list[float]:
        """Extract pts_time values from ffmpeg showinfo filter output.

        Lines look like:
          [Parsed_showinfo_2 ...] n:   0 pts:  12345 pts_time:1.234 ...
        """
        timestamps: list[float] = []
        for line in stderr.splitlines():
            if "pts_time:" not in line:
                continue
            for token in line.split():
                if token.startswith("pts_time:"):
                    try:
                        timestamps.append(float(token.split(":", 1)[1]))
                    except ValueError:
                        pass
                    break
        return timestamps

    @staticmethod
    def _evenly_sample(items: list[Keyframe], n: int) -> list[Keyframe]:
        """Return *n* evenly spaced items from a sorted list."""
        if n >= len(items):
            return items
        step = len(items) / n
        return [items[int(i * step)] for i in range(n)]
