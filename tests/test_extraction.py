"""Tests for probe_video and extract_audio with mocked ffprobe/ffmpeg."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_analyzer.stages.extraction import extract_audio, probe_video
from video_analyzer.types.models import StageStatus


def _ffprobe_output(*, has_video: bool = True, has_audio: bool = True) -> str:
    streams = []
    if has_video:
        streams.append({
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "30/1",
            "duration": "120.0",
        })
    if has_audio:
        streams.append({
            "codec_type": "audio",
            "codec_name": "aac",
            "duration": "120.0",
        })
    return json.dumps({
        "streams": streams,
        "format": {"duration": "120.0", "size": "1048576"},
    })


def _mock_run(stdout: str, returncode: int = 0):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = ""
    return proc


class TestProbeVideo:
    def test_normal_video_returns_metadata(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        with patch("video_analyzer.stages.extraction._run", return_value=_mock_run(_ffprobe_output())):
            result = probe_video(video)

        assert result.ok is True
        meta = result.data
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.duration_seconds == 120.0
        assert meta.has_audio is True
        assert meta.codec == "h264"

    def test_audio_only_mp3_returns_metadata_with_zero_width(self, tmp_path):
        mp3 = tmp_path / "audio.mp3"
        mp3.write_bytes(b"fake")

        with patch("video_analyzer.stages.extraction._run",
                   return_value=_mock_run(_ffprobe_output(has_video=False, has_audio=True))):
            result = probe_video(mp3)

        assert result.ok is True
        meta = result.data
        assert meta.width == 0
        assert meta.height == 0
        assert meta.has_audio is True
        assert meta.codec == "aac"

    def test_file_with_no_streams_fails(self, tmp_path):
        bad = tmp_path / "empty.mp4"
        bad.write_bytes(b"fake")

        with patch("video_analyzer.stages.extraction._run",
                   return_value=_mock_run(json.dumps({"streams": [], "format": {}}))):
            result = probe_video(bad)

        assert result.ok is False
        assert "No video or audio stream found" in result.error

    def test_missing_file_fails(self, tmp_path):
        missing = tmp_path / "missing.mp4"
        result = probe_video(missing)
        assert result.ok is False
        assert "not found" in result.error.lower()

    def test_ffprobe_nonzero_returncode_fails(self, tmp_path):
        video = tmp_path / "bad.mp4"
        video.write_bytes(b"fake")

        with patch("video_analyzer.stages.extraction._run",
                   return_value=_mock_run("", returncode=1)):
            result = probe_video(video)

        assert result.ok is False
        assert result.status == StageStatus.ERROR


class TestExtractAudio:
    def test_video_with_no_audio_returns_skipped(self, tmp_path):
        video = tmp_path / "silent.mp4"
        video.write_bytes(b"fake")

        with patch("video_analyzer.stages.extraction._run",
                   return_value=_mock_run(_ffprobe_output(has_audio=False))):
            result = extract_audio(video, tmp_path / "out")

        assert result.status == StageStatus.SKIPPED
        assert "no audio stream" in result.error.lower()

    def test_audio_only_input_skips_ffmpeg_conversion(self, tmp_path):
        mp3 = tmp_path / "audio.mp3"
        mp3.write_bytes(b"fake audio data")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        probe_out = _ffprobe_output(has_video=False, has_audio=True)

        with patch("video_analyzer.stages.extraction._run", return_value=_mock_run(probe_out)) as mock_run:
            result = extract_audio(mp3, out_dir)

        # For audio-only input the file is copied directly — ffmpeg not called again
        assert mock_run.call_count == 1  # only the probe call
        assert result.ok is True
        assert result.data.suffix == ".mp3"

    def test_failed_probe_propagates_error(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        with patch("video_analyzer.stages.extraction._run",
                   return_value=_mock_run("", returncode=1)):
            result = extract_audio(video, tmp_path / "out")

        assert result.ok is False
