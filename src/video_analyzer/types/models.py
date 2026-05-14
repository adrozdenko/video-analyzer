"""Core data models for the video analysis pipeline."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class StageStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class StageResult(BaseModel, Generic[T]):
    """Wraps any stage output with status tracking for partial failure handling."""

    stage: str
    status: StageStatus
    data: T | None = None
    error: str | None = None
    duration_ms: float = 0

    @classmethod
    def success(cls, stage: str, data: T, duration_ms: float = 0) -> StageResult[T]:
        return cls(stage=stage, status=StageStatus.SUCCESS, data=data, duration_ms=duration_ms)

    @classmethod
    def skipped(cls, stage: str, reason: str) -> StageResult[T]:
        return cls(stage=stage, status=StageStatus.SKIPPED, error=reason)

    @classmethod
    def fail(cls, stage: str, error: str, duration_ms: float = 0) -> StageResult[T]:
        return cls(stage=stage, status=StageStatus.ERROR, error=error, duration_ms=duration_ms)

    @property
    def ok(self) -> bool:
        return self.status == StageStatus.SUCCESS


class VideoMetadata(BaseModel):
    path: Path
    duration_seconds: float
    width: int
    height: int
    fps: float
    has_audio: bool
    codec: str
    file_size_bytes: int


class TranscriptSegment(BaseModel):
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str


class Keyframe(BaseModel):
    path: Path = Field(description="Path to extracted frame image")
    timestamp: float = Field(description="Time in seconds from video start")
    index: int


class VisualDescription(BaseModel):
    timestamp: float
    keyframe_index: int
    description: str
    objects: list[str] = Field(default_factory=list)
    text_detected: str | None = None


class TimelineEntry(BaseModel):
    timestamp: float
    end_timestamp: float | None = None
    transcript: str | None = None
    visual: str | None = None
    objects: list[str] = Field(default_factory=list)
    text_detected: str | None = None


class OutputFormat(str, Enum):
    MARKDOWN = "md"
    JSON = "json"
    TEXT = "txt"


class AnalysisSummary(BaseModel):
    video: VideoMetadata
    timeline: list[TimelineEntry]
    summary_text: str
    transcript_segments: int
    keyframes_analyzed: int
    format: OutputFormat = OutputFormat.MARKDOWN

    def save(self, path: Path) -> None:
        """Write summary_text to a file, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.summary_text, encoding="utf-8")
