"""Pipeline configuration loaded from .env + CLI args."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class WhisperModel(str, Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class VisionProvider(str, Enum):
    CLAUDE = "claude"


class Settings(BaseSettings):
    """Loaded from environment variables and .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Whisper
    whisper_model: WhisperModel = WhisperModel.MEDIUM

    # Vision
    vision_provider: VisionProvider = VisionProvider.CLAUDE
    vision_concurrency: int = Field(default=3, ge=1, le=10)

    # Extraction
    scene_threshold: float = Field(default=0.3, ge=0.05, le=0.9)
    max_keyframes: int = Field(default=20, ge=1, le=100)
    keyframe_max_width: int = Field(default=1024, ge=256, le=4096)
    fallback_interval_seconds: float = Field(default=5.0, ge=1.0)

    # Output
    output_format: str = "md"
    verbose: bool = False


def load_settings(**overrides: object) -> Settings:
    """Load settings from .env, then apply CLI overrides."""
    # Filter out None values so defaults from .env aren't overridden
    clean = {k: v for k, v in overrides.items() if v is not None}
    return Settings(**clean)
