"""Shared data structures and callback types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

# A progress callback receives either a 0..100 percentage or None for an
# indeterminate / status-only update, plus a short human-readable message.
ProgressCallback: TypeAlias = Callable[[float | None, str], None]


@dataclass(frozen=True, slots=True)
class VideoInfo:
    title: str
    video_id: str
    webpage_url: str
    duration: float | None = None
    uploader: str | None = None
    extractor: str | None = None


@dataclass(frozen=True, slots=True)
class MediaFile:
    path: Path
    info: VideoInfo


@dataclass(frozen=True, slots=True)
class AudioFile:
    path: Path
    duration: float
    sample_rate: int
    channels: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    segments: tuple[Segment, ...]
    language: str | None
    backend: str
    model: str


@dataclass(frozen=True, slots=True)
class Correction:
    text: str
    backend: str
    model: str
    chunks: int


@dataclass(frozen=True, slots=True)
class RunResult:
    """Everything the CLI needs to summarize a completed pipeline run."""

    title: str
    source_url: str
    uploader: str | None
    duration: float | None
    language: str | None
    asr_backend: str
    asr_model: str
    segment_count: int
    raw_chars: int
    llm_backend: str | None
    llm_model: str | None
    llm_chunks: int | None
    text: str
    text_corrected: bool
    output_path: Path
    raw_path: Path | None
    audio_path: Path | None
    elapsed_seconds: float
