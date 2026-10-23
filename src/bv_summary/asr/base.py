"""ASR backend interface and selection value object."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..models import AudioFile, ProgressCallback, Transcript


class ASRBackend(ABC):
    """A pluggable speech-to-text implementation."""

    name: str
    model: str

    @abstractmethod
    def transcribe(
        self,
        audio: AudioFile,
        *,
        progress: ProgressCallback | None = None,
    ) -> Transcript:
        """Transcribe *audio* and return timed text segments."""


@dataclass(frozen=True, slots=True)
class ASRSelection:
    backends: tuple[ASRBackend, ...]
    notes: tuple[str, ...] = ()
