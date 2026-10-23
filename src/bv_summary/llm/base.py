"""LLM correction backend interface and selection value object."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..models import Correction, ProgressCallback


class LLMBackend(ABC):
    """A pluggable text-correction implementation."""

    name: str
    model: str
    base_url: str | None

    @abstractmethod
    def correct(
        self,
        text: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> Correction:
        """Return corrected text. Raise LLMError on failure."""


@dataclass(frozen=True, slots=True)
class LLMSelection:
    backends: tuple[LLMBackend, ...]
    notes: tuple[str, ...] = ()
