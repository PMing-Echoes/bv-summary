"""Application-level exceptions with stable process exit codes."""

from __future__ import annotations


class AppError(Exception):
    """A user-facing failure. The CLI prints message + optional hint."""

    exit_code = 1

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class UserInputError(AppError):
    """Bad CLI input (for example, a non-HTTP URL)."""

    exit_code = 2


class ConfigError(AppError):
    """Invalid or incomplete configuration."""

    exit_code = 2


class DependencyError(AppError):
    """A required package or executable is missing."""

    exit_code = 3


class DownloadError(AppError):
    """The source video/audio could not be fetched."""

    exit_code = 4


class AudioError(AppError):
    """Audio extraction or inspection failed."""

    exit_code = 5


class ASRError(AppError):
    """Speech recognition failed."""

    exit_code = 6


class LLMError(AppError):
    """LLM correction failed."""

    exit_code = 7


class OutputError(AppError):
    """Writing output files failed."""

    exit_code = 8
