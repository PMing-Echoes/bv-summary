"""Turn timed ASR segments into readable plain text."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Segment

_CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")
_CJK_END_PUNCT = "。！？；：、，…"
_LATIN_END_PUNCT = ".!?;:,\"'”’）)]】》"
_SENTENCE_END_RE = re.compile(r"[.!?。！？…][\"'”’）)\]}】》]*$")


def _is_cjk(char: str) -> bool:
    return bool(char) and bool(_CJK_CHAR_RE.match(char))


def join_asr_pieces(left: str, right: str) -> str:
    """Join two ASR fragments without inserting spaces around CJK text."""
    if not left:
        return right
    if not right:
        return left
    if left[-1].isspace() or right[0].isspace():
        return left + right
    if _is_cjk(left[-1]) or _is_cjk(right[0]):
        return left + right
    if left[-1] in _CJK_END_PUNCT or right[0] in _LATIN_END_PUNCT[1:]:
        return left + right
    return f"{left} {right}"


def segments_to_text(segments: Iterable[Segment], *, pause_threshold: float = 1.5) -> str:
    """Merge timed segments, starting a new paragraph after a sentence + pause.

    Whisper emits many short segments. Joining every segment with a newline
    makes a fragmented transcript, while joining everything with spaces makes a
    wall of text. We insert paragraphs only at a sentence boundary followed by
    a genuine pause, preserving useful structure without inventing content.
    """
    paragraphs: list[str] = []
    current = ""
    previous_end: float | None = None

    for segment in segments:
        piece = segment.text.strip()
        if not piece:
            continue
        gap = segment.start - previous_end if previous_end is not None else 0.0
        if current and gap >= pause_threshold and _SENTENCE_END_RE.search(current):
            paragraphs.append(current.strip())
            current = piece
        else:
            current = join_asr_pieces(current, piece)
        previous_end = segment.end

    if current.strip():
        paragraphs.append(current.strip())
    return "\n\n".join(paragraphs)
