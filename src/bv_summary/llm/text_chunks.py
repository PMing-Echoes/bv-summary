"""Sentence/paragraph-aware chunking for long LLM inputs."""

from __future__ import annotations

import re

# End punctuation followed by optional closing quotes/brackets. A period only
# counts as a boundary when followed by whitespace/end, so URLs and decimals
# are not split eagerly.
_SENTENCE_END_RE = re.compile(r"""[。！？!?；;]["'”’）)\]}】》]*|\.(?<!\.)(?=\s|$)""")
_CLAUSE_SEPARATORS = ("，", ",", "：", ":", "、", "；", ";")


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks no longer than max_chars, preserving content.

    Preference order for a boundary: paragraph, line, sentence, clause, space.
    At least half of each chunk (except the last) is filled so we do not emit
    tiny fragments. No overlap is used, which avoids duplicated words when the
    corrected chunks are joined again.
    """
    if max_chars < 16:
        raise ValueError("max_chars must be at least 16")
    normalized = (text or "").strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    remaining = normalized
    min_fill = max(1, max_chars // 2)

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = 0

        paragraph_at = window.rfind("\n\n")
        if paragraph_at >= min_fill:
            cut = paragraph_at + 2
        else:
            line_at = window.rfind("\n")
            if line_at >= min_fill:
                cut = line_at + 1
            else:
                for match in _SENTENCE_END_RE.finditer(window):
                    if match.end() >= min_fill:
                        cut = match.end()
                if cut == 0:
                    for separator in _CLAUSE_SEPARATORS:
                        separator_at = window.rfind(separator)
                        if separator_at >= min_fill:
                            cut = separator_at + 1
                            break
                if cut == 0:
                    space_at = window.rfind(" ")
                    if space_at >= min_fill:
                        cut = space_at + 1
                if cut == 0:
                    cut = max_chars

        cut = max(1, min(cut, max_chars))
        piece = remaining[:cut].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[cut:].lstrip()
        if not remaining:
            break

    if remaining:
        chunks.append(remaining.strip())
    return chunks
