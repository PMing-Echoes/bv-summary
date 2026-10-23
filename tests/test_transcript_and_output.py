from __future__ import annotations

import re
from pathlib import Path

import pytest

from bv_summary.llm.text_chunks import chunk_text
from bv_summary.models import Segment
from bv_summary.output import (
    audio_sidecar,
    plan_output_path,
    raw_sidecar,
    save_text,
)
from bv_summary.transcript import join_asr_pieces, segments_to_text
from bv_summary.utils import (
    clean_error_message,
    is_local_http_url,
    sanitize_filename,
    unique_path,
    validate_http_url,
)

from .helpers import make_settings


def _without_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def test_segments_join_cjk_without_spaces() -> None:
    segments = [
        Segment(0.0, 1.0, "你好，"),
        Segment(1.0, 2.0, "世界。"),
        Segment(2.2, 3.0, "这是测试。"),
    ]
    assert segments_to_text(segments) == "你好，世界。这是测试。"


def test_segments_add_space_for_latin() -> None:
    segments = [
        Segment(0.0, 1.0, "Hello"),
        Segment(1.0, 2.0, "world."),
        Segment(2.0, 3.0, "Next sentence."),
    ]
    assert segments_to_text(segments) == "Hello world. Next sentence."


def test_segments_paragraph_on_sentence_and_pause() -> None:
    segments = [
        Segment(0.0, 1.0, "第一句。"),
        Segment(2.6, 3.5, "第二句。"),
    ]
    assert segments_to_text(segments) == "第一句。\n\n第二句。"


def test_join_pieces_handles_punctuation() -> None:
    assert join_asr_pieces("hello", "world") == "hello world"
    assert join_asr_pieces("你好", "世界") == "你好世界"
    assert join_asr_pieces("end.", "Next") == "end. Next"


def test_chunk_text_respects_limit_and_content() -> None:
    text = ("This is a test sentence. " * 80).strip()
    chunks = chunk_text(text, 120)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert _without_whitespace("".join(chunks)) == _without_whitespace(text)
    assert any(chunk.endswith(".") for chunk in chunks[:-1])


def test_chunk_text_chinese_without_spaces() -> None:
    text = "这是一个没有空格的句子，" * 100
    chunks = chunk_text(text, 80)
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert _without_whitespace("".join(chunks)) == _without_whitespace(text)


def test_chunk_text_empty_and_short() -> None:
    assert chunk_text("", 100) == []
    assert chunk_text("  short  ", 100) == ["short"]
    with pytest.raises(ValueError):
        chunk_text("abc", 4)


def test_sanitize_filename_is_windows_safe() -> None:
    sanitized = sanitize_filename('a<b>:"/\\|?*b')
    assert sanitized.startswith("a")
    assert sanitized.endswith("b")
    assert not any(character in '<>:"/\\|?*' for character in sanitized)
    assert sanitize_filename("CON") == "_CON"
    assert sanitize_filename("   ") == "transcript"
    assert len(sanitize_filename("x" * 500)) <= 100


def test_unique_path_avoids_existing_files(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    target.write_text("one", encoding="utf-8")
    assert unique_path(target).name == "result (1).txt"
    assert unique_path(target, force=True) == target


def test_save_text_is_utf8_and_atomic(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    save_text(target, "第一行\n第二行")
    assert target.read_text(encoding="utf-8") == "第一行\n第二行\n"


def test_output_sidecar_names() -> None:
    assert raw_sidecar(Path("episode.txt")) == Path("episode.raw.txt")
    assert audio_sidecar(Path("episode.txt")) == Path("episode.wav")
    assert audio_sidecar(Path("episode.wav")) == Path("episode.audio.wav")


def test_plan_output_path_uses_title_and_does_not_overwrite(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    chosen, requested = plan_output_path(settings, "A/B: title?")
    assert chosen.name == "A_B_ title_.txt"
    assert requested == chosen
    chosen.write_text("existing", encoding="utf-8")
    chosen2, _ = plan_output_path(settings, "A/B: title?")
    assert chosen2.name == "A_B_ title_ (1).txt"


def test_plan_output_path_file_and_directory(tmp_path: Path) -> None:
    target = tmp_path / "custom.txt"
    settings = make_settings(tmp_path, output=target, output_is_dir=False)
    chosen, _ = plan_output_path(settings, "ignored")
    assert chosen == target.resolve()

    out_dir = tmp_path / "sub" / "out"
    settings = make_settings(tmp_path, output=out_dir, output_is_dir=True)
    chosen, requested = plan_output_path(settings, "Video")
    assert chosen == (out_dir / "Video.txt").resolve()
    assert requested == chosen


def test_validate_http_url() -> None:
    assert validate_http_url("https://example.com/a") == "https://example.com/a"
    with pytest.raises(ValueError):
        validate_http_url("file:///tmp/video")
    with pytest.raises(ValueError):
        validate_http_url("not a URL")


def test_local_url_detection() -> None:
    assert is_local_http_url("http://localhost:11434/v1")
    assert is_local_http_url("http://127.0.0.1:8000/v1")
    assert is_local_http_url("http://192.168.1.2:8000/v1")
    assert not is_local_http_url("https://api.openai.com/v1")
    assert not is_local_http_url(None)


def test_clean_error_message() -> None:
    value = "\x1b[31mERROR: something\x1b[0m\r\nsecond line\r\n"
    assert clean_error_message(value) == "something\nsecond line"
