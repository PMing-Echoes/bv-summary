from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bv_summary import pipeline as pipeline_module
from bv_summary.asr.base import ASRSelection
from bv_summary.errors import ASRError, LLMError
from bv_summary.llm.base import LLMSelection
from bv_summary.models import AudioFile, Correction, MediaFile, Segment, Transcript, VideoInfo
from bv_summary.pipeline import Pipeline
from bv_summary.ui import Reporter

from .helpers import make_asr_config, make_llm_config, make_settings, write_wav


class FailingASR:
    name = "failing-asr"
    model = "nope"

    def transcribe(self, audio: AudioFile, *, progress=None) -> Transcript:
        raise ASRError("first ASR failed")


class WorkingASR:
    name = "working-asr"
    model = "good"

    def transcribe(self, audio: AudioFile, *, progress=None) -> Transcript:
        if progress:
            progress(100.0, "done")
        return Transcript(
            text="原始转写。",
            segments=(Segment(0.0, 1.0, "原始转写。"),),
            language="zh",
            backend=self.name,
            model=self.model,
        )


class FailingLLM:
    name = "failing-llm"
    model = "nope"
    base_url = None

    def correct(self, text: str, *, progress=None) -> Correction:
        raise LLMError("first LLM failed")


class WorkingLLM:
    name = "working-llm"
    model = "good"
    base_url = None

    def correct(self, text: str, *, progress=None) -> Correction:
        if progress:
            progress(100.0, "done")
        return Correction(text="纠正后的文本。", backend=self.name, model=self.model, chunks=2)


def _fake_externals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    llm_selection: LLMSelection,
    asr_selection: ASRSelection,
    source_audio: Path,
) -> None:
    def fake_download(url: str, work_dir: Path, **kwargs: Any) -> MediaFile:
        work_dir.mkdir(parents=True, exist_ok=True)
        media = work_dir / "media.m4a"
        media.write_bytes(b"fake")
        progress = kwargs.get("progress")
        if progress:
            progress(100.0, "downloaded")
        return MediaFile(
            path=media,
            info=VideoInfo(
                title="Test video",
                video_id="id",
                webpage_url=url,
                duration=2.0,
                uploader="tester",
                extractor="Fake",
            ),
        )

    def fake_extract(
        input_path: Path,
        output_path: Path,
        *,
        ffmpeg: Path,
        video_duration: float | None = None,
        progress=None,
    ) -> AudioFile:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(source_audio.read_bytes())
        if progress:
            progress(100.0, "extracted")
        return AudioFile(
            path=output_path,
            duration=2.0,
            sample_rate=16_000,
            channels=1,
            size_bytes=output_path.stat().st_size,
        )

    monkeypatch.setattr(pipeline_module, "resolve_ffmpeg", lambda location: tmp_path / "ffmpeg")
    monkeypatch.setattr(pipeline_module, "download_media", fake_download)
    monkeypatch.setattr(pipeline_module, "extract_audio", fake_extract)
    monkeypatch.setattr(pipeline_module, "build_asr_selection", lambda config: asr_selection)
    monkeypatch.setattr(pipeline_module, "build_llm_selection", lambda config: llm_selection)


def test_pipeline_success_with_fallbacks_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_audio = write_wav(tmp_path / "source.wav", duration=2.0)
    settings = make_settings(
        tmp_path,
        asr=make_asr_config(),
        llm=make_llm_config(backend="openai", model="fake", api_key="fake"),
        save_raw=True,
        keep_audio=True,
    )
    _fake_externals(
        monkeypatch,
        tmp_path,
        llm_selection=LLMSelection((FailingLLM(), WorkingLLM()), ("fallback note",)),
        asr_selection=ASRSelection((FailingASR(), WorkingASR())),
        source_audio=source_audio,
    )
    result = Pipeline(settings, Reporter(quiet=True)).run("https://example.com/video")

    expected_output = tmp_path / "out" / "Test video.txt"
    expected_raw = tmp_path / "out" / "Test video.raw.txt"
    expected_audio = tmp_path / "out" / "Test video.wav"
    assert result.text == "纠正后的文本。"
    assert result.text_corrected is True
    assert result.asr_backend == "working-asr"
    assert result.llm_backend == "working-llm"
    assert result.output_path == expected_output
    assert expected_output.read_text(encoding="utf-8") == "纠正后的文本。\n"
    assert result.raw_path == expected_raw
    assert expected_raw.read_text(encoding="utf-8") == "原始转写。\n"
    assert result.audio_path == expected_audio
    assert expected_audio.read_bytes() == source_audio.read_bytes()


def test_pipeline_llm_failure_keeps_raw_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_audio = write_wav(tmp_path / "source.wav", duration=1.0)
    settings = make_settings(
        tmp_path,
        llm=make_llm_config(backend="openai", model="fake", api_key="fake"),
        save_raw=False,
    )
    _fake_externals(
        monkeypatch,
        tmp_path,
        llm_selection=LLMSelection((FailingLLM(),)),
        asr_selection=ASRSelection((WorkingASR(),)),
        source_audio=source_audio,
    )
    with pytest.raises(LLMError):
        Pipeline(settings, Reporter(quiet=True)).run("https://example.com/video")

    raw_files = list((tmp_path / "out").glob("*.raw.txt"))
    assert len(raw_files) == 1
    assert raw_files[0].read_text(encoding="utf-8") == "原始转写。\n"
    assert not (tmp_path / "out" / "Test video.txt").exists()


def test_pipeline_without_llm_writes_raw_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_audio = write_wav(tmp_path / "source.wav", duration=1.0)
    settings = make_settings(tmp_path, llm=make_llm_config(backend="none"))
    _fake_externals(
        monkeypatch,
        tmp_path,
        llm_selection=LLMSelection(()),
        asr_selection=ASRSelection((WorkingASR(),)),
        source_audio=source_audio,
    )
    result = Pipeline(settings, Reporter(quiet=True)).run("https://example.com/video")
    assert result.text_corrected is False
    assert result.text == "原始转写。"
    assert result.raw_path is None
    assert result.output_path.read_text(encoding="utf-8") == "原始转写。\n"
