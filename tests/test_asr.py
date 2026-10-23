from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bv_summary.asr import factory
from bv_summary.asr.factory import build_asr_selection
from bv_summary.asr.faster_whisper_backend import FasterWhisperBackend
from bv_summary.asr.openai_backend import OpenAIASRBackend
from bv_summary.errors import ASRError, ConfigError, DependencyError
from bv_summary.models import AudioFile

from .helpers import make_asr_config, write_wav


def _audio_file(path: Path, duration: float = 1.0) -> AudioFile:
    return AudioFile(
        path=path,
        duration=duration,
        sample_rate=16_000,
        channels=1,
        size_bytes=path.stat().st_size,
    )


def test_auto_prefers_local_then_online(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "_local_available", lambda: True)
    selection = build_asr_selection(make_asr_config(api_key="secret"))
    assert [backend.name for backend in selection.backends] == ["faster-whisper", "openai"]
    assert any("回退在线" in note for note in selection.notes)


def test_auto_uses_online_when_local_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "_local_available", lambda: False)
    selection = build_asr_selection(make_asr_config(api_key="secret"))
    assert [backend.name for backend in selection.backends] == ["openai"]
    assert any("未安装" in note for note in selection.notes)


def test_auto_without_any_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "_local_available", lambda: False)
    with pytest.raises(ConfigError, match="ASR"):
        build_asr_selection(make_asr_config())


def test_explicit_local_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "_local_available", lambda: False)
    with pytest.raises(DependencyError, match="faster-whisper"):
        build_asr_selection(make_asr_config(backend="faster-whisper"))


def test_explicit_online_without_key_raises() -> None:
    with pytest.raises(ConfigError, match="API Key"):
        build_asr_selection(make_asr_config(backend="openai"))


def test_explicit_online_with_local_keyless_endpoint_is_allowed() -> None:
    selection = build_asr_selection(
        make_asr_config(backend="openai", api_base_url="http://127.0.0.1:8000/v1")
    )
    assert selection.backends[0].name == "openai"


def test_online_backend_transcribes_and_joins_chunks(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "test.wav", duration=1.0)
    calls: list[dict[str, Any]] = []

    class FakeTranscriptions:
        def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(text=" 你好，世界。 ", language="zh")

    fake_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=FakeTranscriptions()))
    backend = OpenAIASRBackend(
        make_asr_config(backend="openai", api_key="secret", language="zh"),
        client=fake_client,
    )
    progress: list[tuple[float | None, str]] = []
    transcript = backend.transcribe(
        _audio_file(source), progress=lambda percent, message: progress.append((percent, message))
    )

    assert transcript.text == "你好，世界。"
    assert transcript.language == "zh"
    assert transcript.backend == "openai"
    assert transcript.model == "whisper-1"
    assert len(calls) == 1
    assert calls[0]["model"] == "whisper-1"
    assert calls[0]["language"] == "zh"
    assert calls[0]["response_format"] == "json"
    assert progress[-1][0] == 100.0


def test_online_backend_missing_key_fails_before_request() -> None:
    with pytest.raises(ConfigError, match="API Key"):
        OpenAIASRBackend(make_asr_config(backend="openai"))


def test_local_device_auto_respects_supported_cuda_compute_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCTranslate2:
        @staticmethod
        def get_cuda_device_count() -> int:
            return 1

        @staticmethod
        def get_supported_compute_types(device: str) -> set[str]:
            if device == "cuda":
                return {"int8", "float32", "int8_float32"}
            return {"int8", "float32"}

    monkeypatch.setitem(sys.modules, "ctranslate2", FakeCTranslate2)
    backend = FasterWhisperBackend(make_asr_config(device="auto", compute_type="auto"))
    assert backend._resolve_device() == ("cuda", "int8_float32")


def test_online_backend_maps_auth_error_to_readable_error(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "test.wav", duration=0.5)

    class AuthenticationError(Exception):
        status_code = 401

    class FakeTranscriptions:
        def create(self, **kwargs: Any) -> Any:
            raise AuthenticationError("Incorrect API key provided")

    fake_client = SimpleNamespace(audio=SimpleNamespace(transcriptions=FakeTranscriptions()))
    backend = OpenAIASRBackend(
        make_asr_config(backend="openai", api_key="bad"),
        client=fake_client,
    )
    with pytest.raises(ASRError) as caught:
        backend.transcribe(_audio_file(source))
    assert "认证失败" in caught.value.message
    assert "ASR_API_KEY" in (caught.value.hint or "")
