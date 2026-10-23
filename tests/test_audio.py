from __future__ import annotations

import wave
from pathlib import Path

import pytest

from bv_summary.asr.faster_whisper_backend import _load_wav_float32
from bv_summary.asr.openai_backend import OpenAIASRBackend
from bv_summary.audio import extract_audio, resolve_ffmpeg
from bv_summary.errors import DependencyError

from .helpers import write_wav


def test_extract_audio_converts_to_16k_mono(tmp_path: Path) -> None:
    try:
        ffmpeg = resolve_ffmpeg()
    except DependencyError as exc:
        pytest.skip(f"ffmpeg unavailable: {exc}")

    source = write_wav(tmp_path / "source.wav", duration=0.5, sample_rate=8000)
    output = tmp_path / "out.wav"
    progress: list[tuple[float | None, str]] = []
    audio = extract_audio(
        source,
        output,
        ffmpeg=ffmpeg,
        video_duration=0.5,
        progress=lambda percent, message: progress.append((percent, message)),
    )

    assert audio.path == output
    assert audio.sample_rate == 16_000
    assert audio.channels == 1
    assert 0.4 <= audio.duration <= 0.6
    assert output.stat().st_size > 0
    assert progress[-1][0] == 100.0
    with wave.open(str(output), "rb") as wav:
        assert wav.getframerate() == 16_000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2


def test_invalid_ffmpeg_location_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(DependencyError, match="ffmpeg"):
        resolve_ffmpeg(str(tmp_path / "does-not-exist.exe"))


def test_load_wav_float32_avoids_pyav_decode(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    source = write_wav(tmp_path / "clip.wav", duration=0.25, sample_rate=16_000)
    samples = _load_wav_float32(source)
    assert samples is not None
    assert samples.dtype == np.float32
    assert samples.shape == (4_000,)
    assert 0.09 <= float(np.max(np.abs(samples))) <= 0.11

    unsupported = write_wav(tmp_path / "8k.wav", duration=0.1, sample_rate=8_000)
    assert _load_wav_float32(unsupported) is None


def test_split_wav_for_upload_creates_bounded_chunks(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "long.wav", duration=2.5, sample_rate=16_000)
    chunks = OpenAIASRBackend._split_wav(source, tmp_path / "chunks", chunk_seconds=1)
    assert len(chunks) == 3
    assert chunks[0].start == 0.0
    assert chunks[0].end == 1.0
    assert chunks[-1].end == pytest.approx(2.5, abs=0.01)
    for chunk in chunks:
        assert chunk.path.is_file()
        with wave.open(str(chunk.path), "rb") as wav:
            assert wav.getframerate() == 16_000
            assert wav.getnchannels() == 1
        assert chunk.path.stat().st_size <= 24 * 1024 * 1024
