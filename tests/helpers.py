from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Any

from bv_summary.cli import build_parser
from bv_summary.config import ASRConfig, DownloadConfig, LLMConfig, Settings


def make_asr_config(**overrides: Any) -> ASRConfig:
    values: dict[str, Any] = {
        "backend": "auto",
        "model": "small",
        "device": "auto",
        "compute_type": "auto",
        "language": None,
        "model_dir": None,
        "api_base_url": None,
        "api_key": None,
        "api_model": "whisper-1",
        "chunk_seconds": 600,
        "timeout": 600.0,
        "cpu_threads": 0,
        "beam_size": 5,
    }
    values.update(overrides)
    return ASRConfig(**values)


def make_llm_config(**overrides: Any) -> LLMConfig:
    values: dict[str, Any] = {
        "backend": "auto",
        "model": None,
        "base_url": None,
        "api_key": None,
        "ollama_base_url": "http://127.0.0.1:11434/v1",
        "ollama_model": None,
        "temperature": 0.1,
        "chunk_chars": 3000,
        "timeout": 120.0,
        "prompt_file": None,
    }
    values.update(overrides)
    return LLMConfig(**values)


def make_settings(
    tmp_path: Path,
    *,
    asr: ASRConfig | None = None,
    llm: LLMConfig | None = None,
    download: DownloadConfig | None = None,
    output: Path | None = None,
    output_is_dir: bool = True,
    keep_audio: bool = False,
    save_raw: bool = False,
    force: bool = False,
) -> Settings:
    return Settings(
        output=output or (tmp_path / "out"),
        output_is_dir=output_is_dir,
        keep_audio=keep_audio,
        save_raw=save_raw,
        force=force,
        download=download or DownloadConfig(None, None, None),
        asr=asr or make_asr_config(),
        llm=llm or make_llm_config(backend="none"),
    )


def parse_cli_args(tmp_path: Path, *extra: str):
    config_path = tmp_path / "test-config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text("", encoding="utf-8")
    return build_parser().parse_args(
        ["--config", str(config_path), "https://example.com/watch?v=test", *extra]
    )


def clean_env(tmp_path: Path) -> dict[str, str]:
    return {
        "APPDATA": str(tmp_path / "appdata"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
        "LOCALAPPDATA": str(tmp_path / "localappdata"),
    }


def write_wav(
    path: Path,
    *,
    duration: float = 1.0,
    sample_rate: int = 16_000,
    channels: int = 1,
    frequency: float = 440.0,
) -> Path:
    frame_count = max(1, int(duration * sample_rate))
    frames = bytearray()
    for index in range(frame_count):
        value = int(32767 * 0.1 * math.sin(2 * math.pi * frequency * index / sample_rate))
        packed = struct.pack("<h", value)
        frames.extend(packed * channels)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))
    return path
