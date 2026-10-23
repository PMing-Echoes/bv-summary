"""ASR backend selection and auto-fallback planning."""

from __future__ import annotations

import importlib.util

from ..config import ASRConfig
from ..errors import ConfigError, DependencyError
from ..utils import is_local_http_url
from .base import ASRBackend, ASRSelection
from .faster_whisper_backend import FasterWhisperBackend
from .openai_backend import OpenAIASRBackend


def _local_available() -> bool:
    try:
        return importlib.util.find_spec("faster_whisper") is not None
    except (ImportError, ValueError):
        return False


def _online_availability(config: ASRConfig) -> tuple[bool, str]:
    if config.api_key:
        return True, ""
    if config.api_base_url and is_local_http_url(config.api_base_url):
        return True, ""
    if config.api_base_url:
        return False, "ASR_BASE_URL 已配置，但缺少 ASR_API_KEY / OPENAI_API_KEY。"
    return False, "未配置在线 ASR API Key。"


def _make_local(config: ASRConfig) -> ASRBackend:
    if not _local_available():
        raise DependencyError(
            "未安装 faster-whisper，无法使用本地语音识别。",
            hint='请执行：pip install -e ".[local-asr]"（首次使用模型时还需联网下载模型）',
        )
    return FasterWhisperBackend(config)


def _make_online(config: ASRConfig) -> ASRBackend:
    available, reason = _online_availability(config)
    if not available:
        raise ConfigError(
            f"在线 ASR 不可用：{reason}",
            hint="请设置 ASR_API_KEY / OPENAI_API_KEY，或把 ASR_BASE_URL 指向可无 Key 访问的本机服务。",
        )
    return OpenAIASRBackend(config)


def build_asr_selection(config: ASRConfig) -> ASRSelection:
    """Return ordered backends. Auto may return [local, online] for fallback."""
    if config.backend == "faster-whisper":
        return ASRSelection((_make_local(config),))
    if config.backend == "openai":
        return ASRSelection((_make_online(config),))

    local_ok = _local_available()
    online_ok, online_reason = _online_availability(config)
    backends: list[ASRBackend] = []
    notes: list[str] = []

    if local_ok:
        backends.append(FasterWhisperBackend(config))
    else:
        notes.append("ASR auto：未安装 faster-whisper，跳过本地后端。")

    if online_ok:
        backends.append(OpenAIASRBackend(config))
    elif local_ok:
        notes.append(f"ASR auto：在线回退不可用（{online_reason}）。")

    if not backends:
        detail = "\n".join(f"- {note}" for note in notes) if notes else "- 未检测到可用后端。"
        raise ConfigError(
            "ASR auto 找不到可用后端：本地 faster-whisper 未安装，在线 ASR 也未配置。\n" + detail,
            hint=(
                '本地模式：pip install -e ".[local-asr]"；'
                "在线模式：设置 ASR_API_KEY / OPENAI_API_KEY（可再设置 ASR_BASE_URL、ASR_API_MODEL）。"
            ),
        )
    if len(backends) > 1:
        notes.append("ASR auto：优先本地 faster-whisper，失败后自动回退在线 ASR。")
    return ASRSelection(tuple(backends), tuple(notes))
