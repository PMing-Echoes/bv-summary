"""Configuration loading: CLI > environment > TOML > built-in defaults."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError, DependencyError

ASR_BACKEND_CHOICES = ("auto", "faster-whisper", "openai")
LLM_BACKEND_CHOICES = ("auto", "openai", "ollama", "none")

_ASR_BACKEND_ALIASES = {
    "auto": "auto",
    "faster-whisper": "faster-whisper",
    "fasterwhisper": "faster-whisper",
    "faster_whisper": "faster-whisper",
    "local": "faster-whisper",
    "openai": "openai",
    "openai-api": "openai",
    "openai_whisper": "openai",
    "api": "openai",
}
_LLM_BACKEND_ALIASES = {
    "auto": "auto",
    "openai": "openai",
    "openai-compatible": "openai",
    "openai_compatible": "openai",
    "api": "openai",
    "ollama": "ollama",
    "local": "ollama",
    "none": "none",
    "off": "none",
    "disabled": "none",
}

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}
_MISSING = object()

_ALLOWED_TOP_LEVEL = {"output", "keep_audio", "save_raw", "force", "download", "asr", "llm"}
_ALLOWED_SECTIONS: dict[str, set[str]] = {
    "download": {"cookies", "cookies_from_browser", "ffmpeg_location"},
    "asr": {
        "backend",
        "model",
        "device",
        "compute_type",
        "language",
        "model_dir",
        "cpu_threads",
        "beam_size",
        "api_base_url",
        "api_key",
        "api_model",
        "chunk_seconds",
        "timeout",
    },
    "llm": {
        "backend",
        "model",
        "base_url",
        "api_key",
        "temperature",
        "chunk_chars",
        "timeout",
        "prompt_file",
        "ollama_base_url",
        "ollama_model",
    },
}


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    cookies: Path | None
    cookies_from_browser: str | None
    ffmpeg_location: str | None


@dataclass(frozen=True, slots=True)
class ASRConfig:
    backend: str
    model: str
    device: str
    compute_type: str
    language: str | None
    model_dir: Path | None
    api_base_url: str | None
    api_key: str | None = field(repr=False)
    api_model: str
    chunk_seconds: int
    timeout: float
    cpu_threads: int
    beam_size: int


@dataclass(frozen=True, slots=True)
class LLMConfig:
    backend: str
    model: str | None
    base_url: str | None
    api_key: str | None = field(repr=False)
    ollama_base_url: str
    ollama_model: str | None
    temperature: float
    chunk_chars: int
    timeout: float
    prompt_file: Path | None


@dataclass(frozen=True, slots=True)
class Settings:
    output: Path | None
    output_is_dir: bool
    keep_audio: bool
    save_raw: bool
    force: bool
    download: DownloadConfig
    asr: ASRConfig
    llm: LLMConfig


class _Resolver:
    """Resolve one setting while keeping the winning source for error messages."""

    def __init__(
        self,
        args: Any,
        environ: Mapping[str, str],
        data: Mapping[str, Any],
        config_path: Path | None,
    ) -> None:
        self.args = args
        self.environ = environ
        self.data = data
        self.config_path = config_path

    def _file_value(self, section: str | None, key: str) -> tuple[Any, str | None]:
        if section is None:
            if key in self.data:
                return self.data[key], self._file_source
            return _MISSING, None
        section_data = self.data.get(section)
        if isinstance(section_data, dict) and key in section_data:
            return section_data[key], self._file_source
        return _MISSING, None

    @property
    def _file_source(self) -> str:
        return f"配置文件 {self.config_path}"

    def _pick(
        self,
        *,
        cli_attr: str | None,
        env_names: tuple[str, ...],
        section: str | None,
        key: str,
    ) -> tuple[Any, str | None]:
        if cli_attr is not None:
            cli_value = getattr(self.args, cli_attr, None)
            if cli_value is not None:
                return cli_value, "命令行参数"
        for env_name in env_names:
            env_value = self.environ.get(env_name)
            if env_value is not None and str(env_value).strip() != "":
                return env_value, f"环境变量 {env_name}"
        value, source = self._file_value(section, key)
        if value is not _MISSING:
            return value, source
        return None, None

    def _error(self, label: str, source: str | None, message: str) -> ConfigError:
        origin = f"（来源：{source}）" if source else "（使用默认值）"
        return ConfigError(f"{label}配置无效{origin}：{message}")

    def text(
        self,
        *,
        cli_attr: str | None,
        env_names: tuple[str, ...],
        section: str | None,
        key: str,
        label: str,
        default: str = "",
        allow_empty: bool = False,
    ) -> str:
        raw, source = self._pick(cli_attr=cli_attr, env_names=env_names, section=section, key=key)
        if raw is None:
            value = default
        elif isinstance(raw, str):
            value = raw.strip()
        else:
            raise self._error(label, source, "必须是字符串")
        if not allow_empty and not value:
            raise self._error(label, source, "不能为空")
        return value

    def optional_text(
        self,
        *,
        cli_attr: str | None,
        env_names: tuple[str, ...],
        section: str | None,
        key: str,
        label: str,
    ) -> str | None:
        value = self.text(
            cli_attr=cli_attr,
            env_names=env_names,
            section=section,
            key=key,
            label=label,
            default="",
            allow_empty=True,
        )
        return value or None

    def path(
        self,
        *,
        cli_attr: str | None,
        env_names: tuple[str, ...],
        section: str | None,
        key: str,
        label: str,
    ) -> Path | None:
        value = self.optional_text(
            cli_attr=cli_attr, env_names=env_names, section=section, key=key, label=label
        )
        if value is None:
            return None
        try:
            return Path(value).expanduser()
        except (OSError, ValueError) as exc:
            raise self._error(label, None, f"路径无效：{exc}") from exc

    def boolean(
        self,
        *,
        cli_attr: str | None,
        env_names: tuple[str, ...],
        section: str | None,
        key: str,
        label: str,
        default: bool,
    ) -> bool:
        raw, source = self._pick(cli_attr=cli_attr, env_names=env_names, section=section, key=key)
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in _TRUE:
                return True
            if normalized in _FALSE:
                return False
            raise self._error(label, source, "应为 true/false、yes/no 或 1/0")
        raise self._error(label, source, "必须是布尔值")

    def integer(
        self,
        *,
        cli_attr: str | None,
        env_names: tuple[str, ...],
        section: str | None,
        key: str,
        label: str,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        raw, source = self._pick(cli_attr=cli_attr, env_names=env_names, section=section, key=key)
        if raw is None:
            return default
        if isinstance(raw, bool):
            raise self._error(label, source, "必须是整数")
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise self._error(label, source, "必须是整数") from exc
        if not minimum <= value <= maximum:
            raise self._error(label, source, f"必须在 {minimum} 到 {maximum} 之间")
        return value

    def number(
        self,
        *,
        cli_attr: str | None,
        env_names: tuple[str, ...],
        section: str | None,
        key: str,
        label: str,
        default: float,
        minimum: float,
        maximum: float,
    ) -> float:
        raw, source = self._pick(cli_attr=cli_attr, env_names=env_names, section=section, key=key)
        if raw is None:
            return default
        if isinstance(raw, bool):
            raise self._error(label, source, "必须是数字")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise self._error(label, source, "必须是数字") from exc
        if not minimum <= value <= maximum:
            raise self._error(label, source, f"必须在 {minimum} 到 {maximum} 之间")
        return value


def load_environment(env_file: str | None = None) -> Path | None:
    """Load .env without overriding variables already present in the process."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - exercised only on broken installs
        raise DependencyError(
            "缺少 python-dotenv，无法加载 .env。",
            hint='请执行：pip install -e ".[local-asr]"',
        ) from exc

    path = Path(env_file).expanduser() if env_file else Path.cwd() / ".env"
    if not path.exists():
        if env_file:
            raise ConfigError(f"指定的 .env 文件不存在：{path}")
        return None
    if not path.is_file():
        raise ConfigError(f".env 路径不是文件：{path}")
    try:
        load_dotenv(dotenv_path=path, override=False, encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise ConfigError(f"读取 .env 失败：{path}（{exc}）") from exc
    return path


def _user_config_candidates(environ: Mapping[str, str]) -> list[Path]:
    candidates: list[Path] = []
    appdata = environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "bv-summary" / "config.toml")
    xdg = environ.get("XDG_CONFIG_HOME")
    if xdg:
        candidates.append(Path(xdg) / "bv-summary" / "config.toml")
    try:
        candidates.append(Path.home() / ".config" / "bv-summary" / "config.toml")
    except (OSError, RuntimeError):
        # Some service/container environments do not expose a home directory.
        pass
    return candidates


def _select_config_path(args: Any, environ: Mapping[str, str]) -> Path | None:
    explicit = getattr(args, "config", None)
    env_config = environ.get("BV_SUMMARY_CONFIG", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"指定的配置文件不存在或不是文件：{path}")
        return path
    if env_config:
        path = Path(env_config).expanduser()
        if not path.is_file():
            raise ConfigError(f"BV_SUMMARY_CONFIG 指向的文件不存在：{path}")
        return path

    local = Path.cwd() / "config.toml"
    if local.is_file():
        return local
    for candidate in _user_config_candidates(environ):
        if candidate.is_file():
            return candidate
    return None


def _read_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件：{path}（{exc}）") from exc
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"配置文件必须使用 UTF-8 编码：{path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"配置文件 TOML 语法错误：{path}\n{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是 TOML 表：{path}")

    unknown = set(data) - _ALLOWED_TOP_LEVEL
    if unknown:
        raise ConfigError(f"配置文件包含未知顶层项：{', '.join(sorted(unknown))}（{path}）")
    for section_name, allowed in _ALLOWED_SECTIONS.items():
        section = data.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ConfigError(f"配置项 [{section_name}] 必须是 TOML 表：{path}")
        unknown_keys = set(section) - allowed
        if unknown_keys:
            raise ConfigError(
                f"配置项 [{section_name}] 包含未知字段：{', '.join(sorted(unknown_keys))}（{path}）"
            )
    return data


def _normalize_backend(
    value: str, aliases: dict[str, str], label: str, choices: tuple[str, ...]
) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ConfigError(f"{label}“{value}”无效；可选值：{', '.join(choices)}") from exc


def _validate_http_base_url(value: str | None, label: str) -> str | None:
    if not value:
        return None
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ConfigError(f"{label}必须以 http:// 或 https:// 开头：{value}")
    return value.rstrip("/")


def _normalize_language(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("_", "-")
    if not normalized or normalized == "auto":
        return None
    # Whisper/OpenAI expect ISO 639-1 codes such as "zh"/"en", not locales
    # such as "zh-CN"; taking the primary subtag is the forgiving choice.
    return normalized.split("-", 1)[0]


def _validate_device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ConfigError(f"ASR 设备“{value}”无效；可选：auto、cpu、cuda")
    return normalized


def _validate_compute_type(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {
        "auto",
        "default",
        "int8",
        "int8_float16",
        "int8_float32",
        "int8_bfloat16",
        "float16",
        "bfloat16",
        "float32",
    }
    if normalized not in allowed:
        raise ConfigError(f"ASR 计算类型“{value}”无效；常用值：auto、int8、float16、float32")
    return normalized


def load_settings(args: Any, environ: Mapping[str, str] | None = None) -> Settings:
    """Build validated settings from parsed CLI args plus env and TOML.

    The caller is expected to call :func:`load_environment` first so values from
    .env are present in ``os.environ``. Tests can pass an explicit mapping.
    """
    env = dict(os.environ if environ is None else environ)
    config_path = _select_config_path(args, env)
    data = _read_config(config_path)
    r = _Resolver(args, env, data, config_path)

    output_raw = r.optional_text(
        cli_attr="output",
        env_names=("BV_SUMMARY_OUTPUT",),
        section=None,
        key="output",
        label="输出路径",
    )
    output: Path | None = None
    output_is_dir = True
    if output_raw:
        try:
            output = Path(output_raw).expanduser()
        except (OSError, ValueError) as exc:
            raise ConfigError(f"输出路径无效：{output_raw}（{exc}）") from exc
        output_is_dir = output_raw.endswith(("/", "\\")) or output.is_dir() or output.suffix == ""

    keep_audio = r.boolean(
        cli_attr="keep_audio",
        env_names=("BV_SUMMARY_KEEP_AUDIO",),
        section=None,
        key="keep_audio",
        label="保留音频",
        default=False,
    )
    save_raw = r.boolean(
        cli_attr="save_raw",
        env_names=("BV_SUMMARY_SAVE_RAW",),
        section=None,
        key="save_raw",
        label="保存原始转写",
        default=False,
    )
    force = r.boolean(
        cli_attr="force",
        env_names=("BV_SUMMARY_FORCE",),
        section=None,
        key="force",
        label="覆盖已有文件",
        default=False,
    )

    cookies = r.path(
        cli_attr="cookies",
        env_names=("YTDLP_COOKIES",),
        section="download",
        key="cookies",
        label="cookies 文件",
    )
    if cookies is not None and not cookies.is_file():
        raise ConfigError(f"cookies 文件不存在或不是文件：{cookies}")
    cookies_from_browser = r.optional_text(
        cli_attr="cookies_from_browser",
        env_names=("YTDLP_COOKIES_FROM_BROWSER",),
        section="download",
        key="cookies_from_browser",
        label="浏览器 cookies 来源",
    )
    ffmpeg_location = r.optional_text(
        cli_attr="ffmpeg_location",
        env_names=("FFMPEG_BINARY", "FFMPEG_LOCATION"),
        section="download",
        key="ffmpeg_location",
        label="ffmpeg 路径",
    )

    asr_backend = _normalize_backend(
        r.text(
            cli_attr="asr_backend",
            env_names=("ASR_BACKEND",),
            section="asr",
            key="backend",
            label="ASR 后端",
            default="auto",
        ),
        _ASR_BACKEND_ALIASES,
        "ASR 后端",
        ASR_BACKEND_CHOICES,
    )
    asr_model = r.text(
        cli_attr="asr_model",
        env_names=("ASR_MODEL",),
        section="asr",
        key="model",
        label="ASR 模型",
        default="small",
    )
    asr_device = _validate_device(
        r.text(
            cli_attr="asr_device",
            env_names=("ASR_DEVICE",),
            section="asr",
            key="device",
            label="ASR 设备",
            default="auto",
        )
    )
    asr_compute_type = _validate_compute_type(
        r.text(
            cli_attr="asr_compute_type",
            env_names=("ASR_COMPUTE_TYPE",),
            section="asr",
            key="compute_type",
            label="ASR 计算类型",
            default="auto",
        )
    )
    asr_language = _normalize_language(
        r.optional_text(
            cli_attr="language",
            env_names=("ASR_LANGUAGE",),
            section="asr",
            key="language",
            label="ASR 语言",
        )
    )
    asr_model_dir = r.path(
        cli_attr="asr_model_dir",
        env_names=("ASR_MODEL_DIR",),
        section="asr",
        key="model_dir",
        label="ASR 模型目录",
    )
    asr_api_base_url = _validate_http_base_url(
        r.optional_text(
            cli_attr="asr_base_url",
            env_names=("ASR_BASE_URL",),
            section="asr",
            key="api_base_url",
            label="ASR API 地址",
        ),
        "ASR API 地址",
    )
    asr_api_key = r.optional_text(
        cli_attr=None,
        env_names=("ASR_API_KEY", "OPENAI_API_KEY"),
        section="asr",
        key="api_key",
        label="ASR API Key",
    )
    asr_api_model = r.text(
        cli_attr="asr_api_model",
        env_names=("ASR_API_MODEL",),
        section="asr",
        key="api_model",
        label="ASR API 模型",
        default="whisper-1",
    )
    asr_chunk_seconds = r.integer(
        cli_attr=None,
        env_names=("ASR_CHUNK_SECONDS",),
        section="asr",
        key="chunk_seconds",
        label="ASR 在线分块秒数",
        default=600,
        minimum=30,
        maximum=1800,
    )
    asr_timeout = r.number(
        cli_attr=None,
        env_names=("ASR_TIMEOUT",),
        section="asr",
        key="timeout",
        label="ASR 超时（秒）",
        default=600.0,
        minimum=5.0,
        maximum=7200.0,
    )
    asr_cpu_threads = r.integer(
        cli_attr=None,
        env_names=("ASR_CPU_THREADS",),
        section="asr",
        key="cpu_threads",
        label="ASR CPU 线程数",
        default=0,
        minimum=0,
        maximum=256,
    )
    asr_beam_size = r.integer(
        cli_attr=None,
        env_names=("ASR_BEAM_SIZE",),
        section="asr",
        key="beam_size",
        label="ASR beam_size",
        default=5,
        minimum=1,
        maximum=20,
    )

    llm_backend = _normalize_backend(
        r.text(
            cli_attr="llm_backend",
            env_names=("LLM_BACKEND",),
            section="llm",
            key="backend",
            label="LLM 后端",
            default="auto",
        ),
        _LLM_BACKEND_ALIASES,
        "LLM 后端",
        LLM_BACKEND_CHOICES,
    )
    llm_model = r.optional_text(
        cli_attr="llm_model",
        env_names=("LLM_MODEL",),
        section="llm",
        key="model",
        label="LLM 模型",
    )
    llm_base_url = _validate_http_base_url(
        r.optional_text(
            cli_attr="base_url",
            env_names=("LLM_BASE_URL",),
            section="llm",
            key="base_url",
            label="LLM API 地址",
        ),
        "LLM API 地址",
    )
    llm_api_key = r.optional_text(
        cli_attr=None,
        env_names=("LLM_API_KEY", "OPENAI_API_KEY"),
        section="llm",
        key="api_key",
        label="LLM API Key",
    )
    ollama_base_url = _validate_http_base_url(
        r.text(
            cli_attr=None,
            env_names=("OLLAMA_BASE_URL",),
            section="llm",
            key="ollama_base_url",
            label="Ollama 地址",
            default="http://127.0.0.1:11434/v1",
        ),
        "Ollama 地址",
    )
    assert ollama_base_url is not None
    llm_ollama_model = r.optional_text(
        cli_attr=None,
        env_names=("OLLAMA_MODEL",),
        section="llm",
        key="ollama_model",
        label="Ollama 模型",
    )
    llm_temperature = r.number(
        cli_attr=None,
        env_names=("LLM_TEMPERATURE",),
        section="llm",
        key="temperature",
        label="LLM temperature",
        default=0.1,
        minimum=0.0,
        maximum=2.0,
    )
    llm_chunk_chars = r.integer(
        cli_attr=None,
        env_names=("LLM_CHUNK_CHARS",),
        section="llm",
        key="chunk_chars",
        label="LLM 分块字符数",
        default=3000,
        minimum=500,
        maximum=20000,
    )
    llm_timeout = r.number(
        cli_attr=None,
        env_names=("LLM_TIMEOUT",),
        section="llm",
        key="timeout",
        label="LLM 超时（秒）",
        default=120.0,
        minimum=5.0,
        maximum=1800.0,
    )
    llm_prompt_file = r.path(
        cli_attr="prompt_file",
        env_names=("LLM_PROMPT_FILE",),
        section="llm",
        key="prompt_file",
        label="LLM 提示词文件",
    )
    if llm_prompt_file is not None and not llm_prompt_file.is_file():
        raise ConfigError(f"LLM 提示词文件不存在或不是文件：{llm_prompt_file}")

    return Settings(
        output=output,
        output_is_dir=output_is_dir,
        keep_audio=keep_audio,
        save_raw=save_raw,
        force=force,
        download=DownloadConfig(
            cookies=cookies,
            cookies_from_browser=cookies_from_browser,
            ffmpeg_location=ffmpeg_location,
        ),
        asr=ASRConfig(
            backend=asr_backend,
            model=asr_model,
            device=asr_device,
            compute_type=asr_compute_type,
            language=asr_language,
            model_dir=asr_model_dir,
            api_base_url=asr_api_base_url,
            api_key=asr_api_key,
            api_model=asr_api_model,
            chunk_seconds=asr_chunk_seconds,
            timeout=asr_timeout,
            cpu_threads=asr_cpu_threads,
            beam_size=asr_beam_size,
        ),
        llm=LLMConfig(
            backend=llm_backend,
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            ollama_base_url=ollama_base_url,
            ollama_model=llm_ollama_model,
            temperature=llm_temperature,
            chunk_chars=llm_chunk_chars,
            timeout=llm_timeout,
            prompt_file=llm_prompt_file,
        ),
    )
