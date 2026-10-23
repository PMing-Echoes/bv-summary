"""LLM backend selection, including auto-detected local Ollama fallback."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from ..config import LLMConfig
from ..errors import ConfigError
from ..utils import endpoint_display, is_local_http_url
from .base import LLMBackend, LLMSelection
from .openai_compatible import OpenAICompatibleBackend
from .prompts import load_system_prompt


def _model_available(requested: str, installed: list[str]) -> bool:
    names = {name.strip() for name in installed if name.strip()}
    if not names:
        return True
    candidates = {requested}
    if ":" not in requested:
        candidates.add(f"{requested}:latest")
    return bool(candidates & names)


def _extract_installed_models(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    names: list[str] = []
    data = payload.get("data")
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and entry.get("id"):
                names.append(str(entry["id"]))
    models = payload.get("models")
    if isinstance(models, list):
        for entry in models:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("model")
                if name:
                    names.append(str(name))
    return names


def _api_probe_urls(base_url: str) -> list[str]:
    base = base_url.rstrip("/")
    urls = [f"{base}/models"]
    parts = urlsplit(base)
    path = parts.path.rstrip("/")
    root_path = path[:-3] if path.endswith("/v1") else path
    tags_url = urlunsplit((parts.scheme, parts.netloc, f"{root_path}/api/tags", "", ""))
    if tags_url not in urls and tags_url.startswith(("http://", "https://")):
        urls.append(tags_url)
    return urls


def probe_ollama(base_url: str, model: str | None) -> tuple[bool, str]:
    """Check the configured local endpoint and (when possible) the model tag."""
    try:
        import httpx
    except ImportError:
        return False, "无法检测本地 Ollama：缺少 httpx。"
    timeout = httpx.Timeout(1.5, connect=1.0)
    headers = {"Authorization": "Bearer ollama"}
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            for url in _api_probe_urls(base_url):
                try:
                    response = client.get(url, headers=headers)
                except Exception:
                    continue
                if response.status_code != 200:
                    continue
                try:
                    installed = _extract_installed_models(response.json())
                except Exception:
                    installed = []
                if model and not _model_available(model, installed):
                    return (
                        False,
                        f"Ollama 正在运行，但未找到模型“{model}”。可执行：ollama pull {model}",
                    )
                return True, ""
    except Exception:
        pass
    model_hint = f"；准备模型：ollama pull {model}" if model else ""
    return (
        False,
        f"未检测到可访问的 Ollama 服务（{endpoint_display(base_url)}）。"
        f"启动方式：ollama serve{model_hint}",
    )


def _openai_backend(
    config: LLMConfig,
    *,
    model: str,
    base_url: str | None,
    api_key: str,
    system_prompt: str,
) -> LLMBackend:
    return OpenAICompatibleBackend(
        config=config,
        name="openai",
        model=model,
        base_url=base_url,
        api_key=api_key,
        system_prompt=system_prompt,
    )


def _ollama_backend(
    config: LLMConfig,
    *,
    model: str,
    system_prompt: str,
) -> LLMBackend:
    return OpenAICompatibleBackend(
        config=config,
        name="ollama",
        model=model,
        base_url=config.ollama_base_url,
        api_key="ollama",
        system_prompt=system_prompt,
    )


def _remote_key(config: LLMConfig) -> str | None:
    if config.api_key:
        return config.api_key
    if config.base_url and is_local_http_url(config.base_url):
        return "not-needed"
    return None


def build_llm_selection(config: LLMConfig) -> LLMSelection:
    if config.backend == "none":
        return LLMSelection(())

    system_prompt = load_system_prompt(config.prompt_file)

    if config.backend == "openai":
        if not config.model:
            raise ConfigError(
                "LLM_BACKEND=openai 需要模型名。",
                hint="请用 --model 或 LLM_MODEL 指定模型，例如 --model gpt-4o-mini。",
            )
        key = _remote_key(config)
        if not key:
            raise ConfigError(
                "LLM_BACKEND=openai 缺少 API Key。",
                hint=(
                    "请设置 LLM_API_KEY 或 OPENAI_API_KEY；如果它是本机/私网无鉴权服务，"
                    "请用 --base-url 指向该地址。也可以改用 --llm-backend ollama 或 none。"
                ),
            )
        return LLMSelection(
            (
                _openai_backend(
                    config,
                    model=config.model,
                    base_url=config.base_url,
                    api_key=key,
                    system_prompt=system_prompt,
                ),
            )
        )

    if config.backend == "ollama":
        # For an explicitly selected backend, the generic --model / LLM_MODEL
        # should win over the fallback-specific OLLAMA_MODEL.
        model = config.model or config.ollama_model
        if not model:
            raise ConfigError(
                "LLM_BACKEND=ollama 需要 Ollama 模型名。",
                hint="请设置 OLLAMA_MODEL 或使用 --model，例如 --model qwen2.5:7b；并先执行 ollama pull。",
            )
        return LLMSelection((_ollama_backend(config, model=model, system_prompt=system_prompt),))

    # auto: prefer an explicitly configured OpenAI-compatible target, then a
    # reachable local Ollama with a configured model.
    backends: list[LLMBackend] = []
    notes: list[str] = []
    seen: set[tuple[str, str]] = set()

    remote_model = config.model
    remote_configured = bool(config.base_url or config.api_key)
    remote_key = _remote_key(config)
    remote_endpoint = config.base_url

    if remote_configured:
        if not remote_model:
            notes.append("LLM auto：检测到远程 API 配置，但缺少 LLM_MODEL，已跳过远程后端。")
        elif not remote_key:
            notes.append(
                "LLM auto：LLM_BASE_URL 指向远程地址但未设置 LLM_API_KEY，已跳过远程后端。"
            )
        else:
            backends.append(
                _openai_backend(
                    config,
                    model=remote_model,
                    base_url=remote_endpoint,
                    api_key=remote_key,
                    system_prompt=system_prompt,
                )
            )
            seen.add((remote_endpoint or "", remote_model))

    local_model = config.ollama_model or config.model
    local_duplicate = local_model is not None and (
        (config.ollama_base_url.rstrip("/"), local_model) in seen
    )
    if local_model and not local_duplicate:
        available, reason = probe_ollama(config.ollama_base_url, local_model)
        if available:
            backends.append(_ollama_backend(config, model=local_model, system_prompt=system_prompt))
            seen.add((config.ollama_base_url.rstrip("/"), local_model))
            if backends and remote_configured and remote_model and remote_key:
                notes.append("LLM auto：远程 LLM 失败后会尝试本地 Ollama。")
        elif not backends or config.ollama_model:
            notes.append(f"LLM auto 本地回退不可用：{reason}")
    elif not local_model and not backends:
        notes.append("LLM auto：本地 Ollama 回退未配置模型（OLLAMA_MODEL 或 LLM_MODEL）。")

    if not backends:
        detail = "\n".join(f"- {note}" for note in notes) if notes else "- 未配置可用后端。"
        raise ConfigError(
            "LLM auto 找不到可用后端。\n" + detail,
            hint=(
                "任选一种方式：1) 设置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL；"
                "2) 启动 Ollama 并设置 OLLAMA_MODEL；3) 使用 --llm-backend none 跳过文本纠错。"
            ),
        )
    return LLMSelection(tuple(backends), tuple(notes))
