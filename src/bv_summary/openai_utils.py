"""Shared helpers for OpenAI-compatible SDK calls.

Importing the SDK is deliberately lazy so `--help`, --llm-backend none and
local-only runs do not require the OpenAI package to be importable.
"""

from __future__ import annotations

from typing import Any

from .errors import DependencyError
from .utils import clean_error_message


def create_openai_client(
    *,
    api_key: str,
    base_url: str | None,
    timeout: float,
    max_retries: int = 2,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DependencyError(
            "缺少 openai 包，无法使用 OpenAI 兼容接口。",
            hint='请执行：pip install -e ".[local-asr]"（或 pip install openai）',
        ) from exc
    try:
        return OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception as exc:
        raise DependencyError(
            "无法初始化 OpenAI 兼容客户端。",
            hint=clean_error_message(exc),
        ) from exc


def describe_openai_error(
    exc: BaseException,
    *,
    service: str,
    key_env_hint: str,
) -> tuple[str, str | None]:
    """Map SDK errors by class name/status, stable across SDK major versions."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    detail = clean_error_message(getattr(exc, "message", None) or str(exc), limit=1200)

    if name in {"AuthenticationError", "PermissionDeniedError"} or status in {401, 403}:
        return (
            f"{service}认证失败（HTTP {status or '401/403'}）：{detail}",
            f"请检查 {key_env_hint} 是否有效、是否有该模型的权限，并确认 base_url 指向正确的服务。",
        )
    if name in {"APITimeoutError", "TimeoutError"} or status == 408:
        return (
            f"{service}请求超时：{detail}",
            "可增大对应超时设置、改用更快的模型/服务，或稍后重试。",
        )
    if name in {"APIConnectionError", "ConnectError"} or "connection" in name.lower():
        return (
            f"无法连接{service}服务：{detail}",
            "请检查网络、代理、base_url 和本地服务是否已启动。",
        )
    if name in {"RateLimitError"} or status == 429:
        return (
            f"{service}请求被限流：{detail}",
            "请稍后重试，或检查 API 配额/并发限制。",
        )
    if name in {"NotFoundError"} or status == 404:
        return (
            f"{service}接口或模型不存在：{detail}",
            "请检查模型名与 base_url；OpenAI 兼容地址通常需要以 /v1 结尾。",
        )
    if name in {"BadRequestError", "UnprocessableEntityError"} or status in {400, 422}:
        return (
            f"{service}请求无效（HTTP {status or 400}）：{detail}",
            "常见原因：模型名错误、文件过长/格式不支持、上下文超限或参数不被该服务支持。",
        )
    if name == "APIStatusError" or status is not None:
        return (
            f"{service}服务返回错误（HTTP {status}）：{detail}",
            "请检查服务端日志、模型名、API 地址与网络代理。",
        )
    return f"{service}调用失败：{detail}", None


def response_text(response: Any) -> str:
    """Extract text from dict-like, SDK model, or plain-string responses."""
    if isinstance(response, str):
        value = response
    elif isinstance(response, dict):
        value = response.get("text")
    else:
        value = getattr(response, "text", None)
        if value is None and hasattr(response, "model_dump"):
            try:
                value = response.model_dump().get("text")
            except Exception:
                value = None
    if isinstance(value, list):
        value = "".join(str(part) for part in value)
    if value is None:
        return ""
    return str(value).strip()
