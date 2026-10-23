"""OpenAI-compatible chat backend used for both remote APIs and Ollama."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from ..config import LLMConfig
from ..errors import AppError, LLMError
from ..models import Correction, ProgressCallback
from ..openai_utils import create_openai_client, describe_openai_error
from ..utils import clean_error_message, endpoint_display
from .base import LLMBackend
from .prompts import build_user_prompt
from .text_chunks import chunk_text

_FENCE_RE = re.compile(r"^```[^\n`]*\n?(.*?)\n?```$", flags=re.DOTALL)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                value = part.get("text")
                if value:
                    pieces.append(str(value))
            else:
                value = getattr(part, "text", None)
                if value:
                    pieces.append(str(value))
        return "".join(pieces)
    return str(content)


def _choice_field(choice: Any, name: str) -> Any:
    if isinstance(choice, dict):
        return choice.get(name)
    return getattr(choice, name, None)


def _choices(event: Any) -> list[Any]:
    if isinstance(event, dict):
        value = event.get("choices")
    else:
        value = getattr(event, "choices", None)
    return list(value) if value else []


def _delta_text(event: Any) -> str:
    for choice in _choices(event):
        delta = _choice_field(choice, "delta")
        if delta is not None:
            content = _choice_field(delta, "content")
            if content is not None:
                return _content_to_text(content)
        # Compatibility with non-chat streaming endpoints.
        legacy = _choice_field(choice, "text")
        if legacy:
            return str(legacy)
    return ""


def _completion_text(response: Any) -> str:
    if isinstance(response, dict):
        choices = response.get("choices") or []
    else:
        choices = getattr(response, "choices", None) or []
    for choice in choices:
        message = _choice_field(choice, "message")
        if message is not None:
            content = _choice_field(message, "content")
            if content is not None:
                text = _content_to_text(content)
                if text:
                    return text
        legacy = _choice_field(choice, "text")
        if legacy:
            return str(legacy)
    return ""


def _stream_fallback_allowed(exc: BaseException) -> bool:
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    return name in {"BadRequestError", "UnprocessableEntityError"} or status in {400, 422}


class OpenAICompatibleBackend(LLMBackend):
    """Chat-completions client with chunking, streaming and one non-stream retry."""

    def __init__(
        self,
        *,
        config: LLMConfig,
        name: str,
        model: str,
        base_url: str | None,
        api_key: str,
        system_prompt: str,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.name = name
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self._system_prompt = system_prompt
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = create_openai_client(
                api_key=self._api_key,
                base_url=self.base_url,
                timeout=self.config.timeout,
            )
        return self._client

    def _messages(self, chunk: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": build_user_prompt(chunk)},
        ]

    def _stream_chunk(
        self,
        client: Any,
        chunk: str,
        on_token: Callable[[int], None],
    ) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=self._messages(chunk),
            temperature=self.config.temperature,
            stream=True,
        )
        pieces: list[str] = []
        characters = 0
        try:
            # Some OpenAI-compatible servers ignore stream=True and return a
            # complete ChatCompletion. Treat that as an empty stream and let the
            # caller re-request non-streaming.
            if type(response).__name__ in {"ChatCompletion", "ChatCompletionMessage"}:
                return _completion_text(response)
            for event in response:
                piece = _delta_text(event)
                if piece:
                    pieces.append(piece)
                    characters += len(piece)
                    on_token(characters)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        return "".join(pieces)

    def _nonstream_chunk(
        self,
        client: Any,
        chunk: str,
        *,
        include_temperature: bool,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(chunk),
            "stream": False,
        }
        if include_temperature:
            kwargs["temperature"] = self.config.temperature
        response = client.chat.completions.create(**kwargs)
        return _completion_text(response)

    def _clean_response(self, raw: str, source: str) -> str:
        text = raw.strip().lstrip("\ufeff")
        fenced = _FENCE_RE.fullmatch(text)
        if fenced:
            text = fenced.group(1).strip()
        if text.startswith("<transcript>") and text.endswith("</transcript>"):
            text = text[len("<transcript>") : -len("</transcript>")].strip()
        if not text:
            raise LLMError(
                "LLM 返回了空内容。",
                hint="请检查模型是否正常响应，或尝试其他模型/服务。",
            )
        if len(source) >= 200 and len(text) < len(source) * 0.5:
            raise LLMError(
                "LLM 返回的文本明显短于原文，可能发生截断或内容丢失。",
                hint=(
                    "可减小 LLM_CHUNK_CHARS、增大模型上下文，或改用其他模型。"
                    "本次结果不会写入最终文件。"
                ),
            )
        return text

    def _correct_chunk(
        self,
        client: Any,
        chunk: str,
        *,
        index: int,
        total: int,
        base_percent: float,
        progress: ProgressCallback,
    ) -> str:
        token_state = {"chars": 0}
        last_update = 0.0

        def on_token(characters: int) -> None:
            nonlocal last_update
            token_state["chars"] = characters
            now = time.monotonic()
            if now - last_update < 0.2:
                return
            last_update = now
            estimated = min(0.95, characters / max(1, len(chunk)))
            percent = base_percent + (estimated / total) * 100.0
            progress(
                percent,
                f"LLM 纠错 · 第 {index}/{total} 块 · 已生成 {characters} 字",
            )

        stream_error: BaseException | None = None
        progress(base_percent, f"LLM 纠错 · 正在处理第 {index}/{total} 块（{len(chunk)} 字）…")
        try:
            raw = self._stream_chunk(client, chunk, on_token)
        except Exception as exc:
            if token_state["chars"] > 0 or not _stream_fallback_allowed(exc):
                raise
            # Some OpenAI-compatible servers reject stream=True or certain
            # optional parameters. One non-streaming retry (without temperature)
            # is cheap and greatly improves compatibility.
            stream_error = exc
            raw = ""
        if not raw.strip():
            if stream_error is not None:
                progress(base_percent, f"LLM 纠错 · 第 {index}/{total} 块改为非流式重试…")
            raw = self._nonstream_chunk(
                client,
                chunk,
                include_temperature=stream_error is None,
            )
        return self._clean_response(raw, chunk)

    def correct(
        self,
        text: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> Correction:
        callback = progress or (lambda _percent, _message: None)
        normalized = (text or "").strip()
        if not normalized:
            return Correction(text="", backend=self.name, model=self.model, chunks=0)

        chunks = chunk_text(normalized, self.config.chunk_chars)
        client = self._get_client()
        corrected_chunks: list[str] = []
        total = len(chunks)
        callback(0.0, f"LLM 纠错 · 共 {total} 块（每块最多 {self.config.chunk_chars} 字）")

        for index, chunk in enumerate(chunks, start=1):
            base_percent = (index - 1) / total * 100.0
            try:
                corrected = self._correct_chunk(
                    client,
                    chunk,
                    index=index,
                    total=total,
                    base_percent=base_percent,
                    progress=callback,
                )
            except AppError:
                raise
            except Exception as exc:
                message, hint = describe_openai_error(
                    exc,
                    service="LLM 纠错",
                    key_env_hint="LLM_API_KEY / OPENAI_API_KEY",
                )
                endpoint = endpoint_display(self.base_url)
                detail = clean_error_message(exc, limit=900)
                raise LLMError(
                    f"LLM 纠错第 {index}/{total} 块失败（{endpoint}，模型 {self.model}）：{message}",
                    hint=hint or detail,
                ) from exc
            corrected_chunks.append(corrected)
            callback(
                index / total * 100.0,
                f"LLM 纠错 · 已完成 {index}/{total} 块",
            )

        result_text = "\n\n".join(corrected_chunks).strip()
        if not result_text:
            raise LLMError("LLM 纠错后没有得到任何文本。")
        callback(100.0, f"LLM 纠错完成 · {total} 块")
        return Correction(
            text=result_text,
            backend=self.name,
            model=self.model,
            chunks=total,
        )
