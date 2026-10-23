from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from bv_summary.errors import ConfigError, LLMError
from bv_summary.llm import factory
from bv_summary.llm.base import LLMSelection
from bv_summary.llm.factory import build_llm_selection, probe_ollama
from bv_summary.llm.openai_compatible import OpenAICompatibleBackend
from bv_summary.llm.prompts import DEFAULT_SYSTEM_PROMPT, load_system_prompt

from .helpers import make_llm_config


def _events(*pieces: str) -> list[Any]:
    return [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
        for piece in pieces
    ]


class _FakeCompletions:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.handler(kwargs)


class _FakeClient:
    def __init__(self, handler) -> None:
        self.completions = _FakeCompletions(handler)
        self.chat = SimpleNamespace(completions=self.completions)


def _backend(config=None, handler=None) -> tuple[OpenAICompatibleBackend, _FakeClient]:
    if config is None:
        config = make_llm_config()
    client = _FakeClient(handler)
    backend = OpenAICompatibleBackend(
        config=config,
        name="openai",
        model="test-model",
        base_url="http://127.0.0.1:9999/v1",
        api_key="test-key",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        client=client,
    )
    return backend, client


def test_streaming_correction() -> None:
    def handler(kwargs: dict[str, Any]) -> Any:
        assert kwargs["stream"] is True
        return _events("你好", "，世界。")

    backend, client = _backend(handler=handler)
    progress: list[tuple[float | None, str]] = []
    result = backend.correct("你好世界", progress=lambda p, m: progress.append((p, m)))

    assert result.text == "你好，世界。"
    assert result.backend == "openai"
    assert result.model == "test-model"
    assert result.chunks == 1
    assert len(client.completions.calls) == 1
    assert client.completions.calls[0]["messages"][0]["role"] == "system"
    assert "你好世界" in client.completions.calls[0]["messages"][1]["content"]
    assert progress[-1][0] == 100.0


def test_stream_rejected_falls_back_to_nonstream() -> None:
    class BadRequestError(Exception):
        status_code = 400

    def handler(kwargs: dict[str, Any]) -> Any:
        if kwargs["stream"]:
            raise BadRequestError("stream is not supported")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="纠正文本。"))]
        )

    backend, client = _backend(handler=handler)
    result = backend.correct("纠正文本")
    assert result.text == "纠正文本。"
    assert len(client.completions.calls) == 2
    assert client.completions.calls[1]["stream"] is False
    assert "temperature" not in client.completions.calls[1]


def test_authentication_error_is_wrapped() -> None:
    class AuthenticationError(Exception):
        status_code = 401

    def handler(kwargs: dict[str, Any]) -> Any:
        raise AuthenticationError("bad key")

    backend, _client = _backend(handler=handler)
    with pytest.raises(LLMError) as caught:
        backend.correct("some text")
    assert "认证失败" in caught.value.message
    assert "LLM_API_KEY" in (caught.value.hint or "")


def test_short_model_output_is_rejected() -> None:
    def handler(kwargs: dict[str, Any]) -> Any:
        return _events("short")

    backend, _client = _backend(handler=handler)
    with pytest.raises(LLMError, match="明显短于原文"):
        backend.correct("原文" * 200)


def test_long_text_is_chunked() -> None:
    def handler(kwargs: dict[str, Any]) -> Any:
        user = kwargs["messages"][1]["content"]
        # Produce enough text to satisfy the accidental-truncation guard.
        return _events("修" * len(user))

    config = make_llm_config(chunk_chars=16)
    backend, client = _backend(config=config, handler=handler)
    result = backend.correct("第一句。第二句。第三句。第四句。第五句。")
    assert result.chunks > 1
    assert len(client.completions.calls) > 1


def test_llm_returns_error_on_empty_text() -> None:
    def handler(kwargs: dict[str, Any]) -> Any:
        return iter([])

    backend, _client = _backend(handler=handler)
    with pytest.raises(LLMError, match="空内容|空"):
        backend.correct("some text")


@pytest.fixture()
def local_ollama_models() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                body = json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {"id": "qwen2.5:7b", "object": "model"},
                            {"id": "llama3.2:latest", "object": "model"},
                        ],
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_probe_ollama_detects_models(local_ollama_models: str) -> None:
    assert probe_ollama(local_ollama_models, "qwen2.5:7b") == (True, "")
    available, reason = probe_ollama(local_ollama_models, "missing-model")
    assert available is False
    assert "未找到模型" in reason


def test_factory_auto_uses_reachable_ollama(local_ollama_models: str) -> None:
    selection = build_llm_selection(
        make_llm_config(
            backend="auto",
            ollama_base_url=local_ollama_models,
            ollama_model="qwen2.5:7b",
        )
    )
    assert [backend.name for backend in selection.backends] == ["ollama"]
    assert selection.backends[0].base_url == local_ollama_models


def test_factory_prefers_remote_then_probes_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        factory,
        "probe_ollama",
        lambda base_url, model: (True, ""),
    )
    config = make_llm_config(
        backend="auto",
        model="remote-model",
        base_url="https://api.example.com/v1",
        api_key="secret",
        ollama_model="local-model",
    )
    selection = build_llm_selection(config)
    assert [backend.name for backend in selection.backends] == ["openai", "ollama"]
    assert any("本地 Ollama" in note for note in selection.notes)


def test_factory_uses_ollama_without_remote_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "probe_ollama", lambda base_url, model: (True, ""))
    selection = build_llm_selection(make_llm_config(backend="auto", ollama_model="qwen2.5:7b"))
    assert [backend.name for backend in selection.backends] == ["ollama"]


def test_factory_auto_without_candidates_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "probe_ollama", lambda base_url, model: (False, "offline"))
    with pytest.raises(ConfigError, match="找不到可用后端"):
        build_llm_selection(make_llm_config(backend="auto"))


def test_factory_openai_requires_key_for_remote() -> None:
    with pytest.raises(ConfigError, match="API Key"):
        build_llm_selection(
            make_llm_config(
                backend="openai",
                model="gpt-test",
                base_url="https://api.example.com/v1",
            )
        )


def test_factory_openai_allows_keyless_local() -> None:
    selection = build_llm_selection(
        make_llm_config(
            backend="openai",
            model="local-model",
            base_url="http://127.0.0.1:8000/v1",
        )
    )
    assert [backend.name for backend in selection.backends] == ["openai"]


def test_factory_ollama_requires_model() -> None:
    with pytest.raises(ConfigError, match="模型"):
        build_llm_selection(make_llm_config(backend="ollama"))


def test_factory_explicit_ollama_prefers_generic_model() -> None:
    selection = build_llm_selection(
        make_llm_config(
            backend="ollama",
            model="cli-model",
            ollama_model="fallback-specific",
        )
    )
    assert [backend.model for backend in selection.backends] == ["cli-model"]


def test_none_backend_returns_empty_selection() -> None:
    selection = build_llm_selection(make_llm_config(backend="none"))
    assert selection == LLMSelection(())


def test_custom_prompt_loading(tmp_path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Custom prompt", encoding="utf-8")
    assert load_system_prompt(prompt) == "Custom prompt"
    with pytest.raises(ConfigError):
        load_system_prompt(tmp_path / "missing.txt")
