from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from bv_summary.asr.openai_backend import OpenAIASRBackend
from bv_summary.llm.openai_compatible import OpenAICompatibleBackend
from bv_summary.llm.prompts import DEFAULT_SYSTEM_PROMPT
from bv_summary.models import AudioFile
from bv_summary.openai_utils import create_openai_client

from .helpers import make_asr_config, make_llm_config, write_wav


@pytest.fixture()
def mock_openai_server() -> Iterator[tuple[str, type[BaseHTTPRequestHandler]]]:
    class Handler(BaseHTTPRequestHandler):
        requests: list[dict[str, Any]] = []
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if self.path.endswith("/chat/completions"):
                payload = json.loads(body.decode("utf-8"))
                Handler.requests.append({"kind": "chat", **payload})
                if payload.get("stream"):
                    chunks = [
                        {
                            "id": "chatcmpl-mock",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": "mock",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": None,
                                }
                            ],
                        },
                        {
                            "id": "chatcmpl-mock",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": "mock",
                            "choices": [
                                {"index": 0, "delta": {"content": "纠正"}, "finish_reason": None}
                            ],
                        },
                        {
                            "id": "chatcmpl-mock",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": "mock",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": "文本。"},
                                    "finish_reason": "stop",
                                }
                            ],
                        },
                    ]
                    sse = (
                        "".join(
                            f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks
                        )
                        + "data: [DONE]\n\n"
                    )
                    raw = sse.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                self._send_json(
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "mock",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "非流式文本。"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
                return
            if self.path.endswith("/audio/transcriptions"):
                Handler.requests.append(
                    {
                        "kind": "asr",
                        "content_type": self.headers.get("Content-Type", ""),
                        "has_wav": b"RIFF" in body,
                        "body_len": len(body),
                    }
                )
                self._send_json({"text": "在线转写结果。", "language": "zh"})
                return
            self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1", Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_sdk_chat_streaming_against_openai_wire_format(
    mock_openai_server: tuple[str, type[BaseHTTPRequestHandler]],
) -> None:
    base_url, handler = mock_openai_server
    backend = OpenAICompatibleBackend(
        config=make_llm_config(),
        name="openai",
        model="mock",
        base_url=base_url,
        api_key="test-key",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    result = backend.correct("原文。")
    assert result.text == "纠正文本。"
    assert any(request["kind"] == "chat" and request.get("stream") for request in handler.requests)


def test_sdk_transcription_upload_against_openai_wire_format(
    tmp_path: Path, mock_openai_server: tuple[str, type[BaseHTTPRequestHandler]]
) -> None:
    base_url, handler = mock_openai_server
    source = write_wav(tmp_path / "audio.wav", duration=0.5)
    audio = AudioFile(
        path=source,
        duration=0.5,
        sample_rate=16_000,
        channels=1,
        size_bytes=source.stat().st_size,
    )
    backend = OpenAIASRBackend(
        make_asr_config(
            backend="openai",
            api_key="test-key",
            api_base_url=base_url,
            language="zh",
        )
    )
    transcript = backend.transcribe(audio)
    assert transcript.text == "在线转写结果。"
    asr_requests = [request for request in handler.requests if request["kind"] == "asr"]
    assert len(asr_requests) == 1
    assert "multipart/form-data" in asr_requests[0]["content_type"]
    assert asr_requests[0]["has_wav"] is True


def test_openai_client_accepts_default_base_url() -> None:
    client = create_openai_client(api_key="test-key", base_url=None, timeout=1.0)
    assert hasattr(client, "chat")
    assert hasattr(client, "audio")
