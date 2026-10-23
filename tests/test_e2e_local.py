"""Offline end-to-end test: local media -> real yt-dlp/ffmpeg -> SDK wire mocks."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from bv_summary.audio import resolve_ffmpeg
from bv_summary.errors import DependencyError
from bv_summary.pipeline import Pipeline
from bv_summary.ui import Reporter

from .helpers import make_asr_config, make_llm_config, make_settings, write_wav


@pytest.fixture()
def local_e2e_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    media_path = write_wav(tmp_path / "speech.wav", duration=0.5)
    media_bytes = media_path.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_HEAD(self) -> None:  # noqa: N802
            if self.path == "/speech.wav":
                self._send(200, "audio/wav", media_bytes)
            else:
                self._send(404, "text/plain", b"not found")

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/speech.wav":
                self._send(200, "audio/wav", media_bytes)
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if self.path.endswith("/audio/transcriptions"):
                self._send(
                    200,
                    "application/json; charset=utf-8",
                    json.dumps(
                        {"text": "这是原始转写。", "language": "zh"}, ensure_ascii=False
                    ).encode(),
                )
                return
            if self.path.endswith("/chat/completions"):
                payload = json.loads(body.decode("utf-8"))
                assert payload["stream"] is True
                chunks = [
                    {
                        "id": "chatcmpl-e2e",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "mock",
                        "choices": [
                            {"index": 0, "delta": {"content": "这是修正"}, "finish_reason": None}
                        ],
                    },
                    {
                        "id": "chatcmpl-e2e",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "mock",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "后的转写。"},
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
                self._send(200, "text/event-stream; charset=utf-8", sse.encode("utf-8"))
                return
            self._send(404, "text/plain", b"not found")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", tmp_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_end_to_end_local_media_and_mock_apis(
    local_e2e_server: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        resolve_ffmpeg()
    except DependencyError:
        pytest.skip("ffmpeg is unavailable")
    base_url, work_root = local_e2e_server
    media_url = f"{base_url}/speech.wav"
    settings = make_settings(
        work_root,
        asr=make_asr_config(
            backend="openai",
            api_base_url=f"{base_url}/v1",
            api_key=None,
            language="zh",
        ),
        llm=make_llm_config(
            backend="openai",
            model="mock",
            base_url=f"{base_url}/v1",
            api_key=None,
        ),
    )
    result = Pipeline(settings, Reporter(quiet=True)).run(media_url)

    assert result.title == "speech"
    assert result.asr_backend == "openai"
    assert result.llm_backend == "openai"
    assert result.text == "这是修正后的转写。"
    assert result.output_path.read_text(encoding="utf-8") == "这是修正后的转写。\n"
    assert result.duration and result.duration > 0.3
