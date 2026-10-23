"""Online ASR using an OpenAI-compatible transcription endpoint."""

from __future__ import annotations

import math
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ASRConfig
from ..errors import ASRError, ConfigError
from ..models import AudioFile, ProgressCallback, Segment, Transcript
from ..openai_utils import create_openai_client, describe_openai_error, response_text
from ..utils import is_local_http_url
from .base import ASRBackend

# Stay below OpenAI's 25 MB transcription upload limit.
_MAX_UPLOAD_BYTES = 24 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _AudioChunk:
    path: Path
    start: float
    end: float


class OpenAIASRBackend(ASRBackend):
    name = "openai"

    def __init__(self, config: ASRConfig, *, client: Any | None = None) -> None:
        self.config = config
        self.model = config.api_model
        key = config.api_key or ("not-needed" if is_local_http_url(config.api_base_url) else None)
        if not key:
            raise ConfigError(
                "在线 ASR 需要 API Key。",
                hint=(
                    "请设置 ASR_API_KEY 或 OPENAI_API_KEY；若目标是本地 OpenAI 兼容服务，"
                    "请把 ASR_BASE_URL 指向本机/私网地址（可无 Key）。"
                ),
            )
        self._api_key = key
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = create_openai_client(
                api_key=self._api_key,
                base_url=self.config.api_base_url,
                timeout=self.config.timeout,
            )
        return self._client

    @staticmethod
    def _split_wav(path: Path, output_dir: Path, chunk_seconds: int) -> list[_AudioChunk]:
        """Split PCM WAV by frame boundaries without ffmpeg or decode loss."""
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with wave.open(str(path), "rb") as source:
                channels = source.getnchannels()
                sample_width = source.getsampwidth()
                sample_rate = source.getframerate()
                total_frames = source.getnframes()
                if channels <= 0 or sample_width <= 0 or sample_rate <= 0 or total_frames <= 0:
                    raise ASRError("音频 WAV 参数无效，无法分块上传。")
                bytes_per_frame = channels * sample_width
                # Leave roughly one second of headroom for the WAV header.
                max_frames_by_bytes = max(
                    1, (_MAX_UPLOAD_BYTES - 1024) // bytes_per_frame - sample_rate
                )
                frames_per_chunk = min(chunk_seconds * sample_rate, max_frames_by_bytes)
                if frames_per_chunk < sample_rate:
                    raise ASRError(
                        "单块音频仍超过在线 ASR 的上传上限。", hint="请缩短视频或改用本地 ASR。"
                    )

                chunks: list[_AudioChunk] = []
                chunk_count = math.ceil(total_frames / frames_per_chunk)
                for index in range(chunk_count):
                    start_frame = index * frames_per_chunk
                    frame_count = min(frames_per_chunk, total_frames - start_frame)
                    output_path = output_dir / f"chunk_{index + 1:04d}.wav"
                    source.setpos(start_frame)
                    remaining = frame_count
                    with wave.open(str(output_path), "wb") as target:
                        target.setnchannels(channels)
                        target.setsampwidth(sample_width)
                        target.setframerate(sample_rate)
                        while remaining > 0:
                            block = source.readframes(min(remaining, sample_rate * 10))
                            if not block:
                                break
                            target.writeframes(block)
                            remaining -= min(remaining, sample_rate * 10)
                    if remaining > 0:
                        raise ASRError("音频分块过程中出现意外结束，无法继续在线识别。")
                    if output_path.stat().st_size > _MAX_UPLOAD_BYTES:
                        raise ASRError(
                            "分块后的音频仍超过 24 MB 上传上限。", hint="请改用本地 ASR。"
                        )
                    chunks.append(
                        _AudioChunk(
                            path=output_path,
                            start=start_frame / sample_rate,
                            end=(start_frame + frame_count) / sample_rate,
                        )
                    )
                return chunks
        except wave.Error as exc:
            raise ASRError("无法读取 WAV 音频用于分块。", hint=str(exc)) from exc

    def _prepare_chunks(
        self, audio_path: Path, duration: float, progress: ProgressCallback
    ) -> tuple[list[_AudioChunk], tempfile.TemporaryDirectory[str] | None]:
        if audio_path.stat().st_size <= _MAX_UPLOAD_BYTES:
            return [_AudioChunk(path=audio_path, start=0.0, end=duration)], None
        progress(None, "音频超过 24 MB，正在按上限分块…")
        temp_dir = tempfile.TemporaryDirectory(prefix="bv-asr-chunks-")
        try:
            chunks = self._split_wav(audio_path, Path(temp_dir.name), self.config.chunk_seconds)
        except BaseException:
            temp_dir.cleanup()
            raise
        progress(None, f"已分成 {len(chunks)} 块，将逐块上传识别…")
        return chunks, temp_dir

    def _call_api(self, chunk: _AudioChunk) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "response_format": "json",
        }
        if self.config.language:
            kwargs["language"] = self.config.language
        with chunk.path.open("rb") as handle:
            response = self._get_client().audio.transcriptions.create(
                file=(chunk.path.name, handle, "audio/wav"),
                **kwargs,
            )
        return response

    def transcribe(
        self,
        audio: AudioFile,
        *,
        progress: ProgressCallback | None = None,
    ) -> Transcript:
        callback = progress or (lambda _percent, _message: None)
        chunks, temp_dir = self._prepare_chunks(audio.path, audio.duration, callback)
        try:
            chunk_texts: list[tuple[_AudioChunk, str]] = []
            language: str | None = None
            total = len(chunks)
            for index, chunk in enumerate(chunks, start=1):
                percent = (index - 1) / total * 100.0
                callback(
                    percent,
                    f"在线 ASR · 上传第 {index}/{total} 块（{chunk.end - chunk.start:.0f} 秒）…",
                )
                try:
                    response = self._call_api(chunk)
                except Exception as exc:
                    message, hint = describe_openai_error(
                        exc,
                        service="在线语音识别",
                        key_env_hint="ASR_API_KEY / OPENAI_API_KEY",
                    )
                    raise ASRError(
                        f"在线 ASR 第 {index}/{total} 块失败：{message}",
                        hint=hint,
                    ) from exc

                text = response_text(response)
                if not text:
                    raise ASRError(
                        f"在线 ASR 第 {index}/{total} 块返回了空文本。",
                        hint="请确认音频包含人声，或检查该在线 ASR 服务是否兼容。若兼容服务不支持 WAV，请反馈。",
                    )
                chunk_texts.append((chunk, text))
                if language is None:
                    response_language = (
                        response.get("language")
                        if isinstance(response, dict)
                        else getattr(response, "language", None)
                    )
                    if response_language:
                        language = str(response_language)
                callback(
                    index / total * 100.0,
                    f"在线 ASR · 已完成 {index}/{total} 块",
                )

            if not chunk_texts:
                raise ASRError("在线 ASR 没有返回任何文本。")
            combined_text = "\n\n".join(text for _chunk, text in chunk_texts)
            segments = tuple(
                Segment(start=chunk.start, end=chunk.end, text=text) for chunk, text in chunk_texts
            )
            callback(100.0, f"在线语音识别完成 · {len(segments)} 块")
            return Transcript(
                text=combined_text,
                segments=segments,
                language=language or self.config.language,
                backend=self.name,
                model=self.model,
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
