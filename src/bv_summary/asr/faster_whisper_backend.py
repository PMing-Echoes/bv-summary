"""Local ASR through the actively maintained faster-whisper project."""

from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import Any

from ..config import ASRConfig
from ..cuda import configure_windows_cuda_dlls
from ..errors import ASRError
from ..models import AudioFile, ProgressCallback, Segment, Transcript
from ..transcript import segments_to_text
from ..utils import clean_error_message
from .base import ASRBackend


def _load_wav_float32(path: Path) -> Any | None:
    """Read our known 16 kHz mono PCM WAV into a float32 numpy array.

    faster-whisper's path-based decoding uses PyAV. PyAV 19 removed the
    ``metadata_errors`` argument still used by faster-whisper 1.2.1, so passing
    a NumPy array both avoids that incompatibility and skips a redundant decode.
    """
    if path.suffix.lower() != ".wav":
        return None
    try:
        import numpy as np

        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                return None
            if handle.getframerate() != 16_000:
                return None
            frames = handle.readframes(handle.getnframes())
        if not frames:
            return None
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    except (ImportError, OSError, wave.Error, ValueError):
        return None


class FasterWhisperBackend(ASRBackend):
    name = "faster-whisper"

    def __init__(self, config: ASRConfig) -> None:
        self.config = config
        self.model = config.model

    def _resolve_device(self) -> tuple[str, str]:
        device = self.config.device
        supported: set[str] = set()
        try:
            import ctranslate2

            if device == "auto":
                try:
                    has_cuda = ctranslate2.get_cuda_device_count() > 0
                except Exception:
                    has_cuda = False
                if has_cuda:
                    try:
                        supported = set(ctranslate2.get_supported_compute_types("cuda"))
                    except Exception:
                        supported = set()
                    # A visible GPU can still lack required CUDA libraries.
                    device = "cuda" if supported else "cpu"
                else:
                    device = "cpu"
            if not supported:
                try:
                    supported = set(ctranslate2.get_supported_compute_types(device))
                except Exception:
                    supported = set()
        except Exception:
            if device == "auto":
                device = "cpu"

        compute_type = self.config.compute_type
        if compute_type == "auto":
            # Not every CUDA GPU supports float16; querying CTranslate2 avoids
            # selecting an unavailable type (e.g. older GPUs).
            if device == "cuda":
                preferences = (
                    "float16",
                    "int8_float16",
                    "bfloat16",
                    "int8_float32",
                    "int8",
                    "float32",
                )
                fallback = "float32"
            else:
                preferences = ("int8", "int8_float32", "float32", "int16")
                fallback = "int8"
            if supported:
                compute_type = next(
                    (candidate for candidate in preferences if candidate in supported),
                    fallback,
                )
            else:
                compute_type = fallback
        return device, compute_type

    def transcribe(
        self,
        audio: AudioFile,
        *,
        progress: ProgressCallback | None = None,
    ) -> Transcript:
        callback = progress or (lambda _percent, _message: None)
        # NVIDIA pip wheels are not on PATH by default on Windows. Register
        # their CUDA DLLs before CTranslate2 attempts to load cuBLAS.
        configure_windows_cuda_dlls()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRError(
                "未安装 faster-whisper，无法使用本地语音识别。",
                hint='请执行：pip install -e ".[local-asr]"',
            ) from exc

        device, compute_type = self._resolve_device()
        model_location = self.config.model
        callback(None, f"加载本地 ASR 模型 {self.config.model}（{device}/{compute_type}）…")
        callback(None, "首次使用该模型时需要下载，请耐心等待…")

        model: Any | None = None
        try:
            model = WhisperModel(
                model_location,
                device=device,
                compute_type=compute_type,
                cpu_threads=self.config.cpu_threads,
                download_root=str(self.config.model_dir) if self.config.model_dir else None,
            )
            callback(None, "模型已加载，正在分析音频…")
            audio_input = _load_wav_float32(audio.path)
            segments_iter, info = model.transcribe(
                audio_input if audio_input is not None else str(audio.path),
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            total_duration = float(getattr(info, "duration", 0.0) or audio.duration or 0.0)
            callback(0.0, "语音识别中…")

            collected: list[Segment] = []
            last_update = 0.0
            for item in segments_iter:
                text = str(getattr(item, "text", "") or "").strip()
                if text:
                    segment = Segment(
                        start=float(getattr(item, "start", 0.0) or 0.0),
                        end=float(getattr(item, "end", 0.0) or 0.0),
                        text=text,
                    )
                    collected.append(segment)
                now = time.monotonic()
                if now - last_update >= 0.15:
                    last_update = now
                    percent = None
                    if total_duration > 0 and collected:
                        percent = min(99.0, collected[-1].end / total_duration * 100.0)
                    label = f"语音识别中 · 已识别 {len(collected)} 段"
                    if percent is not None:
                        label += f" · {percent:.0f}%"
                    callback(percent, label)

            if not collected:
                raise ASRError(
                    "本地 ASR 没有识别到有效语音。",
                    hint="请确认视频包含人声，且音轨不是静音或纯音乐。",
                )
            text = segments_to_text(collected)
            language = str(getattr(info, "language", "") or "").strip() or self.config.language
            callback(100.0, f"语音识别完成 · {len(collected)} 段")
            return Transcript(
                text=text,
                segments=tuple(collected),
                language=language,
                backend=self.name,
                model=self.model,
            )
        except ASRError:
            raise
        except Exception as exc:
            detail = clean_error_message(exc)
            lowered = detail.lower()
            hint = (
                "若使用 CPU 或 CUDA 环境不完整，可设置 ASR_DEVICE=cpu、"
                "ASR_COMPUTE_TYPE=int8；也可以改用 ASR_BACKEND=openai。"
            )
            if any(word in lowered for word in ("cuda", "cudnn", "cublas", "gpu", "driver")):
                hint = (
                    "CUDA 环境不可用。请设置 ASR_DEVICE=cpu 和 ASR_COMPUTE_TYPE=int8，"
                    "或安装匹配的 CUDA/cuDNN 运行时。"
                )
            elif any(
                word in lowered
                for word in ("download", "http", "connection", "huggingface", "model")
            ):
                hint = (
                    "模型加载失败。首次运行需要联网下载模型；也可预先下载并把 ASR_MODEL "
                    "设为本地模型目录，或把 ASR_MODEL_DIR 指向可写的模型缓存目录。"
                )
            raise ASRError(f"本地 faster-whisper 识别失败：{detail}", hint=hint) from exc
        finally:
            # Release the model before the next backend attempt / temp cleanup.
            del model
