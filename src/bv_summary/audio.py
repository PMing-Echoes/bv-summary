"""ffmpeg discovery and conversion to 16 kHz mono PCM WAV."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from .errors import AudioError, DependencyError
from .models import AudioFile, ProgressCallback
from .utils import clean_error_message, human_duration


def _verify_ffmpeg(path: Path) -> None:
    try:
        result = subprocess.run(
            [str(path), "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DependencyError(
            f"无法运行 ffmpeg：{path}",
            hint=f"请检查路径与文件权限。底层错误：{clean_error_message(exc)}",
        ) from exc
    if result.returncode != 0:
        detail = clean_error_message(result.stderr or result.stdout)
        raise DependencyError(f"ffmpeg 无法正常运行：{path}\n{detail}")


def resolve_ffmpeg(location: str | None = None) -> Path:
    """Find a usable ffmpeg executable.

    Priority: explicit path/command, PATH, then the imageio-ffmpeg wheel. The
    wheel fallback keeps the CLI usable on machines without a system ffmpeg.
    """
    candidates: list[Path] = []
    if location:
        raw = Path(location).expanduser()
        if raw.is_file():
            candidates.append(raw)
        elif raw.is_dir():
            names = ("ffmpeg.exe", "ffmpeg") if os.name == "nt" else ("ffmpeg", "ffmpeg.exe")
            for base in (raw, raw / "bin"):
                candidates.extend(base / name for name in names if (base / name).is_file())
        else:
            found = shutil.which(location)
            if found:
                candidates.append(Path(found))
        if not candidates:
            raise DependencyError(
                f"找不到指定的 ffmpeg：{location}",
                hint="请提供 ffmpeg 可执行文件的完整路径，或包含 ffmpeg 的目录。",
            )
    else:
        found = shutil.which("ffmpeg")
        if found:
            candidates.append(Path(found))

    if not candidates and not location:
        try:
            import imageio_ffmpeg

            candidates.append(Path(imageio_ffmpeg.get_ffmpeg_exe()))
        except Exception:
            pass

    if not candidates:
        raise DependencyError(
            "找不到 ffmpeg，无法提取音频。",
            hint=(
                "安装 ffmpeg 并加入 PATH，或安装带有 ffmpeg 的 Python 包："
                "pip install imageio-ffmpeg；也可以用 --ffmpeg-location 指定路径。"
            ),
        )

    ffmpeg = candidates[0].resolve()
    _verify_ffmpeg(ffmpeg)
    return ffmpeg


def _read_audio_info(path: Path) -> AudioFile:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
    except (OSError, wave.Error) as exc:
        raise AudioError(f"提取出的音频不是有效的 WAV 文件：{path}", hint=str(exc)) from exc

    if channels != 1 or sample_rate != 16_000 or sample_width != 2:
        raise AudioError(
            "提取出的音频格式不符合预期（需要 16 kHz、单声道、16-bit PCM）。",
            hint=f"实际格式：{sample_rate} Hz、{channels} 声道、{sample_width * 8}-bit。",
        )
    duration = frames / sample_rate if sample_rate else 0.0
    if frames <= 0 or duration <= 0:
        raise AudioError("提取出的音频为空，视频中可能没有可用的音轨。")
    return AudioFile(
        path=path,
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
        size_bytes=path.stat().st_size,
    )


def extract_audio(
    input_path: Path,
    output_path: Path,
    *,
    ffmpeg: Path,
    video_duration: float | None = None,
    progress: ProgressCallback | None = None,
) -> AudioFile:
    """Convert the first audio stream to 16 kHz mono WAV and report progress."""
    if not input_path.is_file():
        raise AudioError(f"找不到已下载的媒体文件：{input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.unlink(missing_ok=True)
    except OSError as exc:
        raise AudioError(f"无法清理旧的音频文件：{output_path}", hint=str(exc)) from exc

    callback = progress or (lambda _percent, _message: None)
    callback(None, "准备提取音频…")
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-progress",
        "pipe:1",
        "-nostats",
        str(output_path),
    ]

    last_update = 0.0
    stderr_text = ""
    failed = False
    with tempfile.TemporaryFile(mode="w+b") as stderr_file:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise AudioError(
                f"无法启动 ffmpeg：{ffmpeg}",
                hint=clean_error_message(exc),
            ) from exc

        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(("out_time_us=", "out_time_ms=")):
                    raw_value = line.split("=", 1)[1]
                    try:
                        # ffmpeg's out_time_ms is actually microseconds, and
                        # out_time_us is available in modern builds. Handle both.
                        elapsed = int(raw_value) / 1_000_000.0
                    except ValueError:
                        continue
                    now = time.monotonic()
                    if now - last_update >= 0.2:
                        last_update = now
                        if video_duration and video_duration > 0:
                            percent = min(99.0, elapsed / video_duration * 100.0)
                            callback(
                                percent,
                                f"提取音频 · {human_duration(elapsed)} / {human_duration(video_duration)}",
                            )
                        else:
                            callback(None, f"提取音频 · 已处理 {human_duration(elapsed)}")
                elif line == "progress=end":
                    callback(None, "封装 WAV 文件…")
            return_code = process.wait()
        except KeyboardInterrupt:
            failed = True
            process.kill()
            process.wait()
            raise
        except BaseException:
            failed = True
            process.kill()
            process.wait()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            stderr_file.seek(0)
            stderr_text = stderr_file.read().decode("utf-8", errors="replace")

    if return_code != 0 or failed:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        detail = clean_error_message(stderr_text)
        raise AudioError(
            f"ffmpeg 音频提取失败（退出码 {return_code}）。",
            hint=detail,
        )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        detail = clean_error_message(stderr_text) if stderr_text.strip() else "视频可能没有音轨。"
        raise AudioError("ffmpeg 未生成有效的音频文件。", hint=detail)

    audio = _read_audio_info(output_path)
    callback(100.0, "音频提取完成")
    return audio
