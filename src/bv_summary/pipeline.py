"""End-to-end orchestration: download, extract, transcribe, correct, save."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from .asr import ASRBackend, build_asr_selection
from .audio import extract_audio, resolve_ffmpeg
from .config import Settings
from .downloader import download_media
from .errors import AppError, ASRError, AudioError, LLMError, UserInputError
from .llm import LLMBackend, LLMSelection, build_llm_selection
from .models import AudioFile, Correction, RunResult, Transcript
from .output import (
    audio_sidecar,
    choose_sidecar,
    copy_audio,
    plan_output_path,
    raw_sidecar,
    save_text,
)
from .ui import Reporter
from .utils import human_bytes, human_duration, validate_http_url


class Pipeline:
    def __init__(self, settings: Settings, reporter: Reporter) -> None:
        self.settings = settings
        self.reporter = reporter

    def _announce_plan(self, asr: tuple[ASRBackend, ...], llm: LLMSelection) -> None:
        asr_chain = " → ".join(f"{backend.name}（{backend.model}）" for backend in asr)
        if llm.backends:
            llm_chain = " → ".join(f"{backend.name}（{backend.model}）" for backend in llm.backends)
        else:
            llm_chain = "未启用"
        self.reporter.info(f"ASR 后端: {asr_chain}")
        self.reporter.info(f"LLM 后端: {llm_chain}")
        for note in llm.notes:
            self.reporter.info(note)

    @staticmethod
    def _brief(error: AppError) -> str:
        first = (
            error.message.strip().splitlines()[0] if error.message.strip() else type(error).__name__
        )
        return first[:500]

    def _transcribe(self, backends: tuple[ASRBackend, ...], audio: AudioFile) -> Transcript:
        failures: list[tuple[ASRBackend, AppError]] = []
        for index, backend in enumerate(backends):
            try:
                with self.reporter.progress(f"语音识别 · {backend.name} / {backend.model}") as bar:
                    result = backend.transcribe(audio, progress=bar.update)
                self.reporter.info(
                    f"识别完成：{result.backend} / {result.model} · "
                    f"{len(result.segments)} 段 · 语言 {result.language or '未知'}"
                )
                return result
            except AppError as exc:
                failures.append((backend, exc))
                if index + 1 < len(backends):
                    self.reporter.warning(
                        f"{backend.name} 识别失败：{self._brief(exc)}；正在尝试下一个 ASR 后端…"
                    )

        if len(failures) == 1:
            raise failures[0][1]
        lines: list[str] = []
        for backend, error in failures:
            lines.append(f"- {backend.name} / {backend.model}: {error.message}")
            if error.hint:
                lines.append(f"  {error.hint}")
        raise ASRError(
            "所有 ASR 后端均失败，未能生成转写。",
            hint="\n".join(lines),
        ) from (failures[-1][1] if failures else None)

    def _ensure_raw_file(
        self,
        output_path: Path,
        raw_path: Path | None,
        raw_text: str,
    ) -> Path:
        if raw_path is not None:
            return raw_path
        path = choose_sidecar(raw_sidecar(output_path), force=self.settings.force)
        save_text(path, raw_text)
        return path

    def _correct(
        self,
        backends: tuple[LLMBackend, ...],
        raw_text: str,
    ) -> Correction:
        failures: list[tuple[LLMBackend, AppError]] = []
        for index, backend in enumerate(backends):
            try:
                with self.reporter.progress(f"LLM 纠错 · {backend.name} / {backend.model}") as bar:
                    result = backend.correct(raw_text, progress=bar.update)
                self.reporter.info(
                    f"纠错完成：{result.backend} / {result.model} · {result.chunks} 块"
                )
                return result
            except AppError as exc:
                failures.append((backend, exc))
                if index + 1 < len(backends):
                    self.reporter.warning(
                        f"{backend.name} 纠错失败：{self._brief(exc)}；正在尝试下一个 LLM 后端…"
                    )

        lines: list[str] = []
        for backend, error in failures:
            lines.append(f"- {backend.name} / {backend.model}: {error.message}")
            if error.hint:
                lines.append(f"  {error.hint}")
        raise LLMError(
            "所有 LLM 后端均失败，未能生成纠错文本。",
            hint="\n".join(lines) if lines else None,
        ) from (failures[-1][1] if failures else None)

    def run(self, url: str) -> RunResult:
        started = time.monotonic()
        try:
            validated_url = validate_http_url(url)
        except ValueError as exc:
            raise UserInputError(
                str(exc), hint='示例：bv-summary "https://www.bilibili.com/video/BV..."'
            ) from exc

        ffmpeg = resolve_ffmpeg(self.settings.download.ffmpeg_location)
        asr_selection = build_asr_selection(self.settings.asr)
        if self.settings.llm.backend == "none":
            llm_selection = LLMSelection(())
        else:
            llm_selection = build_llm_selection(self.settings.llm)
        self._announce_plan(asr_selection.backends, llm_selection)
        for note in asr_selection.notes:
            self.reporter.info(note)

        try:
            temp_context = tempfile.TemporaryDirectory(prefix="bv-summary-")
        except OSError as exc:
            raise AudioError("无法创建临时工作目录。", hint=str(exc)) from exc

        with temp_context as temp_root:
            work_dir = Path(temp_root)
            self.reporter.info("[1/4] 获取视频信息并下载音频…")
            with self.reporter.progress("下载音频") as bar:
                media = download_media(
                    validated_url,
                    work_dir / "download",
                    config=self.settings.download,
                    ffmpeg_location=str(ffmpeg),
                    progress=bar.update,
                )

            self.reporter.info(f"标题: {media.info.title}")
            self.reporter.info(
                f"时长: {human_duration(media.info.duration)} · 平台: {media.info.extractor or '未知'}"
            )
            output_path, requested_path = plan_output_path(self.settings, media.info.title)
            if output_path != requested_path:
                self.reporter.info(f"目标文件已存在，改写为: {output_path}")

            self.reporter.info("[2/4] 提取 16 kHz 单声道 WAV 音频…")
            with self.reporter.progress("提取音频") as bar:
                audio = extract_audio(
                    media.path,
                    work_dir / "audio.wav",
                    ffmpeg=ffmpeg,
                    video_duration=media.info.duration,
                    progress=bar.update,
                )
            self.reporter.info(
                f"音频已就绪: {human_duration(audio.duration)} · "
                f"{audio.sample_rate} Hz · {human_bytes(audio.size_bytes)}"
            )

            audio_path: Path | None = None
            if self.settings.keep_audio:
                audio_path = choose_sidecar(audio_sidecar(output_path), force=self.settings.force)
                copy_audio(audio.path, audio_path)
                self.reporter.info(f"保留音频: {audio_path}")

            raw_path: Path | None = None
            if self.settings.save_raw and llm_selection.backends:
                raw_path = choose_sidecar(raw_sidecar(output_path), force=self.settings.force)
                self.reporter.info(f"原始转写将保存到: {raw_path}")

            self.reporter.info("[3/4] 语音转文字…")
            transcript = self._transcribe(asr_selection.backends, audio)
            raw_text = transcript.text.strip()
            if not raw_text:
                raise ASRError("语音识别结果为空，无法继续。")
            if raw_path is not None:
                save_text(raw_path, raw_text)

            correction: Correction | None = None
            if llm_selection.backends:
                self.reporter.info("[4/4] LLM 文本纠错…")
                try:
                    correction = self._correct(llm_selection.backends, raw_text)
                except LLMError:
                    raw_path = self._ensure_raw_file(output_path, raw_path, raw_text)
                    self.reporter.warning(f"LLM 纠错失败，原始转写已保留: {raw_path}")
                    raise
                final_text = correction.text
            else:
                self.reporter.info("[4/4] 未启用 LLM 纠错，最终文本为原始转写。")
                final_text = raw_text

            save_text(output_path, final_text)
            self.reporter.info(f"输出文件已写入: {output_path}")

        return RunResult(
            title=media.info.title,
            source_url=media.info.webpage_url,
            uploader=media.info.uploader,
            duration=audio.duration,
            language=transcript.language,
            asr_backend=transcript.backend,
            asr_model=transcript.model,
            segment_count=len(transcript.segments),
            raw_chars=len(raw_text),
            llm_backend=correction.backend if correction else None,
            llm_model=correction.model if correction else None,
            llm_chunks=correction.chunks if correction else None,
            text=final_text,
            text_corrected=correction is not None,
            output_path=output_path,
            raw_path=raw_path,
            audio_path=audio_path,
            elapsed_seconds=time.monotonic() - started,
        )
