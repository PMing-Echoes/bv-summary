"""Output path selection and safe file writes."""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import Settings
from .errors import OutputError
from .utils import sanitize_filename, unique_path, write_text_atomic


def plan_output_path(
    settings: Settings, title: str, *, cwd: Path | None = None
) -> tuple[Path, Path]:
    """Return (chosen path, originally requested path).

    A file path is used as-is; a directory path receives a sanitized title.txt.
    Existing files get a numeric suffix unless --force is set.
    """
    filename = f"{sanitize_filename(title)}.txt"
    base = cwd or Path.cwd()
    if settings.output is None:
        requested = base / filename
    elif settings.output_is_dir:
        requested = settings.output / filename
    else:
        requested = settings.output
    try:
        requested = requested.expanduser().resolve(strict=False)
        requested.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as exc:
        raise OutputError(f"无法创建输出目录：{requested.parent}", hint=str(exc)) from exc
    try:
        chosen = unique_path(requested, force=settings.force)
    except OSError as exc:
        raise OutputError(f"无法确定输出文件名：{requested}", hint=str(exc)) from exc
    return chosen, requested


def raw_sidecar(output_path: Path) -> Path:
    return output_path.with_suffix(".raw.txt")


def audio_sidecar(output_path: Path) -> Path:
    candidate = output_path.with_suffix(".wav")
    if candidate == output_path:
        candidate = output_path.with_name(f"{output_path.stem}.audio.wav")
    return candidate


def choose_sidecar(path: Path, *, force: bool) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return unique_path(path, force=force)
    except OSError as exc:
        raise OutputError(f"无法创建输出路径：{path}", hint=str(exc)) from exc


def save_text(path: Path, text: str) -> Path:
    try:
        write_text_atomic(path, text)
    except (OSError, UnicodeError) as exc:
        raise OutputError(f"写入文本文件失败：{path}", hint=str(exc)) from exc
    return path


def copy_audio(source: Path, destination: Path) -> Path:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    except OSError as exc:
        raise OutputError(f"保存音频文件失败：{destination}", hint=str(exc)) from exc
    return destination
