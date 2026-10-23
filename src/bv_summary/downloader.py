"""yt-dlp integration: fetch the best audio stream into a working directory."""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

from .config import DownloadConfig
from .errors import ConfigError, DependencyError, DownloadError
from .models import MediaFile, ProgressCallback, VideoInfo
from .utils import clean_error_message, format_eta, human_bytes

_TEMP_SUFFIXES = (".part", ".ytdl", ".tmp")


class _YtDlpLogger:
    """Keep yt-dlp's console quiet while retaining useful error context."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def debug(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        cleaned = clean_error_message(message, limit=500)
        if cleaned and cleaned not in self.warnings:
            self.warnings.append(cleaned)

    def error(self, message: str) -> None:
        cleaned = clean_error_message(message, limit=1000)
        if cleaned:
            self.errors.append(cleaned)


def _import_yt_dlp() -> Any:
    try:
        return importlib.import_module("yt_dlp")
    except ImportError as exc:
        raise DependencyError(
            "缺少 yt-dlp，无法下载视频。",
            hint='请执行：pip install -e ".[local-asr]"（或 pip install yt-dlp）',
        ) from exc


def _parse_cookies_from_browser(value: str | None) -> tuple[Any, ...] | None:
    """Parse the commonly used BROWSER[+KEYRING][:PROFILE] subset."""
    if not value:
        return None
    spec = value.strip()
    if "::" in spec:
        # Full container syntax varies across yt-dlp versions; point users at
        # cookies.txt rather than silently passing something incompatible.
        raise ConfigError(
            "cookies_from_browser 不支持 ::CONTAINER 语法。",
            hint="请使用 --cookies cookies.txt 或 YTDLP_COOKIES 指定 Netscape cookies 文件。",
        )
    head, separator, profile = spec.partition(":")
    browser, plus, keyring = head.partition("+")
    browser = browser.strip().lower()
    if not browser:
        raise ConfigError("cookies_from_browser 缺少浏览器名称，例如 chrome 或 firefox。")
    if separator and ":" in profile:
        raise ConfigError("cookies_from_browser 的 PROFILE 中不能包含冒号。")
    # yt-dlp's library option is (browser, profile, keyring, container).
    return (browser, profile.strip() or None, keyring.strip() or None, None)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _download_hint(message: str) -> str:
    lowered = message.lower()
    if any(
        word in lowered
        for word in ("sign in", "login", "log in", "cookie", "private video", "members-only")
    ):
        return (
            "该站点可能要求登录。可尝试 --cookies-from-browser chrome 或 "
            "--cookies cookies.txt（YouTube/Bilibili 均适用），并确认你有权下载该内容。"
        )
    if "unsupported url" in lowered:
        return "yt-dlp 可能尚未支持该站点或链接格式。请先升级 yt-dlp，再确认链接可直接播放。"
    if "ffmpeg" in lowered or "postprocessing" in lowered:
        return "请安装 ffmpeg，或通过 --ffmpeg-location / FFMPEG_BINARY 指定可执行文件。"
    if any(
        word in lowered
        for word in ("403", "forbidden", "geo", "region", "not available in your country")
    ):
        return "该内容可能有地区、登录或防盗链限制。可尝试登录 cookies、代理网络，或在浏览器中确认链接可用。"
    if any(
        word in lowered
        for word in ("timed out", "connection", "network", "temporary failure", "http error 5")
    ):
        return "请检查网络连接、代理设置，或稍后重试。也可以在浏览器中先确认该链接可访问。"
    return "请检查链接是否有效且可公开访问，或更新 yt-dlp 后重试。"


def _select_downloaded_file(info: dict[str, Any], work_dir: Path, ydl: Any) -> Path:
    candidates: list[Path] = []
    for requested in info.get("requested_downloads") or []:
        if isinstance(requested, dict):
            for key in ("filepath", "_filename", "filename"):
                value = requested.get(key)
                if value:
                    candidates.append(Path(value))
    for key in ("filepath", "_filename"):
        value = info.get(key)
        if value:
            candidates.append(Path(value))
    try:
        prepared = ydl.prepare_filename(info)
        if prepared:
            candidates.append(Path(prepared))
    except Exception:
        pass

    checked: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = work_dir / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in checked:
            continue
        checked.add(resolved)
        if resolved.is_file() and resolved.stat().st_size > 0:
            return resolved

    # Some extractors put the final path in an unexpected place. The work
    # directory is private to one download, so fall back to its largest media
    # file while excluding yt-dlp's temporary sidecars.
    files = [
        path
        for path in work_dir.rglob("*")
        if path.is_file()
        and not path.name.endswith(_TEMP_SUFFIXES)
        and not any(part.startswith(".") for part in path.relative_to(work_dir).parts)
    ]
    if files:
        return max(files, key=lambda path: path.stat().st_size).resolve()
    raise DownloadError("下载似乎已完成，但找不到已下载的媒体文件。")


def download_media(
    url: str,
    work_dir: Path,
    *,
    config: DownloadConfig,
    ffmpeg_location: str | None = None,
    progress: ProgressCallback | None = None,
) -> MediaFile:
    """Download the best audio stream for *url* and return media metadata."""
    yt_dlp = _import_yt_dlp()
    callback = progress or (lambda _percent, _message: None)
    work_dir.mkdir(parents=True, exist_ok=True)
    ydl_logger = _YtDlpLogger()
    callback(None, "正在解析视频信息…")

    last_update = 0.0

    def progress_hook(data: dict[str, Any]) -> None:
        nonlocal last_update
        status = data.get("status")
        if status == "downloading":
            now = time.monotonic()
            if now - last_update < 0.15:
                return
            last_update = now
            downloaded = _number(data.get("downloaded_bytes")) or 0.0
            total = _number(data.get("total_bytes")) or _number(data.get("total_bytes_estimate"))
            seconds = _number(data.get("eta"))
            speed = _number(data.get("speed"))
            if total and total > 0:
                percent = min(99.0, downloaded / total * 100.0)
                parts = [f"下载音频 · {human_bytes(downloaded)} / {human_bytes(total)}"]
            else:
                percent = None
                parts = [f"下载音频 · 已接收 {human_bytes(downloaded)}"]
            if speed:
                parts.append(f"{human_bytes(speed)}/s")
            if seconds is not None:
                parts.append(f"剩余 {format_eta(seconds)}")
            callback(percent, " · ".join(parts))
        elif status == "finished":
            callback(100.0, "下载完成，正在整理媒体文件…")
        elif status == "error":
            callback(None, "下载出错")

    options: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(work_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": ydl_logger,
        "progress_hooks": [progress_hook],
        "retries": 3,
        "fragment_retries": 3,
        "extractor_retries": 2,
        "socket_timeout": 30,
        "overwrites": False,
        "continuedl": True,
        "restrictfilenames": True,
        "windowsfilenames": True,
    }
    if config.cookies is not None:
        options["cookiefile"] = str(config.cookies)
    browser_cookies = _parse_cookies_from_browser(config.cookies_from_browser)
    if browser_cookies is not None:
        options["cookiesfrombrowser"] = browser_cookies
    if ffmpeg_location:
        options["ffmpeg_location"] = ffmpeg_location

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            if not isinstance(info, dict):
                raise DownloadError("下载器没有返回视频信息。")
            if info.get("entries") is not None:
                raise DownloadError(
                    "检测到播放列表或多个条目。本工具一次只处理一个视频。",
                    hint="请粘贴具体视频页面的链接，而不是播放列表、频道或合集链接。",
                )
            media_path = _select_downloaded_file(info, work_dir, ydl)
    except DownloadError:
        raise
    except Exception as exc:
        # yt-dlp raises DownloadError for most extraction failures; this catches
        # import/extractor edge cases and turns them into readable messages.
        error_type = type(exc).__name__
        if "DownloadError" in error_type:
            detail = clean_error_message(exc)
        else:
            detail = clean_error_message(f"{error_type}: {exc}")
        if ydl_logger.errors:
            detail = f"{detail}\n{ydl_logger.errors[-1]}"
        raise DownloadError(
            f"下载失败：{detail}",
            hint=_download_hint(detail),
        ) from exc

    title = str(info.get("title") or "").strip() or f"video-{info.get('id') or 'unknown'}"
    webpage_url = str(info.get("webpage_url") or info.get("original_url") or url)
    uploader_value = info.get("uploader") or info.get("channel") or info.get("creator")
    video_info = VideoInfo(
        title=title,
        video_id=str(info.get("id") or "unknown"),
        webpage_url=webpage_url,
        duration=_number(info.get("duration")),
        uploader=str(uploader_value).strip() if uploader_value else None,
        extractor=str(info.get("extractor_key") or info.get("extractor") or "").strip() or None,
    )
    callback(100.0, "音频下载完成")
    return MediaFile(path=media_path, info=video_info)
