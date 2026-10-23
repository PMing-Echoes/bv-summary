"""Small dependency-free helpers shared by the modules."""

from __future__ import annotations

import ipaddress
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_INVALID_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def human_bytes(value: float | int | None) -> str:
    if value is None:
        return "?"
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TiB"


def human_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "未知"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_eta(seconds: float | int | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def clean_error_message(value: object, *, limit: int = 3000) -> str:
    """Make third-party error text fit for one readable CLI error block."""
    text = strip_ansi(str(value)).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^(?:\s*ERROR:\s*)+", "", text, flags=re.IGNORECASE)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text or "未知错误（第三方库未提供详细信息）"


def sanitize_filename(title: str, *, fallback: str = "transcript", max_length: int = 100) -> str:
    """Return a portable single-component filename (without extension)."""
    normalized = unicodedata.normalize("NFC", title or "")
    normalized = _INVALID_FILENAME_RE.sub("_", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .")
    if not normalized:
        return fallback
    if normalized.upper() in _WINDOWS_RESERVED:
        normalized = f"_{normalized}"
    if len(normalized) > max_length:
        normalized = normalized[:max_length].rstrip(" .")
    return normalized or fallback


def unique_path(path: Path, *, force: bool = False) -> Path:
    """Avoid clobbering existing files unless force is set."""
    if force or not path.exists():
        return path
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem} ({index}){suffix}")
        if not candidate.exists():
            return candidate
    raise OSError(f"无法为输出文件找到可用名称: {path}")


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text with a trailing newline, replacing atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = text.rstrip() + "\n"
    fd = -1
    temp_name: str | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            fd = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if fd >= 0:
            os.close(fd)
        if temp_name:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def is_local_http_url(url: str | None) -> bool:
    """Whether a base URL points at loopback/private infrastructure.

    Keyless OpenAI-compatible servers are common on localhost, Docker host
    aliases and private LANs. This check is only used to decide whether an API
    key is *required*; TLS and service authentication remain the user's job.
    """
    if not url:
        return False
    try:
        host = (urlsplit(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return False
    if not host:
        return False
    if host in {
        "localhost",
        "localhost.localdomain",
        "host.docker.internal",
        "host.containers.internal",
    }:
        return True
    if host.endswith((".localhost", ".local", ".internal")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # Bare single-label hostnames are normally local.
        return "." not in host
    return address.is_loopback or address.is_private or address.is_link_local


def endpoint_display(url: str | None) -> str:
    """Hide URL credentials/query before showing an endpoint in terminal text."""
    if not url:
        return "OpenAI 默认地址"
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<invalid URL>"
    if not parts.scheme or not parts.netloc:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    path = parts.path.rstrip("/")
    return f"{parts.scheme}://{host}{path}"


def truncate_text(text: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def validate_http_url(url: str) -> str:
    """Validate and lightly normalize a user-supplied media URL."""
    value = (url or "").strip()
    if not value:
        raise ValueError("链接不能为空")
    try:
        parts = urlsplit(value)
    except ValueError as exc:
        raise ValueError(f"链接格式无效: {exc}") from exc
    if parts.scheme.lower() not in {"http", "https"}:
        raise ValueError("仅支持 http:// 或 https:// 视频链接")
    if not parts.netloc:
        raise ValueError("链接缺少域名")
    return value
