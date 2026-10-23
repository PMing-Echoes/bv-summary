from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bv_summary import downloader
from bv_summary.config import DownloadConfig
from bv_summary.errors import ConfigError, DownloadError


def _config(**overrides: Any) -> DownloadConfig:
    values = {"cookies": None, "cookies_from_browser": None, "ffmpeg_location": None}
    values.update(overrides)
    return DownloadConfig(**values)


def test_parse_cookies_from_browser() -> None:
    assert downloader._parse_cookies_from_browser(None) is None
    assert downloader._parse_cookies_from_browser("chrome") == ("chrome", None, None, None)
    assert downloader._parse_cookies_from_browser("firefox:default") == (
        "firefox",
        "default",
        None,
        None,
    )
    with pytest.raises(ConfigError):
        downloader._parse_cookies_from_browser("chrome::work")
    with pytest.raises(ConfigError):
        downloader._parse_cookies_from_browser(":profile")


def test_download_media_with_fake_ytdlp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "download"
    work_dir.mkdir()
    media_path = work_dir / "demo.m4a"
    captured: dict[str, Any] = {}

    class FakeYoutubeDL:
        def __init__(self, options: dict[str, Any]) -> None:
            captured["options"] = options

        def __enter__(self) -> FakeYoutubeDL:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
            captured["url"] = url
            captured["download"] = download
            media_path.write_bytes(b"fake audio bytes")
            hook = captured["options"]["progress_hooks"][0]
            hook(
                {
                    "status": "downloading",
                    "downloaded_bytes": 50,
                    "total_bytes": 100,
                    "speed": 1000,
                    "eta": 2,
                }
            )
            hook({"status": "finished"})
            return {
                "id": "demo",
                "title": "Demo video",
                "webpage_url": "https://example.com/watch?v=demo",
                "duration": 12,
                "uploader": "Tester",
                "extractor_key": "Generic",
                "requested_downloads": [{"filepath": str(media_path)}],
            }

        def prepare_filename(self, info: dict[str, Any]) -> str:
            return str(media_path)

    monkeypatch.setattr(
        downloader,
        "_import_yt_dlp",
        lambda: SimpleNamespace(YoutubeDL=FakeYoutubeDL),
    )
    progress: list[tuple[float | None, str]] = []
    result = downloader.download_media(
        "https://example.com/watch?v=demo",
        work_dir,
        config=_config(cookies_from_browser="chrome"),
        ffmpeg_location="C:/ffmpeg/bin/ffmpeg.exe",
        progress=lambda percent, message: progress.append((percent, message)),
    )

    assert result.path == media_path.resolve()
    assert result.info.title == "Demo video"
    assert result.info.duration == 12
    assert result.info.uploader == "Tester"
    options = captured["options"]
    assert options["format"] == "bestaudio/best"
    assert options["noplaylist"] is True
    assert options["cookiesfrombrowser"] == ("chrome", None, None, None)
    assert options["ffmpeg_location"] == "C:/ffmpeg/bin/ffmpeg.exe"
    assert progress[-1] == (100.0, "音频下载完成")
    assert any(percent == 100.0 for percent, _message in progress)


def test_download_login_error_has_actionable_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeYoutubeDL:
        def __init__(self, options: dict[str, Any]) -> None:
            pass

        def __enter__(self) -> FakeYoutubeDL:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
            class DownloadFailure(Exception):
                pass

            raise DownloadFailure("ERROR: Sign in to confirm you're not a bot")

    monkeypatch.setattr(
        downloader,
        "_import_yt_dlp",
        lambda: SimpleNamespace(YoutubeDL=FakeYoutubeDL),
    )
    with pytest.raises(DownloadError) as caught:
        downloader.download_media(
            "https://www.youtube.com/watch?v=demo",
            tmp_path / "work",
            config=_config(),
        )
    assert "下载失败" in caught.value.message
    assert "cookies" in (caught.value.hint or "").lower()


def test_download_rejects_playlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeYoutubeDL:
        def __init__(self, options: dict[str, Any]) -> None:
            pass

        def __enter__(self) -> FakeYoutubeDL:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
            return {"entries": [{"id": "one"}], "title": "Playlist"}

    monkeypatch.setattr(
        downloader,
        "_import_yt_dlp",
        lambda: SimpleNamespace(YoutubeDL=FakeYoutubeDL),
    )
    with pytest.raises(DownloadError, match="播放列表"):
        downloader.download_media(
            "https://example.com/playlist",
            tmp_path / "work",
            config=_config(),
        )


def test_missing_cookies_file_is_rejected(tmp_path: Path) -> None:
    from bv_summary.cli import build_parser
    from bv_summary.config import load_settings

    config_path = tmp_path / "empty.toml"
    config_path.write_text("", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--cookies",
            str(tmp_path / "none.txt"),
            "https://example.com/v",
        ]
    )
    with pytest.raises(ConfigError, match="cookies"):
        load_settings(args, environ={"APPDATA": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path)})
