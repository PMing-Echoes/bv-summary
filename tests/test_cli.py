from __future__ import annotations

from pathlib import Path

import pytest

from bv_summary import cli
from bv_summary.errors import ConfigError
from bv_summary.models import RunResult


def _result(tmp_path: Path) -> RunResult:
    return RunResult(
        title="Demo",
        source_url="https://example.com/video",
        uploader="tester",
        duration=2.0,
        language="zh",
        asr_backend="fake-asr",
        asr_model="tiny",
        segment_count=1,
        raw_chars=4,
        llm_backend="fake-llm",
        llm_model="model",
        llm_chunks=1,
        text="纠正后的文本。",
        text_corrected=True,
        output_path=tmp_path / "out.txt",
        raw_path=None,
        audio_path=None,
        elapsed_seconds=1.0,
    )


def test_missing_url_exits_with_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main([])
    assert caught.value.code == 2
    assert "--help" in capsys.readouterr().err


def test_config_error_returns_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_environment", lambda path: None)

    def fail_settings(args):
        raise ConfigError("config exploded", hint="fix it")

    monkeypatch.setattr(cli, "load_settings", fail_settings)
    code = cli.main(["https://example.com/video"])
    captured = capsys.readouterr()
    assert code == 2
    assert "config exploded" in captured.err
    assert "fix it" in captured.err


def test_successful_cli_prints_summary_and_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_environment", lambda path: None)
    monkeypatch.setattr(cli, "load_settings", lambda args: object())

    class FakePipeline:
        def __init__(self, settings, reporter) -> None:
            pass

        def run(self, url: str) -> RunResult:
            return _result(tmp_path)

    monkeypatch.setattr(cli, "Pipeline", FakePipeline)
    code = cli.main(["https://example.com/video"])
    captured = capsys.readouterr()
    assert code == 0
    assert "完成" in captured.out
    assert "纠正后的文本。" in captured.out
    assert "out.txt" in captured.out


def test_no_print_text_hides_full_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_environment", lambda path: None)
    monkeypatch.setattr(cli, "load_settings", lambda args: object())

    class FakePipeline:
        def __init__(self, settings, reporter) -> None:
            pass

        def run(self, url: str) -> RunResult:
            return _result(tmp_path)

    monkeypatch.setattr(cli, "Pipeline", FakePipeline)
    assert cli.main(["https://example.com/video", "--no-print-text"]) == 0
    captured = capsys.readouterr()
    assert "完成" in captured.out
    assert "纠正后的文本。" not in captured.out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["--version"])
    assert caught.value.code == 0
    assert "bv-summary" in capsys.readouterr().out
