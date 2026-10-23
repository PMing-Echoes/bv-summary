from __future__ import annotations

import os
from pathlib import Path

import pytest

from bv_summary.cli import build_parser
from bv_summary.config import _user_config_candidates, load_environment, load_settings
from bv_summary.errors import ConfigError

from .helpers import clean_env, parse_cli_args


def test_cli_overrides_environment_and_config(tmp_path: Path) -> None:
    config_path = tmp_path / "test-config.toml"
    config_path.write_text(
        """
output = "from-config.txt"

[asr]
backend = "faster-whisper"
model = "from-config"
api_model = "whisper-from-config"

[llm]
backend = "openai"
model = "model-from-config"
temperature = 0.7
""".strip(),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "https://example.com/video",
            "--asr-model",
            "cli-model",
            "--model",
            "cli-llm",
        ]
    )
    env = {
        **clean_env(tmp_path),
        "ASR_MODEL": "environment-model",
        "LLM_MODEL": "environment-llm",
        "LLM_API_KEY": "sk-secret-should-not-leak",
        "LLM_BASE_URL": "https://llm.example.com/v1",
        "ASR_CPU_THREADS": "3",
        "LLM_TEMPERATURE": "0.2",
    }
    settings = load_settings(args, environ=env)

    assert settings.asr.model == "cli-model"  # CLI wins
    assert settings.llm.model == "cli-llm"
    assert settings.llm.temperature == 0.2  # env beats TOML
    assert settings.llm.api_key == "sk-secret-should-not-leak"
    assert settings.asr.backend == "faster-whisper"
    assert settings.asr.api_model == "whisper-from-config"
    assert settings.asr.cpu_threads == 3
    assert settings.output == Path("from-config.txt")
    assert settings.output_is_dir is False
    assert "sk-secret-should-not-leak" not in repr(settings)


def test_config_file_values_and_aliases(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
keep_audio = true
save_raw = true
force = true

[asr]
backend = "local"
compute_type = "INT8"
language = "ZH-CN"
chunk_seconds = 300

[llm]
backend = "disabled"
""".strip(),
        encoding="utf-8",
    )
    args = build_parser().parse_args(["--config", str(config_path), "https://example.com/video"])
    settings = load_settings(args, environ=clean_env(tmp_path))

    assert settings.keep_audio is True
    assert settings.save_raw is True
    assert settings.force is True
    assert settings.output is None
    assert settings.asr.backend == "faster-whisper"
    assert settings.asr.language == "zh"
    assert settings.asr.compute_type == "int8"
    assert settings.asr.chunk_seconds == 300
    assert settings.llm.backend == "none"


def test_openai_key_is_shared_fallback(tmp_path: Path) -> None:
    args = parse_cli_args(tmp_path)
    env = {
        **clean_env(tmp_path),
        "OPENAI_API_KEY": "sk-shared",
        "LLM_MODEL": "gpt-test",
        "ASR_BACKEND": "openai",
    }
    settings = load_settings(args, environ=env)
    assert settings.llm.api_key == "sk-shared"
    assert settings.asr.api_key == "sk-shared"
    assert settings.llm.model == "gpt-test"


def test_output_flag_file_vs_directory(tmp_path: Path) -> None:
    args = parse_cli_args(tmp_path, "--output", "transcript.txt")
    settings = load_settings(args, environ=clean_env(tmp_path))
    assert settings.output == Path("transcript.txt")
    assert settings.output_is_dir is False

    args = parse_cli_args(tmp_path, "--output", "no-extension")
    settings = load_settings(args, environ=clean_env(tmp_path))
    assert settings.output == Path("no-extension")
    assert settings.output_is_dir is True

    existing = tmp_path / "existing.dir"
    existing.mkdir()
    args = parse_cli_args(tmp_path, "--output", str(existing))
    settings = load_settings(args, environ=clean_env(tmp_path))
    assert settings.output_is_dir is True


def test_boolean_parser_accepts_common_values(tmp_path: Path) -> None:
    args = parse_cli_args(tmp_path, "--keep-audio", "--no-save-raw")
    env = {
        **clean_env(tmp_path),
        "BV_SUMMARY_KEEP_AUDIO": "false",
        "BV_SUMMARY_SAVE_RAW": "true",
        "BV_SUMMARY_FORCE": "yes",
    }
    settings = load_settings(args, environ=env)
    assert settings.keep_audio is True  # explicit CLI wins over env
    assert settings.save_raw is False
    assert settings.force is True


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"ASR_CPU_THREADS": "many"}, "ASR_CPU_THREADS"),
        ({"BV_SUMMARY_KEEP_AUDIO": "maybe"}, "BV_SUMMARY_KEEP_AUDIO"),
        ({"ASR_CHUNK_SECONDS": "1"}, "ASR_CHUNK_SECONDS"),
        ({"LLM_TEMPERATURE": "nope"}, "LLM_TEMPERATURE"),
    ],
)
def test_invalid_environment_values_raise(
    tmp_path: Path, env: dict[str, str], message: str
) -> None:
    args = parse_cli_args(tmp_path)
    with pytest.raises(ConfigError, match=message):
        load_settings(args, environ={**clean_env(tmp_path), **env})


def test_unknown_config_fields_are_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text('[llm]\nmodle = "typo"\n', encoding="utf-8")
    args = build_parser().parse_args(["--config", str(config_path), "https://example.com/video"])
    with pytest.raises(ConfigError, match="未知字段"):
        load_settings(args, environ=clean_env(tmp_path))


def test_invalid_toml_is_reported(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text("[llm\nbackend = 'auto'", encoding="utf-8")
    args = build_parser().parse_args(["--config", str(config_path), "https://example.com/video"])
    with pytest.raises(ConfigError, match="TOML"):
        load_settings(args, environ=clean_env(tmp_path))


def test_missing_explicit_config_raises(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--config", str(tmp_path / "missing.toml"), "https://example.com/video"]
    )
    with pytest.raises(ConfigError, match="不存在"):
        load_settings(args, environ=clean_env(tmp_path))


def test_load_environment_does_not_override_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BV_SUMMARY_OUTPUT", "already-set")
    env_file = tmp_path / ".env"
    env_file.write_text("BV_SUMMARY_OUTPUT=from-dotenv\n", encoding="utf-8")
    loaded = load_environment(str(env_file))
    assert loaded == env_file
    assert os.environ["BV_SUMMARY_OUTPUT"] == "already-set"


def test_missing_explicit_env_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=".env"):
        load_environment(str(tmp_path / "missing.env"))


def test_user_config_candidates_survive_missing_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_home() -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", no_home)
    candidates = _user_config_candidates({"APPDATA": str(tmp_path / "appdata")})
    assert candidates == [tmp_path / "appdata" / "bv-summary" / "config.toml"]
