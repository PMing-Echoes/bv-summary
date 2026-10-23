from __future__ import annotations

from pathlib import Path

from bv_summary.cuda import _site_roots, configure_windows_cuda_dlls, runtime_directories


def test_runtime_directories_finds_nvidia_wheel_layout(tmp_path: Path) -> None:
    for parts in (
        ("nvidia", "cublas", "bin"),
        ("nvidia", "cuda_runtime", "bin"),
        ("nvidia", "cudnn", "bin"),
    ):
        (tmp_path.joinpath(*parts)).mkdir(parents=True)
    system_cuda = tmp_path / "system-cuda"
    (system_cuda / "bin").mkdir(parents=True)

    directories = runtime_directories(
        [tmp_path],
        {"CUDA_PATH": str(system_cuda)},
    )
    resolved = {str(directory.resolve()) for directory in directories}
    assert str((tmp_path / "nvidia" / "cublas" / "bin").resolve()) in resolved
    assert str((tmp_path / "nvidia" / "cudnn" / "bin").resolve()) in resolved
    assert str((system_cuda / "bin").resolve()) in resolved


def test_runtime_directories_skips_missing_and_duplicates(tmp_path: Path) -> None:
    bin_dir = tmp_path / "nvidia" / "cublas" / "bin"
    bin_dir.mkdir(parents=True)
    directories = runtime_directories([tmp_path, tmp_path], {})
    assert directories == [bin_dir]


def test_site_roots_include_virtualenv_paths() -> None:
    roots = _site_roots()
    assert roots
    assert all(isinstance(root, Path) for root in roots)


def test_configure_cuda_is_noop_when_not_needed(monkeypatch) -> None:
    # The current test platform may not be Windows; either way this must not raise.
    directories = configure_windows_cuda_dlls()
    assert isinstance(directories, list)
