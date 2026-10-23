"""Windows CUDA DLL discovery for faster-whisper / CTranslate2.

NVIDIA's Windows wheels install runtime DLLs under ``site-packages/nvidia``
rather than adding them to the system PATH. CTranslate2 loads cuBLAS at runtime,
so those directories must be registered before importing CTranslate2.
"""

from __future__ import annotations

import os
import site
import sysconfig
from collections.abc import Iterable, Mapping
from pathlib import Path

_DLL_SUBDIRS = (
    ("nvidia", "cublas", "bin"),
    ("nvidia", "cuda_runtime", "bin"),
    ("nvidia", "cudnn", "bin"),
    ("nvidia", "cuda_nvrtc", "bin"),
)
_dll_handles: list[object] = []


def _site_roots() -> list[Path]:
    roots: set[Path] = set()
    for key in ("purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if value:
            roots.add(Path(value))
    try:
        roots.update(Path(path) for path in site.getsitepackages())
    except (AttributeError, OSError):
        pass
    try:
        roots.add(Path(site.getusersitepackages()))
    except (AttributeError, OSError):
        pass
    return sorted(roots, key=lambda item: str(item).lower())


def runtime_directories(
    roots: Iterable[Path],
    environ: Mapping[str, str] | None = None,
) -> list[Path]:
    """Return existing NVIDIA DLL directories without touching the process."""
    env = os.environ if environ is None else environ
    directories: list[Path] = []
    seen: set[str] = set()

    for root in roots:
        for parts in _DLL_SUBDIRS:
            candidate = Path(root).joinpath(*parts)
            if not candidate.is_dir():
                continue
            key = str(candidate.resolve()).lower()
            if key not in seen:
                seen.add(key)
                directories.append(candidate)

    for env_name, subdir in (("CUDA_PATH", "bin"), ("CUDNN_PATH", "bin")):
        value = env.get(env_name)
        if not value:
            continue
        candidate = Path(value) / subdir
        if not candidate.is_dir():
            continue
        key = str(candidate.resolve()).lower()
        if key not in seen:
            seen.add(key)
            directories.append(candidate)
    return directories


def configure_windows_cuda_dlls() -> list[Path]:
    """Register NVIDIA wheel DLL directories with Windows and PATH.

    Safe to call repeatedly and a no-op outside Windows. Returns the registered
    directories, which is useful for diagnostics.
    """
    if os.name != "nt":
        return []
    directories = runtime_directories(_site_roots())
    if not directories:
        return []

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    normalized = {entry.rstrip("\\/").lower() for entry in path_entries if entry}
    for directory in directories:
        value = str(directory)
        if value.rstrip("\\/").lower() not in normalized:
            # PATH is important for transitive DLL dependencies on some builds.
            os.environ["PATH"] = value + os.pathsep + os.environ.get("PATH", "")
            normalized.add(value.rstrip("\\/").lower())
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if callable(add_dll_directory):
            try:
                _dll_handles.append(add_dll_directory(value))
            except (FileNotFoundError, OSError):
                pass
    return directories
