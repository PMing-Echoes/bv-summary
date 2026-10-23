"""Terminal presentation. Progress goes to stderr; result text goes to stdout."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from .models import RunResult
from .utils import human_duration


class _ProgressHandle:
    def __init__(self, progress: Progress | None, task_id: int | None) -> None:
        self._progress = progress
        self._task_id = task_id

    def update(self, percent: float | None = None, description: str | None = None) -> None:
        if self._progress is None or self._task_id is None:
            return
        update: dict[str, object] = {}
        if percent is not None:
            update["completed"] = max(0.0, min(100.0, float(percent)))
        if description:
            update["description"] = escape(description)
        if update:
            self._progress.update(self._task_id, **update)


class Reporter:
    """Human-facing output with optional Rich progress bars."""

    def __init__(self, *, quiet: bool = False, no_color: bool = False) -> None:
        self.quiet = quiet
        self.console = Console(highlight=False, no_color=no_color)
        self.err_console = Console(stderr=True, highlight=False, no_color=no_color)

    def info(self, message: str) -> None:
        if not self.quiet:
            self.err_console.print(Text(message), soft_wrap=True)

    def warning(self, message: str) -> None:
        prefix = Text("警告: ", style="bold yellow")
        prefix.append(message, style="yellow")
        self.err_console.print(prefix, soft_wrap=True)

    def error(self, message: str, *, hint: str | None = None) -> None:
        prefix = Text("错误: ", style="bold red")
        prefix.append(message, style="red")
        self.err_console.print(prefix, soft_wrap=True)
        if hint:
            self.err_console.print(Text(hint, style="dim"), soft_wrap=True)

    @contextmanager
    def progress(self, label: str, *, total: float = 100.0) -> Iterator[_ProgressHandle]:
        if self.quiet:
            yield _ProgressHandle(None, None)
            return
        progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.err_console,
            transient=True,
            refresh_per_second=8,
        )
        with progress:
            task_id = progress.add_task(escape(label), total=total)
            yield _ProgressHandle(progress, task_id)

    def summary(self, result: RunResult) -> None:
        self.console.print()
        self.console.print(Text("完成", style="bold green"))

        table = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(overflow="fold")

        table.add_row(Text("标题"), Text(result.title))
        if result.uploader:
            table.add_row(Text("上传者"), Text(result.uploader))
        table.add_row(Text("来源"), Text(result.source_url))
        table.add_row(Text("时长"), Text(human_duration(result.duration)))
        language = result.language or "自动检测（未返回）"
        table.add_row(Text("识别语言"), Text(language))
        table.add_row(
            Text("语音识别"),
            Text(
                f"{result.asr_backend} / {result.asr_model} · {result.segment_count} 段 · {result.raw_chars} 字"
            ),
        )
        if result.text_corrected:
            table.add_row(
                Text("文本纠错"),
                Text(f"{result.llm_backend} / {result.llm_model} · {result.llm_chunks} 块"),
            )
        else:
            table.add_row(Text("文本纠错"), Text("未启用（最终文本即原始转写）"))
        table.add_row(Text("输出文件"), Text(str(result.output_path)))
        if result.raw_path:
            table.add_row(Text("原始转写"), Text(str(result.raw_path)))
        if result.audio_path:
            table.add_row(Text("保留音频"), Text(str(result.audio_path)))
        table.add_row(Text("耗时"), Text(f"{result.elapsed_seconds:.1f} 秒"))

        self.console.print(table, markup=False)

    def print_corrected_text(self, text: str) -> None:
        self.console.print()
        self.console.rule(Text("文本", style="dim"))
        self.console.print(Text(text), markup=False, soft_wrap=True)
