"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence

from . import __version__
from .config import ASR_BACKEND_CHOICES, LLM_BACKEND_CHOICES, load_environment, load_settings
from .errors import AppError
from .pipeline import Pipeline
from .ui import Reporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bv-summary",
        description=(
            "视频链接 → 音频提取 → 语音转文字 → LLM 文本纠错。\n"
            "支持 yt-dlp 可解析的站点（YouTube、Bilibili 等），ASR 与 LLM 后端均可切换。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  bv-summary "https://www.bilibili.com/video/BV1xx411c7mD" -o "outputs/demo.txt" \\
      --model gpt-4o-mini --base-url https://api.openai.com/v1
  bv-summary "https://youtu.be/VIDEO_ID" --asr-model small \\
      --llm-backend ollama --model qwen2.5:7b
  bv-summary "https://www.bilibili.com/video/BV1xx411c7mD" --llm-backend none

模型与服务:
  LLM 的 API Key 从 LLM_API_KEY 或 OPENAI_API_KEY 读取；ASR 的 Key 从
  ASR_API_KEY 或 OPENAI_API_KEY 读取。也可使用 .env 或 --config 配置文件。
  未设置 --llm-backend none 时，auto 会尝试已配置的 OpenAI 兼容服务，
  然后尝试本机 Ollama。

更多说明见 README.md。""",
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="视频页面链接（http/https，支持 Bilibili、YouTube 等 yt-dlp 站点）",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="输出文件；或输出目录（目录会使用视频标题作为文件名）。默认写入当前目录",
    )
    parser.add_argument(
        "--asr-backend",
        choices=ASR_BACKEND_CHOICES,
        default=None,
        help="ASR 后端：auto（默认）、faster-whisper、openai",
    )
    parser.add_argument(
        "--asr-model", default=None, help="本地 faster-whisper 模型（默认 small，或模型目录）"
    )
    parser.add_argument(
        "--asr-device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="本地 ASR 设备：auto（默认）、cpu、cuda",
    )
    parser.add_argument(
        "--asr-compute-type",
        default=None,
        help="本地 ASR 计算类型：auto（默认）、int8、float16、float32 等",
    )
    parser.add_argument("--asr-model-dir", default=None, help="本地 ASR 模型缓存目录")
    parser.add_argument("--asr-base-url", default=None, help="OpenAI 兼容 ASR API 地址")
    parser.add_argument("--asr-api-model", default=None, help="在线 ASR 模型（默认 whisper-1）")
    parser.add_argument(
        "--language",
        "--asr-language",
        dest="language",
        default=None,
        metavar="LANG",
        help="ASR 语言代码（如 zh、en、ja）；auto 表示自动检测",
    )
    parser.add_argument(
        "--llm-backend",
        choices=LLM_BACKEND_CHOICES,
        default=None,
        help="LLM 后端：auto（默认）、openai（OpenAI 兼容）、ollama、none",
    )
    parser.add_argument(
        "--model",
        "--llm-model",
        dest="llm_model",
        default=None,
        help="LLM 模型名（也可通过 LLM_MODEL 设置）",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI 兼容 LLM API 地址，例如 https://api.openai.com/v1 或 http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--prompt-file", default=None, help="自定义 LLM 纠错系统提示词（UTF-8 .txt）"
    )
    parser.add_argument(
        "--cookies", metavar="FILE", help="Netscape cookies.txt（Bilibili/YouTube 登录内容）"
    )
    parser.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="从浏览器读取 cookies，例如 chrome、edge、firefox 或 firefox:PROFILE",
    )
    parser.add_argument("--ffmpeg-location", metavar="PATH", help="ffmpeg 可执行文件或所在目录")
    parser.add_argument(
        "--keep-audio",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="在输出文件旁保留提取后的 WAV 音频",
    )
    parser.add_argument(
        "--save-raw",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用 LLM 时另存未纠错的 *.raw.txt",
    )
    parser.add_argument(
        "-f",
        "--force",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖已存在的输出文件（默认自动改名，不覆盖）",
    )
    parser.add_argument("--config", metavar="PATH", help="TOML 配置文件路径")
    parser.add_argument("--env-file", metavar="PATH", help=".env 文件路径（默认读取当前目录 .env）")
    parser.add_argument(
        "--no-print-text", action="store_true", help="终端只打印摘要，不打印完整文本"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="隐藏进度和普通日志（错误/警告仍会显示）"
    )
    parser.add_argument("--no-color", action="store_true", help="禁用终端颜色")
    parser.add_argument(
        "--verbose", action="store_true", help="出错时显示 Python traceback，便于诊断"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("必须提供一个视频链接；使用 --help 查看示例。")

    reporter = Reporter(no_color=args.no_color)
    try:
        load_environment(args.env_file)
        settings = load_settings(args)
        reporter = Reporter(quiet=args.quiet, no_color=args.no_color)
        if not args.quiet:
            reporter.info(f"bv-summary {__version__}")
        result = Pipeline(settings, reporter).run(args.url)
        reporter.summary(result)
        if not args.no_print_text:
            reporter.print_corrected_text(result.text)
        return 0
    except AppError as exc:
        reporter.error(exc.message, hint=exc.hint)
        if args.verbose:
            traceback.print_exc()
        return exc.exit_code
    except KeyboardInterrupt:
        reporter.error("已取消。")
        return 130
    except Exception as exc:  # pragma: no cover - last-resort user-facing boundary
        reporter.error(f"未处理的错误：{type(exc).__name__}: {exc}")
        if args.verbose:
            traceback.print_exc()
        else:
            reporter.error("可加 --verbose 查看详细堆栈。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
