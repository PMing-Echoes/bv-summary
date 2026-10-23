# bv-summary

一个用 Python 编写的命令行工具，把视频链接变成一份经过 LLM 纠错的文字稿：

```text
视频链接 → yt-dlp 下载音频 → ffmpeg 转为 16 kHz 单声道 WAV
        → ASR（本地 faster-whisper / OpenAI 兼容 API）
        → LLM 纠错（OpenAI 兼容接口 / 本地 Ollama）
        → UTF-8 .txt
```

支持 yt-dlp 能解析的站点，例如 **Bilibili、YouTube**，以及其他 yt-dlp 支持的平台。下载使用维护活跃的 `yt-dlp`，不使用已停止维护的 `youtube-dl`。

---

## 功能概览

- **输入**：单个视频页面链接；支持 cookies 和从浏览器读取 cookies 以处理登录内容。
- **音频**：下载最佳音轨，用 ffmpeg 统一转为 16 kHz、单声道、16-bit PCM WAV。
- **可插拔 ASR**：
  - 本地 `faster-whisper`，提供分段流式进度；
  - OpenAI 兼容的音频转写 API（可自定义 base URL 与模型）；
  - `auto` 模式优先本地，失败后可自动回退在线。
- **可插拔 LLM**：
  - 任意 OpenAI 兼容 `chat/completions` 接口（OpenAI、vLLM、LM Studio、Ollama 的 `/v1` 等）；
  - 专用本地 `ollama` 后端；
  - `auto` 模式优先配置好的远程服务，并自动探测本机 Ollama 作为回退；
  - 长文本按段落/句子边界分块，流式接收结果。
- **可靠输出**：
  - 默认不覆盖已有文件，自动生成 `文件名 (1).txt`；
  - 可选保留原始转写 `.raw.txt` 和音频 `.wav`；
  - LLM 失败时仍会保存原始转写，避免白跑；
  - 下载、提取、转写、纠错均有进度提示和可读错误信息。
- **不硬编码任何密钥**：统一从环境变量、`.env` 或 TOML 配置文件读取。

---

## 安装

### 环境要求

- Python **3.11+**
- 网络（安装依赖、首次下载 ASR 模型、在线 API）
- ffmpeg **可选**：若系统没有 ffmpeg，会自动使用 `imageio-ffmpeg` 自带的二进制；也可以手动指定路径
- 本地 ASR 建议至少 8 GB 内存；GPU 不是必需，CUDA 可用时会自动尝试
- Windows GPU 用户可直接安装 `gpu` extra，程序会自动登记 NVIDIA pip 包中的 CUDA/cuDNN DLL，不要求安装完整 CUDA Toolkit

### Windows PowerShell

```powershell
cd C:\projects\bv-summary
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
Copy-Item .env.example .env
bv-summary --help
```

`requirements.txt` 安装项目本身（editable）以及本地 ASR 依赖；它**不会**下载模型。

如果 PowerShell 禁止执行激活脚本，可以不激活，直接使用：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\bv-summary.exe --help
```

如果只需要在线 ASR、不需要本地 faster-whisper，可以更轻量地安装：

```powershell
pip install -e .
```

### Windows GPU（可选）

已有 NVIDIA 驱动（支持 CUDA 12 及以上）时，可以安装项目提供的 Windows GPU 运行库，无需完整 CUDA Toolkit：

```powershell
pip install -e ".[local-asr,gpu]"
```

这会安装 cuBLAS 12、NVRTC、CUDA runtime 12、cuDNN 9 的官方 Windows 运行库（约 1.4 GB，下载体积较大）。程序会在调用 faster-whisper 前自动发现并登记这些 DLL 所在目录。

然后显式启用 GPU：

```powershell
bv-summary "https://www.bilibili.com/video/BV..." --asr-device cuda
```

`ASR_COMPUTE_TYPE=auto` 会查询 CTranslate2 支持的类型：较老的 GTX 10 系 GPU 往往不支持 `float16`，此时会自动选择 `int8_float32`/`int8`（本次在 GTX 1080 Ti 上选择的是 `int8_float32`）。若 GPU 初始化仍失败，可回退到 CPU：

```powershell
bv-summary "https://www.bilibili.com/video/BV..." --asr-device cpu --asr-compute-type int8
```

也可以使用系统安装的 CUDA/cuDNN，只需确保对应的 `bin` 目录在 `PATH` 中。NVIDIA 的 Windows pip 包默认不会修改系统 `PATH`；本项目已内置自动发现逻辑。

### Linux / macOS（简要）

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
cp .env.example .env
bv-summary --help
```

### ffmpeg

程序按以下顺序查找 ffmpeg：

1. `--ffmpeg-location` / `FFMPEG_BINARY` / 配置文件中的 `download.ffmpeg_location`
2. 系统 `PATH` 中的 `ffmpeg`
3. `imageio-ffmpeg` 包内附带的 ffmpeg

为了兼容更多编码和获得更好的性能，仍建议安装系统 ffmpeg：

```powershell
winget install Gyan.FFmpeg
# 重开终端后：
ffmpeg -version
```

也可以设置：

```dotenv
FFMPEG_BINARY=C:\ffmpeg\bin\ffmpeg.exe
```

### 首次使用本地 ASR

本地 ASR 的 Python 包由 `local-asr` 可选依赖安装；语音模型在**第一次转写时**才会从 Hugging Face 下载，并缓存到默认位置（或 `ASR_MODEL_DIR`）。默认模型是 `small`。可选常用模型：

| 模型 | 大致特点 |
|---|---|
| `tiny` / `base` | 更快、更小，准确率较低，适合先跑通 |
| `small` | 默认，CPU 也可用，质量/速度折中 |
| `medium` / `large-v3` | 更准确，内存和显存占用更高 |

想在离线环境使用：先在有网机器上准备模型目录，然后把 `ASR_MODEL` 设成该本地模型目录（也可把缓存目录设为 `ASR_MODEL_DIR`）。

---

## 快速开始

### 1. 配置一个 OpenAI 兼容 LLM

复制 `.env.example` 为 `.env`，至少填写：

```dotenv
# 本地 ASR（默认引擎）
ASR_BACKEND=auto
ASR_MODEL=small

# OpenAI 兼容 LLM
LLM_BACKEND=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4o-mini
```

> `.env` 已加入 `.gitignore`。请勿把真实 Key 写进 README、代码或版本库。

### 2. 运行

```powershell
bv-summary "https://www.bilibili.com/video/BV1xx411c7mD" `
  --output "outputs/demo.txt" `
  --asr-model small `
  --model gpt-4o-mini `
  --base-url https://api.openai.com/v1
```

运行时终端会显示 4 个阶段：

1. 下载视频音轨
2. 提取音频（16 kHz 单声道 WAV）
3. 语音转文字
4. LLM 纠正（如果启用）

完成后，摘要和完整纠错文本会打印到终端，同时写入 UTF-8 文本文件。`--no-print-text` 可只在终端显示摘要。

### 3. 完全本地（Ollama + faster-whisper）

先安装并启动 Ollama，准备一个模型：

```powershell
ollama serve
# 另开终端：
ollama pull qwen2.5:7b
```

`.env`：

```dotenv
ASR_BACKEND=faster-whisper
ASR_MODEL=small
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_MODEL=qwen2.5:7b
```

运行：

```powershell
bv-summary "https://www.youtube.com/watch?v=VIDEO_ID" `
  --output "outputs/local.txt" `
  --asr-model small `
  --llm-backend ollama `
  --model qwen2.5:7b
```

显式 `--llm-backend ollama` 时，`--model` 也可以指定模型；`OLLAMA_MODEL` 优先用于自动回退。

### 4. 跳过 LLM 纠错

没有可用的 LLM 环境时，直接输出 ASR 原文：

```powershell
bv-summary "https://www.bilibili.com/video/BV1xx411c7mD" --llm-backend none -o transcript.txt
```

### 5. 在线 ASR 作为本地后备

```dotenv
ASR_BACKEND=auto
ASR_API_KEY=sk-your-key-here
# 可选：
ASR_BASE_URL=https://api.openai.com/v1
ASR_API_MODEL=whisper-1
```

本地 faster-whisper 未安装或运行失败时，`auto` 会尝试在线 ASR。若要强制在线：`--asr-backend openai`。

---

## 配置

### 优先级

```text
命令行参数 > 环境变量 > TOML 配置文件 > 内置默认值
```

- 自动加载当前目录的 `.env`；不会覆盖进程环境里已有的变量。
- `--env-file PATH` 可指定其他 `.env`。
- 配置文件查找顺序：
  1. `--config PATH`
  2. 环境变量 `BV_SUMMARY_CONFIG`
  3. 当前目录 `config.toml`
  4. 用户目录：
     - Windows: `%APPDATA%\bv-summary\config.toml`
     - Linux/macOS: `$XDG_CONFIG_HOME/bv-summary/config.toml` 或 `~/.config/bv-summary/config.toml`
- 配置示例见 [`config.example.toml`](config.example.toml)。
- API Key 可以写进配置文件，但更推荐环境变量；请确保配置文件权限安全且不被提交。

### 输出路径

- `--output "dir/"`、`--output outputs`（无扩展名）或已存在的目录：目录模式，文件名为 `<视频标题>.txt`。
- `--output "my-file.txt"`：文件模式。
- 默认：当前目录下 `<视频标题>.txt`。
- 默认不覆盖：如果目标已存在，会自动改成 `my-file (1).txt`。`--force` 才会覆盖。
- `--save-raw`：启用 LLM 时，另外写入 `<文件名>.raw.txt`。
- `--keep-audio`：在输出文件旁写入 `<文件名>.wav`。
- LLM 全部失败时，即使没有 `--save-raw`，也会保存 `<文件名>.raw.txt` 并以非零状态退出。

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `BV_SUMMARY_OUTPUT` | 当前目录 | 输出文件或目录 |
| `BV_SUMMARY_KEEP_AUDIO` | `false` | 保留 WAV |
| `BV_SUMMARY_SAVE_RAW` | `false` | 启用 LLM 时保留原始转写 |
| `BV_SUMMARY_FORCE` | `false` | 覆盖已有输出 |
| `BV_SUMMARY_CONFIG` | — | TOML 配置文件路径 |
| `FFMPEG_BINARY` / `FFMPEG_LOCATION` | 自动查找 | ffmpeg 路径 |
| `YTDLP_COOKIES` | — | Netscape `cookies.txt` 路径 |
| `YTDLP_COOKIES_FROM_BROWSER` | — | `chrome`、`edge`、`firefox` 或 `firefox:PROFILE` |
| `ASR_BACKEND` | `auto` | `auto` / `faster-whisper` / `openai` |
| `ASR_MODEL` | `small` | faster-whisper 模型名或本地模型目录 |
| `ASR_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `ASR_COMPUTE_TYPE` | `auto` | 根据设备与 CTranslate2 支持的类型自动选择（CPU 通常 `int8`；CUDA 优先 `float16`，不支持时会选可用的 int8/float32 组合） |
| `ASR_LANGUAGE` | 自动检测 | 如 `zh`、`en`；也支持空值/`auto` |
| `ASR_MODEL_DIR` | Hugging Face 缓存 | 本地模型缓存目录 |
| `ASR_CPU_THREADS` | `0`（库默认） | CPU 线程数 |
| `ASR_BEAM_SIZE` | `5` | beam search 宽度 |
| `ASR_BASE_URL` | OpenAI 默认 | 在线 ASR 的 OpenAI 兼容地址 |
| `ASR_API_KEY` | — | 在线 ASR Key；缺省时尝试 `OPENAI_API_KEY` |
| `ASR_API_MODEL` | `whisper-1` | 在线 ASR 模型 |
| `ASR_CHUNK_SECONDS` | `600` | 超过 24 MB 时每块最大秒数 |
| `ASR_TIMEOUT` | `600` | 在线 ASR 请求超时（秒） |
| `LLM_BACKEND` | `auto` | `auto` / `openai` / `ollama` / `none` |
| `LLM_BASE_URL` | OpenAI 官方地址 | OpenAI 兼容 API 地址 |
| `LLM_API_KEY` | — | LLM Key；缺省时尝试 `OPENAI_API_KEY` |
| `LLM_MODEL` | — | LLM 模型名（启用远程 LLM 时必填） |
| `LLM_TEMPERATURE` | `0.1` | 0.0–2.0 |
| `LLM_CHUNK_CHARS` | `3000` | 每块最大字符数 |
| `LLM_TIMEOUT` | `120` | LLM 请求超时（秒） |
| `LLM_PROMPT_FILE` | 内置提示词 | 自定义系统提示词文件 |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434/v1` | Ollama 的 OpenAI 兼容地址 |
| `OLLAMA_MODEL` | — | Ollama 模型名；自动回退时优先使用 |
| `OPENAI_API_KEY` | — | ASR/LLM 的共享后备 Key |

安全提示：

- 密钥不通过命令行参数传递，避免进入 shell 历史；支持 `.env`、环境变量和配置文件。
- 日志和摘要不会打印 Key。
- 如果 `LLM_BASE_URL` / `ASR_BASE_URL` 指向本机或私网地址且未设置 Key，会使用占位 Key 调用（适用于无鉴权的本地 vLLM / Ollama / LM Studio 等）。公网地址默认要求 Key。

---

## CLI 参数

```text
bv-summary URL [options]
```

最常用的参数：

| 参数 | 说明 |
|---|---|
| `-o, --output PATH` | 输出文件或目录 |
| `--asr-backend` | `auto` / `faster-whisper` / `openai` |
| `--asr-model` | faster-whisper 模型，例如 `small` |
| `--asr-device` | `auto` / `cpu` / `cuda` |
| `--asr-compute-type` | `int8`、`float16`、`float32` 等 |
| `--asr-base-url` | 在线 ASR 服务地址 |
| `--language` | ASR 语言，如 `zh`、`en` |
| `--llm-backend` | `auto` / `openai` / `ollama` / `none` |
| `--model` | LLM 模型名 |
| `--base-url` | OpenAI 兼容 LLM API 地址 |
| `--prompt-file` | 自定义纠错系统提示词 |
| `--cookies` | cookies.txt |
| `--cookies-from-browser` | 从浏览器读取 cookies |
| `--keep-audio` | 保留提取后的 WAV |
| `--save-raw` | 保留未纠错转写 |
| `--force` | 覆盖已有文件 |
| `--no-print-text` | 终端不打印完整文本 |
| `--quiet` | 隐藏进度和普通日志 |
| `--verbose` | 出错时显示 traceback |

完整列表使用 `bv-summary --help` 查看。

### Cookies 示例

Bilibili/YouTube 的部分视频需要登录 Cookie：

```powershell
bv-summary "https://www.bilibili.com/video/BV..." --cookies-from-browser edge
# 或
bv-summary "https://www.bilibili.com/video/BV..." --cookies "C:\secrets\cookies.txt"
```

`--cookies-from-browser` 支持 `BROWSER`、`BROWSER:PROFILE`、`BROWSER+KEYRING:PROFILE` 子集；不支持 yt-dlp 的 `::CONTAINER` 完整语法。若需要该语法，请直接导出 cookies.txt 并通过 `--cookies` 使用。

---

## 后端选择与降级策略

### ASR

`--asr-backend auto`（默认，或 `ASR_BACKEND=auto`）：

1. 如果安装了 `faster-whisper`，优先生成本地后端。
2. 如果配置了可用的在线 ASR（有 `ASR_API_KEY`/`OPENAI_API_KEY`，或 `ASR_BASE_URL` 指向本机/私网且可无 Key 访问），同时准备在线后端。
3. 本地运行失败（模型下载失败、CUDA 不可用、依赖损坏等）时，自动尝试在线后端。
4. 两者都不可用或都失败时，给出包含各方失败的明确错误，不会静默产出空文件。

显式指定 `--asr-backend faster-whisper` 或 `--asr-backend openai` 时**不会**自动跨后端回退，便于严格控制成本和隐私。

本地 ASR 通过 `faster-whisper` 的生成器逐段收集结果并刷新进度；超长音频不会一次性返回一个不可见的长时间阻塞结果。在线 ASR 会把大于约 24 MB 的 WAV 在本机按帧切块（默认每块最多 600 秒），逐块上传并合并，避免撞到 API 的 25 MB 限制。切块是按帧进行的，边界处可能切开一个词，后续 LLM 通常会修正；若对边界很敏感，可改用本地 ASR。

### LLM

`--llm-backend auto`（默认）：

1. 若配置了 `LLM_BASE_URL` 和/或 `LLM_API_KEY`，且存在 `LLM_MODEL`，优先使用 OpenAI 兼容后端。
   - `LLM_BASE_URL` 指向本机/私网时可以无 Key（使用占位 Key）。
   - 公网地址没有 Key 时会跳过远程后端。
2. 如果能探测到本机 Ollama（默认 `http://127.0.0.1:11434/v1`），且能确定一个 Ollama 模型名（`OLLAMA_MODEL` 优先，其次 `LLM_MODEL`），准备 Ollama 作为回退。
3. 远程后端失败时，自动重试整个纠错任务到 Ollama；当前进度和临时文本不会写入最终结果。
4. 全部失败时保存原始转写到 `.raw.txt`，以非零状态退出。若所有 LLM 都不适合，可显式 `--llm-backend none` 跳过纠错。

显式 `--llm-backend openai` 或 `--llm-backend ollama` 时不跨后端回退。`auto` 探测 Ollama 只访问本机/配置的地址，不发送转写文本；只有实际调用 LLM 时才发送。

长文本按“段落 → 行 → 句子 → 子句 → 空格”的优先级分块，默认每块最多 3000 字符，避免超出上下文。每个块优先使用流式 `chat/completions`；某些兼容服务不支持流式或拒绝某些参数时，会自动用非流式请求重试一次。若文本明显短于原文，会判定为可能截断并失败，不会覆盖最终文件。

---

## 错误处理与退出码

程序会在开始耗时下载之前尽早检查 ffmpeg、ASR 和 LLM 配置。预期错误会变成简短的终端信息，并附带修复建议。

| 退出码 | 含义 |
|---|---|
| `0` | 成功 |
| `1` | 未预期的内部错误 |
| `2` | 命令行/配置/URL 错误 |
| `3` | 缺少依赖或 ffmpeg |
| `4` | 下载失败 |
| `5` | 音频提取失败 |
| `6` | 所有 ASR 后端失败 |
| `7` | 所有 LLM 后端失败（原始转写会尽量保留） |
| `130` | Ctrl+C 取消 |

## 常见问题

### `找不到 ffmpeg`

先安装系统 ffmpeg，或：

```powershell
pip install imageio-ffmpeg
pip install --force-reinstall imageio-ffmpeg  # 如果内置二进制缺失
```

然后通过 `--ffmpeg-location C:\ffmpeg\bin\ffmpeg.exe` 指定，或确认 `ffmpeg -version` 可用。

### faster-whisper 安装或加载失败

- 先尝试 CPU：`--asr-device cpu --asr-compute-type int8`。
- GPU 用户需匹配 CUDA/cuDNN 版本；若驱动/运行库不完整，设置 `ASR_DEVICE=cpu` 即可回到 CPU。
- 首次运行需要从 Hugging Face 下载模型，网络需要能访问模型仓库。若下载失败，可手动准备模型目录并把 `ASR_MODEL` 设为该路径。
- 本地 ASR 不可用时，可改用在线 ASR（设置 `ASR_API_KEY`），或安装 `faster-whisper` 后重试。
- 本流水线把已提取的 16 kHz 单声道 WAV 直接以 NumPy 数组送入 faster-whisper，避免 PyAV 19 与 faster-whisper 1.2.1 在 `metadata_errors` 参数上的不兼容；无需为此降级 PyAV。

### Bilibili 403、YouTube “Sign in to confirm you're not a bot”

这类内容通常需要登录或触发风控：

1. 用浏览器登录并播放该视频，确认浏览器可访问；
2. 使用 `--cookies-from-browser chrome` 或导出 cookies.txt；
3. 升级 yt-dlp：`python -m pip install -U yt-dlp`；
4. 部分 YouTube 视频还需要 yt-dlp 的 JS 运行时（如 Deno）；请参考 yt-dlp 最新文档；
5. 仅下载你有权访问和处理的内容。

### LLM 认证失败 / 模型不存在

- 检查 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 三者是否匹配同一服务。
- OpenAI 兼容地址通常以 `/v1` 结尾，例如 `http://localhost:8000/v1`。
- Ollama：确认 `ollama serve` 正在运行，`OLLAMA_BASE_URL` 是 OpenAI 兼容地址，且 `ollama list` 中确实有该模型。
- 没有 LLM 服务时：`--llm-backend none`。

### 输出文件名很奇怪

程序会清理 Windows 不允许的字符、保留字（如 `CON`）和尾部空格/点，并限制标题长度。可用 `-o` 手动指定文件名。

### 代理

`yt-dlp` 和 OpenAI SDK 会遵循常见的代理环境变量（如 `HTTPS_PROXY`）。本机 Ollama 探测刻意绕过代理，避免把 localhost 请求发给代理。

### 编码

所有文本文件按 UTF-8 写入，并带末尾换行。终端显示依赖所用终端；Windows Terminal / PowerShell 7 通常没有问题。

---

## 隐私与安全

- **本地模式**（本地 ASR + Ollama）不会把音视频/转写发送到外部服务；只有首次下载模型需要网络。
- **在线模式**会把音频发送给 ASR 服务，把转写文本发送给 LLM 服务，请自行评估隐私与合规要求。
- 本工具不收集遥测，不内置任何密钥。
- Cookies 文件包含登录凭据，应妥善保管，切勿提交。
- 请遵守站点服务条款和版权规定，只下载、处理你有权使用的内容。

## 项目结构

```text
bv-summary/
├── pyproject.toml              # 包元数据、依赖、CLI entry point
├── requirements.txt            # 推荐安装入口（含 local-asr extra）
├── README.md
├── LICENSE
├── .env.example
├── config.example.toml
├── src/
│   └── bv_summary/
│       ├── cli.py              # argparse CLI、错误边界、退出码
│       ├── pipeline.py         # 端到端编排与降级
│       ├── config.py           # CLI/环境变量/TOML 配置解析
│       ├── cuda.py             # Windows NVIDIA pip 包 DLL 自动发现
│       ├── downloader.py       # yt-dlp 下载
│       ├── audio.py            # ffmpeg 查找与 WAV 提取
│       ├── output.py           # 输出路径、原子写入、sidecar 文件
│       ├── transcript.py       # ASR 分段合并为可读段落
│       ├── ui.py               # Rich 进度条与摘要
│       ├── openai_utils.py     # OpenAI SDK 创建/错误映射
│       ├── asr/
│       │   ├── base.py         # ASRBackend 接口
│       │   ├── factory.py      # ASR 后端选择/降级
│       │   ├── faster_whisper_backend.py
│       │   └── openai_backend.py
│       └── llm/
│           ├── base.py         # LLMBackend 接口
│           ├── factory.py      # LLM 后端选择/Ollama 探测
│           ├── openai_compatible.py
│           ├── prompts.py
│           └── text_chunks.py
├── tests/
│   ├── test_config.py
│   ├── test_transcript_and_output.py
│   ├── test_downloader.py
│   ├── test_audio.py
│   ├── test_asr.py
│   ├── test_llm.py
│   ├── test_pipeline.py
│   ├── test_cli.py
│   ├── test_cuda.py
│   ├── test_openai_sdk_integration.py
│   └── test_e2e_local.py
```

### 扩展新后端

- ASR：实现 [`ASRBackend`](src/bv_summary/asr/base.py) 的 `transcribe()`，在 `asr/factory.py` 注册。
- LLM：实现 [`LLMBackend`](src/bv_summary/llm/base.py) 的 `correct()`，在 `llm/factory.py` 中接入选择逻辑。
- 两个接口都只依赖普通 Python 数据类型和进度回调，便于用假实现做离线测试。

## 开发与测试

```powershell
pip install -e ".[dev,local-asr]"
pytest
ruff check src tests
ruff format --check src tests
```

离线测试不会调用真实平台下载或付费 API；测试覆盖配置、分块、下载编排、ASR/LLM 降级、输出与 CLI 错误边界。其中包含：

- 通过本机 mock HTTP 服务验证真实 OpenAI SDK 的流式 chat 与 multipart 音频上传；
- 一条本地端到端测试，使用真实 yt-dlp 下载本地服务器上的 WAV、真实 ffmpeg 提取、SDK 请求 mock ASR/LLM 并写入结果；
- ffmpeg 实际转换测试在没有 ffmpeg（含 `imageio-ffmpeg`）时会自动跳过。

当前验证环境为 Windows / Python 3.14.7 / faster-whisper 1.2.1 / NVIDIA GTX 1080 Ti（CUDA 12 + cuDNN 9）。

- **83 项离线测试通过**，`ruff` 检查与格式检查通过。
- 已完成一条真实 Bilibili 视频的处理（14:15，203 段，约 3236 字）：GPU ASR、真实 OpenAI 兼容 LLM（`deepseek-flash`，2 块）纠错和文件输出全部成功，整条流水线耗时约 226 秒。
- Whisper `small` 模型已下载并通过官方 SHA-256 校验；未提交模型、`.env` 或任何密钥。

## 已知限制

- 一次只处理一个视频，不支持播放列表/频道批量下载。
- 只输出纯文本，不输出 SRT/VTT；时间戳仅用于内部进度与摘要。
- 在线 ASR 的分块是在帧边界切开的，可能切开一个单词；需要严格逐字边界时建议用本地 ASR。
- 文件名基于标题生成；同一个标题重复运行时默认生成带序号的新文件。
- 具体站点能否下载取决于 yt-dlp 当前版本与平台的访问限制。
- LLM 纠错质量取决于所选模型与提示词；自定义提示词请自行验证。

## License

MIT，见 [LICENSE](LICENSE)。
