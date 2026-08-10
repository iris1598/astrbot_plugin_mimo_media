# astrbot_plugin_mimo_media

为 AstrBot 中的小米 MiMo（`xiaomi_chat_completion`，`mimo-v2.5` 系列）补全**视频**与**音频**多模态理解能力。

## 功能
- 用户在对话中发送**视频**：插件自动下载 → 强制重编码为标准 **H264 (libx264) + AAC** 的 MP4 → Base64 上传（`video_url` 内容块，可配 `fps` / `media_resolution`）。
- 用户在对话中发送**音频/语音**：插件自动转换为标准 **WAV** → Base64 上传（`input_audio` 内容块，`data:` 前缀格式，同 MiMo 官方文档）。
- 与 AstrBot 图片处理走**同一对话流水线**（`on_llm_request` 钩子），保留人设、会话历史与多轮上下文。
- 非 MiMo 提供商（或未开启插件）时完全无操作。

## 安装
1. 将本目录复制到 AstrBot 的 `data/plugins/astrbot_plugin_mimo_media`。
2. 重启 AstrBot 或使用 WebUI “插件管理 → 重载插件”。
3. 确保系统已安装 `ffmpeg`（需编译启用 `libx264`）且 `ffmpeg`/`ffprobe` 在 PATH 中。

## 使用
- 在配置了 MiMo（`xiaomi_chat_completion` + `mimo-v2.5` 系列）的会话中，直接发送视频或语音消息即可。
- 可附加文字指令（如“描述一下这个视频讲了什么”），纯媒体时插件会自动补充引导文本。

## 配置
| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `enable` | `true` | 是否启用本插件 |
| `video_fps` | `2.0` | 视频每秒抽帧数，范围 [0.1, 10] |
| `video_resolution` | `default` | 视频单帧分辨率档次：`default` / `max` |
| `video_max_base64_mb` | `49` | 视频 Base64 上限（MiMo 要求 ≤50MB），超限先压缩一次 |
| `audio_max_base64_mb` | `49` | 音频 Base64 上限（MiMo 要求 ≤50MB） |
| `max_video_width` | `1280` | 超限压缩一次时的目标宽度（CRF 28） |
| `persist_media_to_history` | `false` | 是否把媒体写入会话历史（默认不入库避免撑爆数据库） |

## 体积超限策略
- 视频：先按标准 H264 转码；超过上限则**压缩一次**（缩放 + CRF 28）；仍超限则不再上传，改为向模型注入文本提示“视频过大”，模型会据此回复，请求不会失败。
- 音频：超过上限则直接注入文本提示“音频过大”，请求不会失败。

## 说明
- 插件无新增第三方 Python 依赖；音频与视频都直接调用系统 `ffmpeg` 转码，音频固定输出单声道、16 kHz、16-bit PCM WAV。`MediaResolver` 仅用于将 URL/Base64 音频落盘，不负责格式转换。
- 注入的媒体内容块默认不写入会话历史（`mark_as_temp`），如需同一会话多轮追问请开启 `persist_media_to_history`。
