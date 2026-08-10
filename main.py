"""astrbot_plugin_mimo_media —— 为 AstrBot 中的小米 MiMo 补全视频/音频多模态理解。

工作方式：
- 通过 on_llm_request 钩子，在 MiMo 提供商收到请求前，扫描消息链中的 Video/Record 组件。
- 视频：下载后强制重编码为标准 H264 (libx264) + AAC 的 MP4，Base64 后以
  {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}, "fps": ..., "media_resolution": ...}
  注入当前请求，与图片处理走同一对话流水线。
- 音频：取得原始音频后通过系统 FFmpeg 强制转换为标准 WAV，Base64 后以
  {"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,..."}}
  注入（MiMo 要求 data 携带 data: 前缀、不含 format 字段）。
- 非 MiMo 提供商时插件完全无操作。
"""

import asyncio
import base64
import os
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import ContentPart, TextPart
from astrbot.core.message.components import Record, Reply, Video
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from astrbot.core.utils.datetime_utils import generate_timestamp_id
from astrbot.core.utils.media_utils import MediaResolver

DEFAULT_FPS = 2.0
DEFAULT_RESOLUTION = "default"
DEFAULT_MAX_BASE64_MB = 49.0
DEFAULT_MAX_VIDEO_WIDTH = 1280

VIDEO_MIME = "video/mp4"
AUDIO_MIME = "audio/wav"

# build_main_agent 生成的占位符前缀，插件会把它们替换为真实媒体内容块
_VIDEO_PLACEHOLDER_PREFIXES = (
    "[Video Attachment:",
    "[Video Attachment in quoted message:",
)
_AUDIO_PLACEHOLDER_PREFIXES = (
    "[Audio Attachment:",
    "[Audio Attachment in quoted message:",
)


class VideoURLPart(ContentPart):
    """MiMo 视频理解内容块。"""

    class VideoURL(BaseModel):
        url: str

    type: str = "video_url"
    video_url: VideoURL
    fps: float = DEFAULT_FPS
    media_resolution: str = DEFAULT_RESOLUTION


class InputAudioPart(ContentPart):
    """MiMo 音频理解内容块（data 为带 data: 前缀的 Data URL）。"""

    class InputAudio(BaseModel):
        data: str

    type: str = "input_audio"
    input_audio: InputAudio


def _temp_path(suffix: str) -> Path:
    temp_dir = Path(get_astrbot_temp_path())
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / f"mimo_media_{generate_timestamp_id()}{suffix}"


def _is_temp_path(path: str) -> bool:
    try:
        temp_dir = os.path.realpath(get_astrbot_temp_path())
        return os.path.realpath(path).startswith(temp_dir + os.sep)
    except Exception:
        return False


def _cleanup_paths(paths: list[str]) -> None:
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MiMoMedia] 清理临时文件失败 %s: %s", path, exc)


async def _run_ffmpeg(args: list[str]) -> None:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        error_msg = (stderr or b"").decode(errors="ignore")[:1024]
        raise RuntimeError(f"ffmpeg 执行失败: {error_msg}")


async def _to_h264_mp4(
    src: str,
    *,
    max_width: int | None = None,
    crf: int | None = None,
) -> Path:
    """始终重编码为标准 H264 + AAC 的 MP4，不依赖源扩展名早退。"""
    out = _temp_path(".mp4")
    args = [
        "ffmpeg",
        "-y",
        "-i",
        src,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
    ]
    if crf:
        args += ["-crf", str(int(crf))]
    args += ["-c:a", "aac", "-movflags", "+faststart"]
    if max_width:
        args += ["-vf", f"scale='min({int(max_width)},iw)':-2"]
    args.append(str(out))
    try:
        await _run_ffmpeg(args)
    except Exception:
        try:
            out.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return out


async def _to_standard_wav(src: str) -> Path:
    """始终通过系统 FFmpeg 转换为标准 PCM WAV。"""
    out = _temp_path(".wav")
    args = [
        "ffmpeg",
        "-y",
        "-i",
        src,
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(out),
    ]
    try:
        await _run_ffmpeg(args)
    except Exception:
        try:
            out.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return out


def _bytes_to_data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _base64_size_mb(payload: str) -> float:
    return len(payload) / (1024 * 1024)


def _validate_wav_data_url(data_url: str, source: str) -> None:
    """校验音频 Data URL 的字节为 RIFF/WAVE，避免把 SILK 等坏字节发给 API。"""
    try:
        encoded = "".join(data_url.split(",", 1)[-1][:64].split())
        padding = len(encoded) % 4
        if padding:
            encoded += "=" * (4 - padding)
        header = base64.b64decode(encoded)
    except Exception:
        header = b""
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return
    raise ValueError(f"音频未能转换为标准 WAV（无法识别音频字节）: {source}")


class MiMoMediaPlugin(Star):
    def __init__(self, context: Context, config: Any = None):
        super().__init__(context)
        self.config = config if isinstance(config, dict) else {}

    def _cfg(self, key: str, default: Any) -> Any:
        return self.config.get(key, default)

    def _enabled(self) -> bool:
        return bool(self._cfg("enable", True))

    def _is_mimo_provider(self, event: AstrMessageEvent) -> bool:
        """判定当前会话激活的对话提供商是否为小米 MiMo。"""
        try:
            prov = self.context.get_using_provider(umo=event.unified_msg_origin)
        except Exception:
            return False
        if prov is None:
            return False
        try:
            model = str(prov.get_model() or "").lower()
            api_base = str(prov.provider_config.get("api_base", "") or "").lower()
            provider_type = str(prov.provider_config.get("type", "") or "").lower()
        except Exception:
            return False
        return (
            "mimo" in model
            or "xiaomimimo" in api_base
            or provider_type in ("xiaomi_chat_completion", "xiaomi_token_plan")
        )

    @filter.on_llm_request(priority=10)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """在 MiMo 的 LLM 请求前，把消息里的视频/音频注入为多模态内容块。"""
        try:
            if not self._enabled():
                return
            if not self._is_mimo_provider(event):
                return
            await self._handle(event, req)
        except Exception as exc:  # noqa: BLE001
            logger.error("[MiMoMedia] on_llm_request 处理失败: %s", exc, exc_info=True)

    async def _handle(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        videos = self._collect_video_components(event)
        request_audio_refs = list(req.audio_urls or [])
        audio_records = self._collect_audio_components(event)
        if audio_records:
            audio_refs = []
            for record in audio_records:
                source = await self._get_record_source(record)
                if source:
                    audio_refs.append(source)
            # 某些非标准事件没有可用的原始 Record 源，保留请求中的路径作为兼容回退。
            if not audio_refs:
                audio_refs = list(req.audio_urls or [])
        else:
            audio_refs = list(req.audio_urls or [])
        if not videos and not audio_refs:
            return

        added_media_parts: list[ContentPart] = []
        notes: list[str] = []
        cleanup_paths: list[str] = []
        # 原始 Record 优先时，清理由 AstrBot 预先生成但插件不会再使用的音频文件。
        for request_audio_ref in request_audio_refs:
            if (
                _is_temp_path(request_audio_ref)
                and request_audio_ref not in audio_refs
            ):
                cleanup_paths.append(request_audio_ref)

        # 1. 视频：下载 -> 强制重编码 H264 -> Base64 Data URL
        for video in videos:
            part, note, paths = await self._process_video(video)
            cleanup_paths.extend(paths)
            if part is not None:
                added_media_parts.append(part)
            if note:
                notes.append(note)

        # 2. 音频：取得原始文件 -> FFmpeg 转标准 WAV -> Base64 Data URL
        req.audio_urls = []
        for audio_ref in audio_refs:
            part, note, paths = await self._process_audio(audio_ref)
            cleanup_paths.extend(paths)
            if part is not None:
                added_media_parts.append(part)
            if note:
                notes.append(note)

        if not added_media_parts and not notes:
            return

        # 3. 移除核心生成的占位符文本，替换为真实媒体内容
        self._drop_placeholders(req)

        # 4. 纯媒体无文本时补充引导指令（MiMo 示例要求文本与媒体共存）
        if not (req.prompt or "").strip() and added_media_parts:
            req.extra_user_content_parts.append(
                TextPart(text="请分析这段视频/音频的内容。")
            )

        persist = bool(self._cfg("persist_media_to_history", False))
        for part in added_media_parts:
            if not persist:
                part.mark_as_temp()  # 不入库，避免大段 Base64 撑爆会话历史
            req.extra_user_content_parts.append(part)
        for note in notes:
            req.extra_user_content_parts.append(TextPart(text=note))

        # 5. 清理插件创建的临时文件（下载源 + 转码产物）
        _cleanup_paths(cleanup_paths)

    @staticmethod
    def _collect_video_components(event: AstrMessageEvent) -> list[Video]:
        videos: list[Video] = []
        for comp in event.message_obj.message:
            if isinstance(comp, Video):
                videos.append(comp)
            elif isinstance(comp, Reply) and comp.chain:
                for reply_comp in comp.chain:
                    if isinstance(reply_comp, Video):
                        videos.append(reply_comp)
        return videos

    @staticmethod
    def _collect_audio_components(event: AstrMessageEvent) -> list[Record]:
        records: list[Record] = []
        for comp in event.message_obj.message:
            if isinstance(comp, Record):
                records.append(comp)
            elif isinstance(comp, Reply) and comp.chain:
                for reply_comp in comp.chain:
                    if isinstance(reply_comp, Record):
                        records.append(reply_comp)
        return records

    @staticmethod
    async def _get_record_source(record: Record) -> str:
        """取得 Record 的原始引用，不调用 Record.convert_to_file_path。"""
        resolve_source = getattr(record, "_resolve_file_source", None)
        if resolve_source is not None:
            return str(await resolve_source() or "")
        return str(
            getattr(record, "file", None)
            or getattr(record, "url", None)
            or getattr(record, "path", None)
            or ""
        )

    async def _process_video(self, video: Video):
        """处理单个视频组件，返回 (ContentPart | None, note | None, cleanup_paths)。"""
        cleanup_paths: list[str] = []
        try:
            src = await video.convert_to_file_path()
            if not src or not os.path.exists(src):
                return None, "[视频无法解析，已跳过]", cleanup_paths
            if _is_temp_path(src):
                cleanup_paths.append(src)

            out = await _to_h264_mp4(src)
            cleanup_paths.append(str(out))

            max_mb = float(self._cfg("video_max_base64_mb", DEFAULT_MAX_BASE64_MB))
            data_url = _bytes_to_data_url(out.read_bytes(), VIDEO_MIME)
            if _base64_size_mb(data_url) > max_mb:
                # 超限：压缩一次（缩放到 max_video_width + CRF 28）
                max_width = int(self._cfg("max_video_width", DEFAULT_MAX_VIDEO_WIDTH))
                out2 = await _to_h264_mp4(src, max_width=max_width, crf=28)
                cleanup_paths.append(str(out2))
                data_url = _bytes_to_data_url(out2.read_bytes(), VIDEO_MIME)

            if _base64_size_mb(data_url) > max_mb:
                return (
                    None,
                    f"[视频过大，已跳过：压缩后仍超过 {max_mb:.0f}MB]",
                    cleanup_paths,
                )

            fps = float(self._cfg("video_fps", DEFAULT_FPS))
            fps = max(0.1, min(10.0, fps))
            resolution = str(self._cfg("video_resolution", DEFAULT_RESOLUTION))
            if resolution not in ("default", "max"):
                resolution = DEFAULT_RESOLUTION

            part = VideoURLPart(
                video_url=VideoURLPart.VideoURL(url=data_url),
                fps=fps,
                media_resolution=resolution,
            )
            return part, None, cleanup_paths
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MiMoMedia] 视频处理失败: %s", exc)
            return None, "[视频解析失败，已跳过]", cleanup_paths

    async def _process_audio(self, audio_ref: str):
        """处理单个音频引用，返回 (ContentPart | None, note | None, cleanup_paths)。"""
        cleanup_paths: list[str] = []
        try:
            # 只负责把 URL / Base64 / file URI 落成原始文件，不指定 media_type=audio，
            # 避免触发 AstrBot 的 ensure_wav / SILK 内部转换。
            source = Path(
                await MediaResolver(
                    audio_ref,
                    media_type="file",
                    default_suffix=".audio",
                ).to_path()
            )
            if not source.exists():
                return None, "[音频解析失败，已跳过]", cleanup_paths
            if _is_temp_path(str(source)):
                cleanup_paths.append(str(source))

            out = await _to_standard_wav(str(source))
            cleanup_paths.append(str(out))
            data_url = _bytes_to_data_url(out.read_bytes(), AUDIO_MIME)
            _validate_wav_data_url(data_url, audio_ref)

            max_mb = float(self._cfg("audio_max_base64_mb", DEFAULT_MAX_BASE64_MB))
            if _base64_size_mb(data_url) > max_mb:
                return None, f"[音频过大，已跳过：超过 {max_mb:.0f}MB]", cleanup_paths

            part = InputAudioPart(
                input_audio=InputAudioPart.InputAudio(data=data_url),
            )
            return part, None, cleanup_paths
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MiMoMedia] 音频处理失败: %s", exc)
            return None, "[音频解析失败，已跳过]", cleanup_paths

    @staticmethod
    def _drop_placeholders(req: ProviderRequest) -> None:
        """移除 build_main_agent 生成的 [Video/Audio Attachment ...] 占位文本。"""
        prefixes = (*_VIDEO_PLACEHOLDER_PREFIXES, *_AUDIO_PLACEHOLDER_PREFIXES)
        kept = []
        for part in req.extra_user_content_parts:
            text = getattr(part, "text", "")
            if isinstance(text, str) and text.startswith(prefixes):
                continue
            kept.append(part)
        req.extra_user_content_parts = kept

    async def terminate(self):
        """插件卸载/停用时调用。"""
