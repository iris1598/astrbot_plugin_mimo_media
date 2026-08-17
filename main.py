"""astrbot_plugin_mimo_media —— 为 AstrBot 中的小米 MiMo 补全视频/音频多模态理解。

工作方式：
- 通过 on_llm_request 钩子，在 MiMo 提供商收到请求前，扫描消息链中的 Video/Record 组件。
- 视频：下载后强制重编码为标准 H264 (libx264) + AAC 的 MP4，Base64 后以
  {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}, "fps": ..., "media_resolution": ...}
  注入当前请求，与图片处理走同一对话流水线。
- 音频：取得原始音频后通过系统 FFmpeg 强制转换为标准 WAV，Base64 后以
  {"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,..."}}
  注入（MiMo 要求 data 携带 data: 前缀、不含 format 字段）。
- llonebot_stt 会在任意当前模型下执行；其他媒体处理仍只对 MiMo 生效。
- 路由开启后，多模态消息会临时选择配置的 MiMo 提供商。
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
from astrbot.core.message.components import Image, Record, Reply, Video
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from astrbot.core.utils.datetime_utils import generate_timestamp_id
from astrbot.core.utils.media_utils import MediaResolver

DEFAULT_FPS = 2.0
DEFAULT_RESOLUTION = "default"
DEFAULT_MAX_BASE64_MB = 49.0
DEFAULT_MAX_VIDEO_WIDTH = 1280
DEFAULT_AUDIO_MODE = "multimodal"
LLONEBOT_STT_AUDIO_MODE = "llonebot_stt"

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
        # Remaining requests in the temporary MiMo routing window, keyed by UMO.
        self._routing_remaining: dict[str, int] = {}

    def _cfg(self, key: str, default: Any) -> Any:
        return self.config.get(key, default)

    def _enabled(self) -> bool:
        return bool(self._cfg("enable", True))

    def _routing_enabled(self) -> bool:
        return bool(self._cfg("multimodal_routing_enabled", False))

    def _routing_provider_id(self) -> str:
        return str(self._cfg("multimodal_provider_id", "") or "").strip()

    def _routing_turns(self) -> int:
        raw_value = self._cfg("multimodal_route_turns", 1)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            logger.warning(
                "[MiMoMedia] 无效的 multimodal_route_turns=%r，使用默认值 1",
                raw_value,
            )
            return 1
        return max(1, value)

    @staticmethod
    def _session_key(event: AstrMessageEvent) -> str:
        return str(
            getattr(event, "unified_msg_origin", None)
            or getattr(getattr(event, "message_obj", None), "session_id", None)
            or getattr(event, "session_id", None)
            or "default"
        )

    @staticmethod
    def _message_components(event: AstrMessageEvent) -> list[Any]:
        components: list[Any] = []
        message_obj = getattr(event, "message_obj", None)
        for component in getattr(message_obj, "message", []) or []:
            components.append(component)
            if isinstance(component, Reply) and component.chain:
                components.extend(component.chain)
        return components

    def _has_multimodal_message(self, event: AstrMessageEvent) -> bool:
        return any(
            isinstance(component, (Image, Video, Record))
            for component in self._message_components(event)
        )

    def _target_provider_available(self) -> bool:
        target_id = self._routing_provider_id()
        if not target_id:
            logger.warning("[MiMoMedia] 未配置多模态路由目标 MiMo provider")
            return False
        provider = self.context.get_provider_by_id(target_id)
        if provider is None:
            logger.error(
                "[MiMoMedia] 多模态路由目标 provider %r 不存在，请检查插件配置",
                target_id,
            )
            return False
        if not self._provider_is_mimo(provider):
            logger.error(
                "[MiMoMedia] 多模态路由目标 provider %r 不是 MiMo provider",
                target_id,
            )
            return False
        return True

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def prepare_multimodal_routing(self, event: AstrMessageEvent):
        """Temporarily select the configured MiMo provider for multimodal turns."""
        try:
            if not self._enabled() or not self._routing_enabled():
                self._routing_remaining.clear()
                return

            key = self._session_key(event)
            remaining = self._routing_remaining.get(key, 0)
            has_multimodal = self._has_multimodal_message(event)

            # Each multimodal message refreshes the routing window. Plain follow-up
            # messages consume the remaining requests without extending it.
            if has_multimodal:
                remaining = self._routing_turns()

            if remaining <= 0:
                self._routing_remaining.pop(key, None)
                return
            if not self._target_provider_available():
                self._routing_remaining.pop(key, None)
                return

            event.set_extra("selected_provider", self._routing_provider_id())
            event.set_extra("mimo_media_routed", True)
            remaining -= 1
            if remaining:
                self._routing_remaining[key] = remaining
            else:
                self._routing_remaining.pop(key, None)
            logger.info(
                "[MiMoMedia] 多模态路由至 %s，本次后剩余 %d 轮",
                self._routing_provider_id(),
                remaining,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[MiMoMedia] 多模态路由选择失败: %s", exc, exc_info=True)

    def _audio_mode(self) -> str:
        mode = str(self._cfg("audio_mode", DEFAULT_AUDIO_MODE) or "").strip().lower()
        if mode not in (DEFAULT_AUDIO_MODE, LLONEBOT_STT_AUDIO_MODE):
            return DEFAULT_AUDIO_MODE
        return mode

    @staticmethod
    def _provider_is_mimo(provider: Any) -> bool:
        """Return whether a provider is a Xiaomi MiMo chat provider."""
        if provider is None:
            return False
        try:
            model = str(provider.get_model() or "").lower()
            api_base = str(provider.provider_config.get("api_base", "") or "").lower()
            provider_type = str(provider.provider_config.get("type", "") or "").lower()
        except Exception:
            return False
        return (
            "mimo" in model
            or "xiaomimimo" in api_base
            or provider_type in ("xiaomi_chat_completion", "xiaomi_token_plan")
        )

    def _is_mimo_provider(self, event: AstrMessageEvent) -> bool:
        """判定本次请求实际选择的对话提供商是否为小米 MiMo。"""
        selected_provider = None
        try:
            selected_provider = event.get_extra("selected_provider")
        except Exception:
            pass
        if selected_provider:
            provider = self.context.get_provider_by_id(str(selected_provider))
            return self._provider_is_mimo(provider)
        try:
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
        except Exception:
            return False
        return self._provider_is_mimo(provider)

    @filter.on_llm_request(priority=10)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """处理 MiMo 多模态媒体，或为任意模型执行 llonebot 语音转写。"""
        try:
            if not self._enabled():
                return
            is_mimo_provider = self._is_mimo_provider(event)
            if self._audio_mode() != LLONEBOT_STT_AUDIO_MODE and not is_mimo_provider:
                return
            await self._handle(event, req, process_videos=is_mimo_provider)
        except Exception as exc:  # noqa: BLE001
            logger.error("[MiMoMedia] on_llm_request 处理失败: %s", exc, exc_info=True)

    async def _handle(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        *,
        process_videos: bool = True,
    ) -> None:
        videos = self._collect_video_components(event) if process_videos else []
        request_audio_refs = list(req.audio_urls or [])
        audio_targets = self._collect_audio_targets(event)
        audio_records = [record for record, _, _ in audio_targets]
        audio_mode = self._audio_mode()
        audio_refs: list[str] = []
        transcriptions: list[tuple[str, bool]] = []
        notes: list[str] = []

        if audio_mode == LLONEBOT_STT_AUDIO_MODE:
            req.audio_urls = []
            if audio_targets:
                transcriptions = await self._transcribe_with_llonebot(
                    event,
                    audio_targets,
                )
                self._inject_llonebot_transcriptions(req, transcriptions)
                if not transcriptions:
                    notes.append("[语音转文字失败，已跳过]")
            elif request_audio_refs:
                notes.append("[语音转文字失败：无法获取 OneBot 消息 ID]")
        else:
            if audio_records:
                for record in audio_records:
                    source = await self._get_record_source(record)
                    if source:
                        audio_refs.append(source)
                # 某些非标准事件没有可用的原始 Record 源，保留请求中的路径作为兼容回退。
                if not audio_refs:
                    audio_refs = list(req.audio_urls or [])
                self._replace_empty_text_for_quoted_audio(req)
            else:
                audio_refs = list(req.audio_urls or [])

        if not videos and not audio_refs and not transcriptions and not notes:
            return

        added_media_parts: list[ContentPart] = []
        cleanup_paths: list[str] = []
        # 原始 Record 优先时，清理由 AstrBot 预先生成但插件不会再使用的音频文件。
        for request_audio_ref in request_audio_refs:
            if _is_temp_path(request_audio_ref) and request_audio_ref not in audio_refs:
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
        if audio_mode == DEFAULT_AUDIO_MODE:
            for audio_ref in audio_refs:
                part, note, paths = await self._process_audio(audio_ref)
                cleanup_paths.extend(paths)
                if part is not None:
                    added_media_parts.append(part)
                if note:
                    notes.append(note)

        if not added_media_parts and not notes and not transcriptions:
            return

        # 3. 移除核心生成的占位符文本，替换为真实媒体内容
        self._drop_placeholders(req, drop_video=process_videos)

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
    def _collect_audio_targets(
        event: AstrMessageEvent,
    ) -> list[tuple[Record, str | int | None, bool]]:
        raw_message = getattr(event.message_obj, "raw_message", None)
        direct_message_id = None
        if isinstance(raw_message, dict):
            direct_message_id = raw_message.get("message_id") or raw_message.get("id")
        if direct_message_id is None:
            direct_message_id = getattr(event.message_obj, "message_id", None)

        targets: list[tuple[Record, str | int | None, bool]] = []
        for comp in event.message_obj.message:
            if isinstance(comp, Record):
                targets.append((comp, direct_message_id, False))
            elif isinstance(comp, Reply) and comp.chain:
                for reply_comp in comp.chain:
                    if isinstance(reply_comp, Record):
                        targets.append((reply_comp, comp.id, True))
        return targets

    async def _transcribe_with_llonebot(
        self,
        event: AstrMessageEvent,
        audio_targets: list[tuple[Record, str | int | None, bool]],
    ) -> list[tuple[str, bool]]:
        bot = getattr(event, "bot", None)
        call_action = getattr(bot, "call_action", None)
        if not callable(call_action):
            logger.warning("[MiMoMedia] 当前 OneBot11 事件不支持 call_action")
            return []

        transcriptions: list[tuple[str, bool]] = []
        seen_message_ids: set[str] = set()
        for _, message_id, is_reply in audio_targets:
            if message_id is None or not str(message_id).strip():
                logger.warning(
                    "[MiMoMedia] 语音消息缺少 message_id，无法调用 llonebot 转写"
                )
                continue
            message_id_key = str(message_id)
            if message_id_key in seen_message_ids:
                continue
            seen_message_ids.add(message_id_key)

            try:
                api_message_id: str | int = message_id
                if isinstance(message_id, str) and message_id.isdigit():
                    api_message_id = int(message_id)
                result = await call_action(
                    action="voice_msg_to_text",
                    message_id=api_message_id,
                )
                data = result.get("data", result) if isinstance(result, dict) else {}
                text = data.get("text") if isinstance(data, dict) else None
                if isinstance(text, str) and text.strip():
                    transcriptions.append((text.strip(), is_reply))
                else:
                    logger.warning(
                        "[MiMoMedia] llonebot 转写返回空文本，message_id=%s",
                        message_id,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[MiMoMedia] llonebot 语音转文字失败，message_id=%s: %s",
                    message_id,
                    exc,
                )
        return transcriptions

    @staticmethod
    def _inject_llonebot_transcriptions(
        req: ProviderRequest,
        transcriptions: list[tuple[str, bool]],
    ) -> None:
        for text, is_reply in transcriptions:
            if is_reply:
                for part in req.extra_user_content_parts:
                    if not isinstance(part, TextPart):
                        continue
                    if "<Quoted Message>" not in part.text:
                        continue
                    if "[Empty Text]" not in part.text:
                        continue
                    part.text = part.text.replace("[Empty Text]", text, 1)
                    break
                else:
                    req.extra_user_content_parts.append(TextPart(text=text))
            else:
                req.extra_user_content_parts.append(TextPart(text=text))

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
    def _drop_placeholders(
        req: ProviderRequest,
        *,
        drop_video: bool = True,
    ) -> None:
        """移除 build_main_agent 生成的 [Video/Audio Attachment ...] 占位文本。"""
        prefixes = _AUDIO_PLACEHOLDER_PREFIXES
        if drop_video:
            prefixes = (*_VIDEO_PLACEHOLDER_PREFIXES, *prefixes)
        kept = []
        for part in req.extra_user_content_parts:
            text = getattr(part, "text", "")
            if isinstance(text, str) and text.startswith(prefixes):
                continue
            kept.append(part)
        req.extra_user_content_parts = kept

    @staticmethod
    def _replace_empty_text_for_quoted_audio(req: ProviderRequest) -> None:
        """Mark an otherwise empty quoted message as audio when it contains a Record."""
        for part in req.extra_user_content_parts:
            if not isinstance(part, TextPart):
                continue
            if "<Quoted Message>" not in part.text:
                continue
            part.text = part.text.replace("[Empty Text]", "[Audio]")

    async def terminate(self):
        """插件卸载/停用时调用。"""
        self._routing_remaining.clear()
