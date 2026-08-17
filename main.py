"""astrbot_plugin_mimo_media —— 为 AstrBot 中的小米 MiMo 补全视频/音频多模态理解。

工作方式：
- 通过 on_llm_request 钩子，在 MiMo 提供商收到请求前，扫描消息链中的 Video/Record 组件。
- 视频：下载后强制重编码为标准 H264 (libx264) + AAC 的 MP4，可选择 Base64
  或 AstrBot 文件服务临时公网链接，并以
  {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}, "fps": ..., "media_resolution": ...}
  注入当前请求，与图片处理走同一对话流水线。
- 音频：取得原始音频后通过系统 FFmpeg 强制转换为标准 WAV，Base64 后以
  {"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,..."}}
  注入（MiMo 要求 data 携带 data: 前缀、不含 format 字段）。
- llonebot_stt 会在任意当前模型下执行；其他媒体处理仍只对 MiMo 生效。
- 支持直通、路由和转述三种模式；转述模式只处理视频/音频，图片继续使用
  AstrBot 官方图片转述功能。
"""

import asyncio
import base64
import json
import os
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import Provider, ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core import astrbot_config, file_token_service
from astrbot.core.agent.message import ContentPart, Message, TextPart
from astrbot.core.message.components import (
    Forward,
    Image,
    Node,
    Nodes,
    Record,
    Reply,
    Video,
)
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from astrbot.core.utils.datetime_utils import generate_timestamp_id
from astrbot.core.utils.media_utils import MediaResolver
from astrbot.core.utils.quoted_message import extract_quoted_message_images
from astrbot.core.utils.quoted_message.onebot_client import OneBotClient

DEFAULT_FPS = 2.0
DEFAULT_RESOLUTION = "default"
DEFAULT_MAX_BASE64_MB = 49.0
DEFAULT_MAX_VIDEO_WIDTH = 1280
DEFAULT_MAX_VIDEO_COUNT = 3
DEFAULT_AUDIO_MODE = "multimodal"
DEFAULT_VIDEO_TRANSPORT = "base64"
DEFAULT_MULTIMODAL_MODE = "direct"
ROUTE_MULTIMODAL_MODE = "route"
CAPTION_MULTIMODAL_MODE = "caption"
ASTRBOT_FILE_SERVICE_TRANSPORT = "astrbot_file_service"
FILE_SERVICE_TOKEN_TTL_SECONDS = 15 * 60
LLONEBOT_STT_AUDIO_MODE = "llonebot_stt"
MAX_COMPONENT_DEPTH = 8
MAX_FORWARD_FETCH = 8

_FILE_REGISTRY_ATTR = "_mimo_media_reusable_video_files"
_FILE_HANDLER_ATTR = "_mimo_media_original_handle_file"
_FILE_WRAPPER_ATTR = "_mimo_media_handle_file_wrapper"
_FILE_OWNERS_ATTR = "_mimo_media_file_handler_owners"

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
        self._file_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._served_files: set[str] = set()
        self._file_tokens: set[str] = set()
        self._file_handler_owner_registered = False

    def _cfg(self, key: str, default: Any) -> Any:
        return self.config.get(key, default)

    def _enabled(self) -> bool:
        return bool(self._cfg("enable", True))

    def _multimodal_mode(self) -> str:
        raw_mode = self._cfg("multimodal_mode", None)
        if raw_mode is None:
            return (
                ROUTE_MULTIMODAL_MODE
                if bool(self._cfg("multimodal_routing_enabled", False))
                else DEFAULT_MULTIMODAL_MODE
            )
        mode = str(raw_mode or "").strip().lower()
        if mode not in (
            DEFAULT_MULTIMODAL_MODE,
            ROUTE_MULTIMODAL_MODE,
            CAPTION_MULTIMODAL_MODE,
        ):
            logger.warning(
                "[MiMoMedia] 无效的 multimodal_mode=%r，使用直通模式",
                raw_mode,
            )
            return DEFAULT_MULTIMODAL_MODE
        return mode

    def _routing_enabled(self) -> bool:
        return self._multimodal_mode() == ROUTE_MULTIMODAL_MODE

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

    def _max_video_count(self) -> int:
        raw_value = self._cfg("video_max_count", DEFAULT_MAX_VIDEO_COUNT)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            logger.warning(
                "[MiMoMedia] 无效的 video_max_count=%r，使用默认值 %d",
                raw_value,
                DEFAULT_MAX_VIDEO_COUNT,
            )
            return DEFAULT_MAX_VIDEO_COUNT
        return max(1, value)

    def _video_transport(self) -> str:
        transport = (
            str(self._cfg("video_transport", DEFAULT_VIDEO_TRANSPORT) or "")
            .strip()
            .lower()
        )
        if transport not in (
            DEFAULT_VIDEO_TRANSPORT,
            ASTRBOT_FILE_SERVICE_TRANSPORT,
        ):
            logger.warning(
                "[MiMoMedia] 无效的 video_transport=%r，使用 Base64",
                transport,
            )
            return DEFAULT_VIDEO_TRANSPORT
        return transport

    def _ensure_reusable_file_handler(self) -> None:
        if not hasattr(file_token_service, _FILE_REGISTRY_ATTR):
            setattr(file_token_service, _FILE_REGISTRY_ATTR, {})
        if not hasattr(file_token_service, _FILE_OWNERS_ATTR):
            setattr(file_token_service, _FILE_OWNERS_ATTR, set())

        if not hasattr(file_token_service, _FILE_HANDLER_ATTR):
            original_handler = file_token_service.handle_file

            async def reusable_handle_file(file_token: str) -> str:
                registry = getattr(file_token_service, _FILE_REGISTRY_ATTR, {})
                entry = registry.get(file_token)
                if entry is not None:
                    file_path, expires_at, access_count = entry
                    if expires_at < time.time():
                        registry.pop(file_token, None)
                        raise KeyError(f"Invalid or expired file token: {file_token}")
                    if not os.path.exists(file_path):
                        registry.pop(file_token, None)
                        raise FileNotFoundError(f"File does not exist: {file_path}")
                    access_count += 1
                    registry[file_token] = (file_path, expires_at, access_count)
                    logger.info(
                        "[MiMoMedia] 文件服务视频被拉取，第 %d 次，token=%s...",
                        access_count,
                        file_token[:8],
                    )
                    return file_path

                original = getattr(file_token_service, _FILE_HANDLER_ATTR)
                return await original(file_token)

            setattr(file_token_service, _FILE_HANDLER_ATTR, original_handler)
            setattr(file_token_service, _FILE_WRAPPER_ATTR, reusable_handle_file)
            file_token_service.handle_file = reusable_handle_file

        if not self._file_handler_owner_registered:
            owners = getattr(file_token_service, _FILE_OWNERS_ATTR)
            owners.add(id(self))
            self._file_handler_owner_registered = True

    def _release_reusable_file_handler(self) -> None:
        if not self._file_handler_owner_registered:
            return
        owners = getattr(file_token_service, _FILE_OWNERS_ATTR, set())
        owners.discard(id(self))
        self._file_handler_owner_registered = False
        if owners:
            return

        wrapper = getattr(file_token_service, _FILE_WRAPPER_ATTR, None)
        original_handler = getattr(file_token_service, _FILE_HANDLER_ATTR, None)
        if wrapper is not None and file_token_service.handle_file is wrapper:
            file_token_service.handle_file = original_handler
        for attr in (
            _FILE_REGISTRY_ATTR,
            _FILE_HANDLER_ATTR,
            _FILE_WRAPPER_ATTR,
            _FILE_OWNERS_ATTR,
        ):
            try:
                delattr(file_token_service, attr)
            except AttributeError:
                pass

    async def _register_video_file(self, path: Path) -> tuple[str, str]:
        callback_base = str(astrbot_config.get("callback_api_base", "") or "").strip()
        callback_base = callback_base.rstrip("/")
        if not callback_base.startswith(("http://", "https://")):
            raise ValueError("未配置有效的 callback_api_base")

        self._ensure_reusable_file_handler()
        token = str(uuid.uuid4())
        registry = getattr(file_token_service, _FILE_REGISTRY_ATTR)
        registry[token] = (
            str(path),
            time.time() + FILE_SERVICE_TOKEN_TTL_SECONDS,
            0,
        )
        self._file_tokens.add(token)
        return f"{callback_base}/api/file/{token}", token

    def _remove_file_token(self, token: str) -> None:
        registry = getattr(file_token_service, _FILE_REGISTRY_ATTR, None)
        if isinstance(registry, dict):
            registry.pop(token, None)
        self._file_tokens.discard(token)

    def _schedule_served_file_cleanup(self, path: Path, token: str) -> None:
        path_str = str(path)
        self._served_files.add(path_str)

        async def cleanup_after_ttl() -> None:
            try:
                await asyncio.sleep(FILE_SERVICE_TOKEN_TTL_SECONDS)
            except asyncio.CancelledError:
                raise
            finally:
                self._remove_file_token(token)
                _cleanup_paths([path_str])
                self._served_files.discard(path_str)

        task = asyncio.create_task(cleanup_after_ttl())
        self._file_cleanup_tasks.add(task)
        task.add_done_callback(self._file_cleanup_tasks.discard)

    @staticmethod
    def _session_key(event: AstrMessageEvent) -> str:
        return str(
            getattr(event, "unified_msg_origin", None)
            or getattr(getattr(event, "message_obj", None), "session_id", None)
            or getattr(event, "session_id", None)
            or "default"
        )

    @classmethod
    def _walk_components(
        cls,
        components: list[Any] | None,
        *,
        reply_id: str | int | None = None,
        in_forward: bool = False,
        depth: int = 0,
    ) -> Iterator[tuple[Any, str | int | None, bool]]:
        """Recursively walk reply and forwarded-message component containers."""
        if not isinstance(components, list) or depth > MAX_COMPONENT_DEPTH:
            return
        for component in components:
            yield component, reply_id, in_forward
            if isinstance(component, Reply) and component.chain:
                yield from cls._walk_components(
                    component.chain,
                    reply_id=component.id,
                    in_forward=in_forward,
                    depth=depth + 1,
                )
            elif isinstance(component, Node):
                yield from cls._walk_components(
                    component.content,
                    reply_id=reply_id,
                    in_forward=True,
                    depth=depth + 1,
                )
            elif isinstance(component, Nodes):
                for node in component.nodes:
                    yield node, reply_id, True
                    yield from cls._walk_components(
                        node.content,
                        reply_id=reply_id,
                        in_forward=True,
                        depth=depth + 1,
                    )

    @classmethod
    def _component_entries(
        cls, event: AstrMessageEvent
    ) -> list[tuple[Any, str | int | None, bool]]:
        message_obj = getattr(event, "message_obj", None)
        entries = list(cls._walk_components(getattr(message_obj, "message", []) or []))
        entries.extend(event.get_extra("mimo_media_remote_entries") or [])
        return entries

    @classmethod
    def _message_components(cls, event: AstrMessageEvent) -> list[Any]:
        return [component for component, _, _ in cls._component_entries(event)]

    @classmethod
    def _parse_forward_payload(
        cls, payload: Any, *, include_audio: bool
    ) -> tuple[list[Image | Video | Record], list[str]]:
        """Extract media components and nested forward IDs from OneBot payloads."""
        media: list[Image | Video | Record] = []
        forward_ids: list[str] = []
        seen_media: set[tuple[str, str]] = set()

        def visit(value: Any, depth: int = 0, in_forward: bool = False) -> None:
            if depth > MAX_COMPONENT_DEPTH:
                return
            if isinstance(value, list):
                for item in value:
                    visit(item, depth + 1, in_forward)
                return
            if not isinstance(value, dict):
                return

            segment_type = str(value.get("type", "") or "").lower()
            segment_data = value.get("data")
            data = segment_data if isinstance(segment_data, dict) else value
            source = str(
                data.get("url") or data.get("file") or data.get("path") or ""
            ).strip()
            if segment_type in ("image", "img") and source:
                key = ("image", source)
                if key not in seen_media:
                    seen_media.add(key)
                    media.append(
                        Image(
                            file=source,
                            url=str(data.get("url") or ""),
                            path=str(data.get("path") or ""),
                        )
                    )
                return
            if segment_type in ("video", "shortvideo") and source:
                key = ("video", source)
                if key not in seen_media:
                    seen_media.add(key)
                    media.append(
                        Video(
                            file=source,
                            url=str(data.get("url") or ""),
                            path=str(data.get("path") or ""),
                        )
                    )
                return
            if (
                include_audio
                and not in_forward
                and segment_type in ("record", "audio", "voice")
                and source
            ):
                key = ("record", source)
                if key not in seen_media:
                    seen_media.add(key)
                    media.append(
                        Record(
                            file=source,
                            url=str(data.get("url") or ""),
                            path=str(data.get("path") or ""),
                        )
                    )
                return
            if segment_type in ("forward", "forward_msg", "nodes"):
                forward_id = data.get("id") or data.get("message_id")
                if forward_id is not None and str(forward_id).strip():
                    forward_ids.append(str(forward_id).strip())

            containers = [value]
            if data is not value:
                containers.append(data)
            nested_in_forward = in_forward or segment_type in ("node", "nodes")
            for container in containers:
                for key in ("messages", "message", "nodes", "content"):
                    nested = container.get(key)
                    if isinstance(nested, (list, dict)):
                        visit(nested, depth + 1, nested_in_forward)
                    elif isinstance(nested, str) and nested.strip().startswith(
                        ("[", "{")
                    ):
                        try:
                            visit(json.loads(nested), depth + 1, nested_in_forward)
                        except json.JSONDecodeError:
                            pass

        visit(payload)
        return media, list(dict.fromkeys(forward_ids))

    async def _resolve_remote_media(self, event: AstrMessageEvent) -> None:
        """Resolve media omitted from Reply chains and Forward ID components."""
        if event.get_extra("mimo_media_remote_resolved"):
            return
        event.set_extra("mimo_media_remote_resolved", True)

        local_entries = list(
            self._walk_components(
                getattr(getattr(event, "message_obj", None), "message", []) or []
            )
        )
        forward_refs = [
            (str(component.id).strip(), reply_id)
            for component, reply_id, _ in local_entries
            if isinstance(component, Forward) and str(component.id).strip()
        ]
        unresolved_replies: list[Reply] = []
        for component, _, _ in local_entries:
            if not isinstance(component, Reply) or not str(component.id).strip():
                continue
            embedded_media = any(
                isinstance(nested, (Image, Video, Record, Forward))
                for nested, _, _ in self._walk_components(component.chain or [])
            )
            if not embedded_media:
                unresolved_replies.append(component)

        if not forward_refs and not unresolved_replies:
            return

        client = OneBotClient(event)
        remote_entries: list[tuple[Any, str | int | None, bool]] = []
        for reply in unresolved_replies:
            payload = await client.get_msg(reply.id)
            if not payload:
                continue
            media, nested_ids = self._parse_forward_payload(payload, include_audio=True)
            remote_entries.extend((component, reply.id, False) for component in media)
            forward_refs.extend((forward_id, reply.id) for forward_id in nested_ids)

        pending = list(dict.fromkeys(forward_refs))
        seen_ids: set[tuple[str, str | int | None]] = set()
        while pending and len(seen_ids) < MAX_FORWARD_FETCH:
            forward_id, source_reply_id = pending.pop(0)
            forward_key = (forward_id, source_reply_id)
            if forward_key in seen_ids:
                continue
            seen_ids.add(forward_key)
            payload = await client.get_forward_msg(forward_id)
            if not payload:
                continue
            media, nested_ids = self._parse_forward_payload(
                payload, include_audio=False
            )
            remote_entries.extend(
                (component, source_reply_id, True) for component in media
            )
            pending.extend(
                (item, source_reply_id)
                for item in nested_ids
                if (item, source_reply_id) not in seen_ids
            )

        deduplicated_entries = []
        seen_entries: set[tuple[str, str, str, bool]] = set()
        for component, reply_id, in_forward in remote_entries:
            source = str(
                getattr(component, "url", None)
                or getattr(component, "file", None)
                or getattr(component, "path", None)
                or ""
            )
            key = (
                component.__class__.__name__,
                source,
                str(reply_id or ""),
                in_forward,
            )
            if key in seen_entries:
                continue
            seen_entries.add(key)
            deduplicated_entries.append((component, reply_id, in_forward))
        event.set_extra("mimo_media_remote_entries", deduplicated_entries)

    def _has_multimodal_message(self, event: AstrMessageEvent) -> bool:
        return any(
            isinstance(component, (Image, Video))
            or (isinstance(component, Record) and not in_forward)
            for component, _, in_forward in self._component_entries(event)
        )

    async def _merge_forward_images(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """Add only top-level forwarded images not handled by AstrBot core."""
        image_urls = list(req.image_urls or [])
        seen = {str(item) for item in image_urls}
        for component, reply_id, in_forward in self._component_entries(event):
            if (
                not isinstance(component, Image)
                or not in_forward
                or reply_id is not None
            ):
                continue
            try:
                image_path = await component.convert_to_file_path()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[MiMoMedia] 合并转发图片解析失败: %s", exc)
                continue
            if image_path and str(image_path) not in seen:
                image_urls.append(str(image_path))
                seen.add(str(image_path))

        req.image_urls = image_urls

    def _target_provider_available(self, purpose: str = "路由") -> bool:
        target_id = self._routing_provider_id()
        if not target_id:
            logger.warning("[MiMoMedia] 未配置用于%s的 MiMo provider", purpose)
            return False
        provider = self.context.get_provider_by_id(target_id)
        if provider is None:
            logger.error(
                "[MiMoMedia] 用于%s的 provider %r 不存在，请检查插件配置",
                purpose,
                target_id,
            )
            return False
        if not self._provider_is_mimo(provider):
            logger.error(
                "[MiMoMedia] 用于%s的 provider %r 不是 MiMo provider",
                purpose,
                target_id,
            )
            return False
        return True

    @filter.on_waiting_llm_request(priority=100)
    async def prepare_multimodal_routing(self, event: AstrMessageEvent):
        """Select MiMo only after AstrBot confirms this message will call an LLM."""
        try:
            if not self._enabled() or not self._routing_enabled():
                self._routing_remaining.clear()
                return

            await self._resolve_remote_media(event)
            key = self._session_key(event)
            remaining = self._routing_remaining.get(key, 0)
            has_multimodal = self._has_multimodal_message(event)
            if not has_multimodal:
                quoted_image_refs: list[str] = []
                for component in getattr(event.message_obj, "message", []) or []:
                    if not isinstance(component, Reply):
                        continue
                    try:
                        quoted_image_refs.extend(
                            await extract_quoted_message_images(event, component)
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[MiMoMedia] 引用消息图片解析失败: %s", exc)
                if quoted_image_refs:
                    event.set_extra(
                        "mimo_media_quoted_image_refs",
                        list(dict.fromkeys(quoted_image_refs)),
                    )
                    has_multimodal = True

            # A new multimodal LLM request refreshes the routing window. Plain LLM
            # follow-ups consume the remaining requests without extending it.
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
            event.set_extra("mimo_media_route_new_window", has_multimodal)
            logger.info(
                "[MiMoMedia] 已为待执行的 LLM 请求选择 %s",
                self._routing_provider_id(),
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
            mode = self._multimodal_mode()
            if event.get_extra("mimo_media_routed") and not event.get_extra(
                "mimo_media_route_consumed"
            ):
                key = self._session_key(event)
                if event.get_extra("mimo_media_route_new_window"):
                    remaining = self._routing_turns()
                else:
                    remaining = self._routing_remaining.get(key, 0)
                remaining = max(0, remaining - 1)
                if remaining:
                    self._routing_remaining[key] = remaining
                else:
                    self._routing_remaining.pop(key, None)
                event.set_extra("mimo_media_route_consumed", True)
                logger.info(
                    "[MiMoMedia] LLM 请求实际触发，本次后剩余 %d 轮 MiMo 路由",
                    remaining,
                )
            is_mimo_provider = self._is_mimo_provider(event)
            audio_mode = self._audio_mode()
            if mode == CAPTION_MULTIMODAL_MODE:
                await self._resolve_remote_media(event)
                if audio_mode == LLONEBOT_STT_AUDIO_MODE:
                    await self._handle(
                        event,
                        req,
                        process_videos=False,
                    )
                await self._caption_media(
                    event,
                    req,
                    process_audio=audio_mode == DEFAULT_AUDIO_MODE,
                )
                return
            if audio_mode != LLONEBOT_STT_AUDIO_MODE and not is_mimo_provider:
                return
            if is_mimo_provider or audio_mode == LLONEBOT_STT_AUDIO_MODE:
                await self._resolve_remote_media(event)
            if is_mimo_provider:
                await self._merge_forward_images(event, req)
            await self._handle(event, req, process_videos=is_mimo_provider)
        except Exception as exc:  # noqa: BLE001
            logger.error("[MiMoMedia] on_llm_request 处理失败: %s", exc, exc_info=True)

    async def _handle(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        *,
        process_videos: bool = True,
        process_audio: bool = True,
    ) -> None:
        videos = self._collect_video_components(event) if process_videos else []
        request_audio_refs = list(req.audio_urls or []) if process_audio else []
        audio_targets = self._collect_audio_targets(event) if process_audio else []
        audio_records = [record for record, _, _ in audio_targets]
        audio_mode = self._audio_mode()
        audio_refs: list[str] = []
        transcriptions: list[tuple[str, bool]] = []
        notes: list[str] = []
        max_video_count = self._max_video_count()
        if len(videos) > max_video_count:
            notes.append(
                f"[本次包含 {len(videos)} 个视频，仅处理前 {max_video_count} 个]"
            )
            videos = videos[:max_video_count]

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

    async def _caption_media(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        *,
        process_audio: bool,
    ) -> None:
        """Use the configured MiMo provider to caption video and audio as text."""
        videos = self._collect_video_components(event)
        has_audio = bool(self._collect_audio_targets(event) or req.audio_urls)
        if not videos and (not process_audio or not has_audio):
            return

        caption_req = ProviderRequest(
            prompt=str(
                self._cfg(
                    "media_caption_prompt",
                    "请详细转述视频和音频中的内容，包括画面、对话、声音和关键事件。",
                )
            ),
            audio_urls=list(req.audio_urls or []),
            extra_user_content_parts=[],
        )
        await self._handle(
            event,
            caption_req,
            process_videos=True,
            process_audio=process_audio,
        )
        media_parts = [
            part
            for part in caption_req.extra_user_content_parts
            if isinstance(part, (VideoURLPart, InputAudioPart))
        ]
        notes = [
            part
            for part in caption_req.extra_user_content_parts
            if isinstance(part, TextPart) and part.text.startswith("[")
        ]
        self._drop_placeholders(req, drop_video=True)
        req.audio_urls = []
        if not media_parts:
            req.extra_user_content_parts.extend(notes)
            return
        req.extra_user_content_parts.extend(notes)
        if not self._target_provider_available("转述"):
            req.extra_user_content_parts.append(TextPart(text="[视频/音频转述失败]"))
            return

        provider = self.context.get_provider_by_id(self._routing_provider_id())
        if not isinstance(provider, Provider):
            req.extra_user_content_parts.append(TextPart(text="[视频/音频转述失败]"))
            return
        try:
            # Reuse the direct-mode assembly path so custom MiMo content parts are
            # converted to raw context dictionaries before provider validation.
            caption_context = Message.model_validate(
                await caption_req.assemble_context()
            )
            response = await provider.text_chat(
                contexts=[caption_context],
            )
            caption = str(response.completion_text or "").strip()
            if caption:
                req.extra_user_content_parts.append(
                    TextPart(text=f"<media_caption>{caption}</media_caption>")
                )
            else:
                req.extra_user_content_parts.append(
                    TextPart(text="[视频/音频转述失败]")
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("[MiMoMedia] 视频/音频转述失败: %s", exc, exc_info=True)
            req.extra_user_content_parts.append(TextPart(text="[视频/音频转述失败]"))

    @classmethod
    def _collect_video_components(cls, event: AstrMessageEvent) -> list[Video]:
        return [
            component
            for component, _, _ in cls._component_entries(event)
            if isinstance(component, Video)
        ]

    @classmethod
    def _collect_audio_components(cls, event: AstrMessageEvent) -> list[Record]:
        return [
            component
            for component, _, in_forward in cls._component_entries(event)
            if isinstance(component, Record) and not in_forward
        ]

    @classmethod
    def _collect_audio_targets(
        cls,
        event: AstrMessageEvent,
    ) -> list[tuple[Record, str | int | None, bool]]:
        raw_message = getattr(event.message_obj, "raw_message", None)
        direct_message_id = None
        if isinstance(raw_message, dict):
            direct_message_id = raw_message.get("message_id") or raw_message.get("id")
        if direct_message_id is None:
            direct_message_id = getattr(event.message_obj, "message_id", None)

        targets: list[tuple[Record, str | int | None, bool]] = []
        for component, reply_id, in_forward in cls._component_entries(event):
            if not isinstance(component, Record) or in_forward:
                continue
            message_id = reply_id or direct_message_id
            targets.append((component, message_id, reply_id is not None))
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

            always_compress = bool(self._cfg("video_always_compress", False))
            max_width = int(self._cfg("max_video_width", DEFAULT_MAX_VIDEO_WIDTH))
            if always_compress:
                out = await _to_h264_mp4(src, max_width=max_width, crf=28)
            else:
                out = await _to_h264_mp4(src)
            cleanup_paths.append(str(out))

            if self._video_transport() == ASTRBOT_FILE_SERVICE_TRANSPORT:
                try:
                    public_url, token = await self._register_video_file(out)
                    self._schedule_served_file_cleanup(out, token)
                    cleanup_paths.remove(str(out))

                    fps = float(self._cfg("video_fps", DEFAULT_FPS))
                    fps = max(0.1, min(10.0, fps))
                    resolution = str(self._cfg("video_resolution", DEFAULT_RESOLUTION))
                    if resolution not in ("default", "max"):
                        resolution = DEFAULT_RESOLUTION

                    logger.info(
                        "[MiMoMedia] 已通过 AstrBot 文件服务提供视频，链接有效期 %d 秒",
                        FILE_SERVICE_TOKEN_TTL_SECONDS,
                    )
                    return (
                        VideoURLPart(
                            video_url=VideoURLPart.VideoURL(url=public_url),
                            fps=fps,
                            media_resolution=resolution,
                        ),
                        None,
                        cleanup_paths,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[MiMoMedia] AstrBot 文件服务不可用，视频已跳过: %s",
                        exc,
                    )
                    return (
                        None,
                        "[AstrBot 文件服务不可用，视频已跳过]",
                        cleanup_paths,
                    )

            max_mb = float(self._cfg("video_max_base64_mb", DEFAULT_MAX_BASE64_MB))
            data_url = _bytes_to_data_url(out.read_bytes(), VIDEO_MIME)
            if _base64_size_mb(data_url) > max_mb and not always_compress:
                # 超限：压缩一次（缩放到 max_video_width + CRF 28）
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
        tasks = list(self._file_cleanup_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._file_cleanup_tasks.clear()
        for token in list(self._file_tokens):
            self._remove_file_token(token)
        _cleanup_paths(list(self._served_files))
        self._served_files.clear()
        self._release_reusable_file_handler()
