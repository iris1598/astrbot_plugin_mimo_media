"""astrbot_plugin_mimo_media 单元验证。

运行方式（在 AstrBot 仓库 venv 下）：
    python -m pytest astrbot_plugin_mimo_media/tests -v
"""

import subprocess
import sys
from pathlib import Path

# The plugin and AstrBot source directories are added to sys.path below.
# ruff: noqa: E402

PLUGIN_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PLUGIN_DIR.parent
ASTRBOT_DIR = REPO_DIR / "AstrBot"
for _p in (str(PLUGIN_DIR), str(ASTRBOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest

import main
from astrbot.core.agent.message import (
    Message,
    TextPart,
    dump_messages_with_checkpoints,
)
from astrbot.core.message.components import (
    Forward,
    Image,
    Node,
    Nodes,
    Record,
    Reply,
    Video,
)
from astrbot.core.provider.entities import ProviderRequest
from main import (
    MiMoMediaPlugin,
    InputAudioPart,
    VideoURLPart,
    _base64_size_mb,
    _bytes_to_data_url,
    _to_h264_mp4,
    _to_standard_wav,
    _validate_wav_data_url,
)


def _run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {args}\n{proc.stderr}")
    return proc.stdout.strip()


def _ffprobe_entry(file: Path, selector: str, entry: str) -> str:
    return _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            selector,
            "-show_entries",
            f"stream={entry}",
            "-of",
            "csv=p=0",
            str(file),
        ]
    )


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    src = tmp_path / "source.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(src),
        ]
    )
    return src


@pytest.fixture()
def sample_webm(tmp_path: Path) -> Path:
    src = tmp_path / "source.webm"
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=10",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "200k",
            "-f",
            "webm",
            str(src),
        ]
    )
    return src


@pytest.fixture()
def sample_wav(tmp_path: Path) -> Path:
    src = tmp_path / "tone.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(src),
        ]
    )
    return src


@pytest.fixture()
def sample_mp3(tmp_path: Path) -> Path:
    src = tmp_path / "tone.mp3"
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            str(src),
        ]
    )
    return src


# ---- 视频 H264 转码 ----


@pytest.mark.asyncio
async def test_h264_conversion_always_reencores(sample_video: Path, tmp_path: Path):
    out = await _to_h264_mp4(str(sample_video))
    try:
        assert out.suffix == ".mp4"
        assert _ffprobe_entry(out, "v:0", "codec_name") == "h264"
        assert _ffprobe_entry(out, "a:0", "codec_name") == "aac"
    finally:
        out.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_h264_conversion_from_webm(sample_webm: Path):
    """非 mp4 源也能被强制重编码为标准 H264。"""
    out = await _to_h264_mp4(str(sample_webm))
    try:
        assert _ffprobe_entry(out, "v:0", "codec_name") == "h264"
    finally:
        out.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_h264_downscale_pass(sample_video: Path):
    out = await _to_h264_mp4(str(sample_video), max_width=160, crf=28)
    try:
        width = int(_ffprobe_entry(out, "v:0", "width"))
        assert width <= 160
        assert _ffprobe_entry(out, "v:0", "codec_name") == "h264"
    finally:
        out.unlink(missing_ok=True)


# ---- 音频 WAV 转换与校验 ----


@pytest.mark.asyncio
async def test_audio_to_standard_wav_with_ffmpeg(sample_mp3: Path):
    out = await _to_standard_wav(str(sample_mp3))
    try:
        assert out.suffix == ".wav"
        assert _ffprobe_entry(out, "a:0", "codec_name") == "pcm_s16le"
        assert _ffprobe_entry(out, "a:0", "sample_rate") == "16000"
        assert _ffprobe_entry(out, "a:0", "channels") == "1"
        data_url = _bytes_to_data_url(out.read_bytes(), "audio/wav")
        _validate_wav_data_url(data_url, str(sample_mp3))
    finally:
        out.unlink(missing_ok=True)


def test_validate_wav_rejects_garbage():
    bad = _bytes_to_data_url(b"hello world not a wav file", "audio/wav")
    with pytest.raises(ValueError):
        _validate_wav_data_url(bad, "/tmp/bad.wav")


def test_base64_size_mb():
    payload = _bytes_to_data_url(b"x" * 1024 * 1024, "video/mp4")
    assert abs(_base64_size_mb(payload) - (4.0 / 3.0)) < 0.05


# ---- ContentPart 序列化 ----


def test_video_url_part_serialization():
    part = VideoURLPart(
        video_url=VideoURLPart.VideoURL(url="data:video/mp4;base64,AAAA"),
        fps=2.0,
        media_resolution="default",
    )
    assert part.model_dump() == {
        "type": "video_url",
        "video_url": {"url": "data:video/mp4;base64,AAAA"},
        "fps": 2.0,
        "media_resolution": "default",
    }


def test_input_audio_part_serialization():
    part = InputAudioPart(
        input_audio=InputAudioPart.InputAudio(data="data:audio/wav;base64,BBBB")
    )
    assert part.model_dump() == {
        "type": "input_audio",
        "input_audio": {"data": "data:audio/wav;base64,BBBB"},
    }


def test_mark_as_temp_excludes_from_history():
    part = VideoURLPart(
        video_url=VideoURLPart.VideoURL(url="data:video/mp4;base64,AAAA"),
    ).mark_as_temp()
    msg = Message(role="user", content=[TextPart(text="hi"), part])
    dumped = dump_messages_with_checkpoints([msg])
    content = dumped[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"


# ---- 端到端：assemble_context -> Message 校验与透传 ----


@pytest.mark.asyncio
async def test_assemble_context_injection_roundtrip():
    video_part = VideoURLPart(
        video_url=VideoURLPart.VideoURL(url="data:video/mp4;base64,AAAA"),
        fps=2.0,
        media_resolution="max",
    ).mark_as_temp()
    audio_part = InputAudioPart(
        input_audio=InputAudioPart.InputAudio(data="data:audio/wav;base64,BBBB")
    ).mark_as_temp()
    req = ProviderRequest(
        prompt="describe this",
        image_urls=[],
        audio_urls=[],
        extra_user_content_parts=[video_part, audio_part],
    )
    user_message = await req.assemble_context()
    assert user_message["role"] == "user"
    types = [block.get("type") for block in user_message["content"]]
    assert types == ["text", "video_url", "input_audio"]

    # 消息能通过 runner 的 Message.model_validate（依赖插件注册的自定义 ContentPart）
    msg = Message.model_validate(user_message)
    assert any(isinstance(p, VideoURLPart) for p in msg.content)
    assert any(isinstance(p, InputAudioPart) for p in msg.content)

    # 历史持久化时，临时媒体块不应写入
    dumped = dump_messages_with_checkpoints([msg])
    saved_types = [b.get("type") for b in dumped[0]["content"] if isinstance(b, dict)]
    assert "video_url" not in saved_types
    assert "input_audio" not in saved_types


# ---- 占位符移除 ----


def test_drop_placeholders():
    req = ProviderRequest(
        extra_user_content_parts=[
            TextPart(text="[Video Attachment: name v.mp4, path C:/tmp/v.mp4]"),
            TextPart(text="[Audio Attachment: path C:/tmp/a.wav]"),
            TextPart(text="[Video Attachment in quoted message: name q.mp4, path x]"),
            TextPart(text="hello"),
        ],
    )
    MiMoMediaPlugin._drop_placeholders(req)
    assert len(req.extra_user_content_parts) == 1
    assert req.extra_user_content_parts[0].text == "hello"


def test_drop_audio_placeholder_keeps_video_for_non_mimo_provider():
    req = ProviderRequest(
        extra_user_content_parts=[
            TextPart(text="[Video Attachment: name v.mp4, path C:/tmp/v.mp4]"),
            TextPart(text="[Audio Attachment: path C:/tmp/a.wav]"),
        ],
    )

    MiMoMediaPlugin._drop_placeholders(req, drop_video=False)

    assert [part.text for part in req.extra_user_content_parts] == [
        "[Video Attachment: name v.mp4, path C:/tmp/v.mp4]"
    ]


def test_replace_empty_text_for_quoted_audio():
    req = ProviderRequest(
        extra_user_content_parts=[
            TextPart(
                text=(
                    "<Quoted Message>\n"
                    "(Iris-カラーアイリス): [Empty Text]\n"
                    "</Quoted Message>"
                )
            ),
            TextPart(text="[Empty Text]"),
        ],
    )

    MiMoMediaPlugin._replace_empty_text_for_quoted_audio(req)

    assert req.extra_user_content_parts[0].text == (
        "<Quoted Message>\n(Iris-カラーアイリス): [Audio]\n</Quoted Message>"
    )
    assert req.extra_user_content_parts[1].text == "[Empty Text]"


def test_audio_mode_defaults_to_multimodal():
    assert _plugin()._audio_mode() == "multimodal"
    assert _plugin({"audio_mode": "llonebot_stt"})._audio_mode() == "llonebot_stt"
    assert _plugin({"audio_mode": "invalid"})._audio_mode() == "multimodal"


class _FakeBot:
    def __init__(self, text: str = ""):
        self.text = text
        self.calls = []

    async def call_action(self, **kwargs):
        self.calls.append(kwargs)
        return {"data": {"text": self.text}}


@pytest.mark.asyncio
async def test_llonebot_mode_replaces_quoted_empty_text():
    plugin = _plugin({"audio_mode": "llonebot_stt"})
    record = Record(file="quote.amr")
    reply = Reply(
        id=456,
        chain=[record],
        sender_nickname="Iris-カラーアイリス",
    )
    event = _FakeEvent([reply], audio_urls=[])
    event.bot = _FakeBot("这是语音转写")
    event.message_obj.raw_message = {"message_id": 999}
    req = ProviderRequest(
        extra_user_content_parts=[
            TextPart(
                text=(
                    "<Quoted Message>\n"
                    "(Iris-カラーアイリス): [Empty Text]\n"
                    "</Quoted Message>"
                )
            ),
            TextPart(text="[Audio Attachment in quoted message: path quote.amr]"),
        ]
    )

    await plugin._handle(event, req)

    assert event.bot.calls == [{"action": "voice_msg_to_text", "message_id": 456}]
    texts = [
        part.text for part in req.extra_user_content_parts if hasattr(part, "text")
    ]
    assert "这是语音转写" not in texts
    assert any("(Iris-カラーアイリス): 这是语音转写" in text for text in texts)
    assert all("Audio Attachment" not in text for text in texts)
    assert not any(
        isinstance(part, InputAudioPart) for part in req.extra_user_content_parts
    )


# ---- 插件实例级处理（音频超限/失败提示） ----


def _plugin(config: dict | None = None, context=None) -> MiMoMediaPlugin:
    return MiMoMediaPlugin(context=context, config=config or {})


@pytest.mark.asyncio
async def test_process_audio_oversize_returns_note(sample_wav: Path):
    plugin = _plugin({"audio_max_base64_mb": 0.0001})
    part, note, paths = await plugin._process_audio(str(sample_wav))
    assert part is None
    assert note is not None and "过大" in note


@pytest.mark.asyncio
async def test_process_audio_success(sample_mp3: Path):
    plugin = _plugin({})
    part, note, paths = await plugin._process_audio(str(sample_mp3))
    try:
        assert part is not None
        assert note is None
        assert isinstance(part, InputAudioPart)
        assert part.input_audio.data.startswith("data:audio/wav;base64,")
        assert _ffprobe_entry(Path(paths[-1]), "a:0", "codec_name") == "pcm_s16le"
    finally:
        main._cleanup_paths(paths)


@pytest.mark.asyncio
async def test_process_video_success(sample_video: Path):
    plugin = _plugin({})
    video = Video.fromFileSystem(path=str(sample_video))
    part, note, paths = await plugin._process_video(video)
    assert part is not None
    assert note is None
    assert isinstance(part, VideoURLPart)
    assert part.video_url.url.startswith("data:video/mp4;base64,")
    # 插件应清理自身创建的临时文件
    assert len(paths) >= 1
    main._cleanup_paths(paths)
    for p in paths:
        assert not Path(p).exists()


@pytest.mark.asyncio
async def test_process_video_always_compresses(sample_video: Path):
    plugin = _plugin({"video_always_compress": True, "max_video_width": 160})
    video = Video.fromFileSystem(path=str(sample_video))

    part, note, paths = await plugin._process_video(video)
    try:
        assert part is not None
        assert note is None
        assert _ffprobe_entry(Path(paths[-1]), "v:0", "width") == "160"
    finally:
        main._cleanup_paths(paths)


@pytest.mark.asyncio
async def test_process_video_uses_astrbot_file_service(sample_video: Path, monkeypatch):
    class FakeFileTokenService:
        async def handle_file(self, token):
            raise KeyError(token)

    fake_service = FakeFileTokenService()
    monkeypatch.setattr(
        main,
        "astrbot_config",
        {"callback_api_base": "https://bot.example.com/"},
    )
    monkeypatch.setattr(main, "file_token_service", fake_service)
    plugin = _plugin({"video_transport": "astrbot_file_service"})
    video = Video.fromFileSystem(path=str(sample_video))

    part, note, paths = await plugin._process_video(video)
    token = part.video_url.url.rsplit("/", 1)[-1]
    served_path = Path(await fake_service.handle_file(token))
    try:
        assert note is None
        assert part.video_url.url.startswith("https://bot.example.com/api/file/")
        assert await fake_service.handle_file(token) == str(served_path)
        assert str(served_path) not in paths
        assert served_path.exists()
    finally:
        main._cleanup_paths(paths)
        await plugin.terminate()

    assert not served_path.exists()


@pytest.mark.asyncio
async def test_process_video_file_service_failure_skips_video(
    sample_video: Path, monkeypatch
):
    monkeypatch.setattr(main, "astrbot_config", {"callback_api_base": ""})
    plugin = _plugin({"video_transport": "astrbot_file_service"})
    video = Video.fromFileSystem(path=str(sample_video))

    part, note, paths = await plugin._process_video(video)
    try:
        assert part is None
        assert note is not None and "文件服务不可用" in note
    finally:
        main._cleanup_paths(paths)


@pytest.mark.asyncio
async def test_process_video_oversize_two_pass_then_note(sample_video: Path):
    plugin = _plugin({"video_max_base64_mb": 0.0001, "max_video_width": 64})
    video = Video.fromFileSystem(path=str(sample_video))
    part, note, paths = await plugin._process_video(video)
    assert part is None
    assert note is not None and "过大" in note
    assert len(paths) >= 1
    main._cleanup_paths(paths)
    for p in paths:
        assert not Path(p).exists()


# ---- Routing and _handle ----


class _FakeEvent:
    def __init__(self, chain, audio_urls=None, umo="test:user@test"):
        self.unified_msg_origin = umo
        self.message_obj = type(
            "Obj",
            (),
            {"message": chain, "session_id": umo, "raw_message": None},
        )()
        self.audio_urls = audio_urls or []
        self._extras = {}

    def get_extra(self, key):
        return self._extras.get(key)

    def set_extra(self, key, value):
        self._extras[key] = value


class _FakeProvider:
    def __init__(self, model="text-model", provider_type="openai_chat_completion"):
        self.model = model
        self.provider_config = {"type": provider_type, "api_base": ""}
        self.chat_calls = []

    def get_model(self):
        return self.model

    async def text_chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return type("Response", (), {"completion_text": "媒体转述结果"})()


class _FakeContext:
    def __init__(self):
        self.providers = {
            "original": _FakeProvider(),
            "mimo": _FakeProvider(
                model="mimo-v2.5",
                provider_type="xiaomi_chat_completion",
            ),
        }

    def get_provider_by_id(self, provider_id):
        return self.providers.get(provider_id)

    def get_using_provider(self, umo=None):
        return self.providers["original"]


class _FakeOneBotApi:
    def __init__(self, forward_payloads):
        self.forward_payloads = forward_payloads
        self.calls = []

    async def call_action(self, action, **params):
        self.calls.append((action, params))
        forward_id = str(params.get("message_id") or params.get("id") or "")
        return self.forward_payloads.get(forward_id)


class _FakeOneBot:
    def __init__(self, forward_payloads, stt_text=""):
        self.api = _FakeOneBotApi(forward_payloads)
        self.stt_text = stt_text
        self.stt_calls = []

    async def call_action(self, **kwargs):
        self.stt_calls.append(kwargs)
        return {"data": {"text": self.stt_text}}


@pytest.mark.parametrize(
    "component",
    [
        Image(file="image.jpg"),
        Video(file="video.mp4"),
        Record(file="audio.wav"),
        Reply(id=1, chain=[Image(file="quoted.jpg")]),
    ],
)
def test_detects_multimodal_components_and_quoted_media(component):
    plugin = _plugin()
    assert plugin._has_multimodal_message(_FakeEvent([component]))


def test_recursively_detects_media_inside_reply_and_forward_nodes():
    plugin = _plugin()
    record = Record(file="nested.wav")
    event = _FakeEvent(
        [
            Reply(
                id=456,
                chain=[
                    Nodes(
                        [
                            Node(
                                content=[
                                    Image(file="nested.jpg"),
                                    Video(file="nested.mp4"),
                                    record,
                                ]
                            )
                        ]
                    )
                ],
            )
        ]
    )

    assert plugin._has_multimodal_message(event)
    assert [video.file for video in plugin._collect_video_components(event)] == [
        "nested.mp4"
    ]
    assert plugin._collect_audio_components(event) == []
    assert plugin._collect_audio_targets(event) == []


@pytest.mark.asyncio
async def test_forward_node_images_are_added_to_provider_request(monkeypatch):
    async def fake_convert_to_file_path(image):
        return f"resolved/{image.file}"

    monkeypatch.setattr(Image, "convert_to_file_path", fake_convert_to_file_path)
    plugin = _plugin()
    event = _FakeEvent([Nodes([Node(content=[Image(file="nested.jpg")])])])
    req = ProviderRequest(image_urls=[])

    await plugin._merge_forward_images(event, req)

    assert req.image_urls == ["resolved/nested.jpg"]


@pytest.mark.asyncio
async def test_forward_id_resolves_remote_multimodal_components_for_routing():
    context = _FakeContext()
    plugin = _plugin(
        {
            "multimodal_mode": "route",
            "multimodal_provider_id": "mimo",
        },
        context=context,
    )
    event = _FakeEvent([Forward(id="forward-1")])
    event.bot = _FakeOneBot(
        {
            "forward-1": {
                "data": {
                    "messages": [
                        {
                            "content": [
                                {"type": "image", "data": {"file": "a.jpg"}},
                                {"type": "video", "data": {"file": "b.mp4"}},
                                {"type": "record", "data": {"file": "c.amr"}},
                            ]
                        }
                    ]
                }
            }
        }
    )

    await plugin.prepare_multimodal_routing(event)

    assert event.get_extra("selected_provider") == "mimo"
    assert [video.file for video in plugin._collect_video_components(event)] == [
        "b.mp4"
    ]
    assert plugin._collect_audio_components(event) == []
    assert event.bot.api.calls[0][0] == "get_forward_msg"


@pytest.mark.asyncio
async def test_reply_images_are_not_duplicated_by_plugin():
    context = _FakeContext()
    plugin = _plugin(
        {
            "multimodal_mode": "route",
            "multimodal_provider_id": "mimo",
        },
        context=context,
    )
    event = _FakeEvent([Reply(id=789, chain=[])])
    event.bot = _FakeOneBot(
        {
            "789": {
                "data": {
                    "message": [
                        {"type": "image", "data": {"file": "reply.jpg"}},
                        {"type": "video", "data": {"file": "reply.mp4"}},
                        {"type": "record", "data": {"file": "reply.amr"}},
                    ]
                }
            }
        }
    )
    req = ProviderRequest(image_urls=["official/reply.jpg"])

    await plugin.prepare_multimodal_routing(event)
    await plugin._merge_forward_images(event, req)

    assert event.get_extra("selected_provider") == "mimo"
    assert req.image_urls == ["official/reply.jpg"]
    audio_targets = plugin._collect_audio_targets(event)
    assert len(audio_targets) == 1
    assert audio_targets[0][1:] == ("789", True)
    assert event.bot.api.calls[0][0] == "get_msg"


@pytest.mark.asyncio
async def test_video_count_limit_processes_only_configured_number(monkeypatch):
    plugin = _plugin({"video_max_count": 2})
    videos = [Video(file=f"video-{index}.mp4") for index in range(4)]
    event = _FakeEvent(videos)
    req = ProviderRequest(prompt="分析这些视频")
    processed = []

    async def fake_process_video(video):
        processed.append(video.file)
        return None, None, []

    monkeypatch.setattr(plugin, "_process_video", fake_process_video)

    await plugin._handle(event, req)

    assert processed == ["video-0.mp4", "video-1.mp4"]
    texts = [
        part.text for part in req.extra_user_content_parts if hasattr(part, "text")
    ]
    assert texts == ["[本次包含 4 个视频，仅处理前 2 个]"]


@pytest.mark.asyncio
async def test_llonebot_stt_resolves_reply_id_audio_with_non_mimo_provider():
    context = _FakeContext()
    plugin = _plugin({"audio_mode": "llonebot_stt"}, context=context)
    event = _FakeEvent([Reply(id=321, chain=[])])
    event.bot = _FakeOneBot(
        {
            "321": {
                "data": {"message": [{"type": "record", "data": {"file": "reply.amr"}}]}
            }
        },
        stt_text="远程引用语音转写成功",
    )
    req = ProviderRequest(
        extra_user_content_parts=[
            TextPart(text=("<Quoted Message>\n(user): [Empty Text]\n</Quoted Message>"))
        ]
    )

    await plugin.on_llm_request(event, req)

    assert event.bot.stt_calls == [{"action": "voice_msg_to_text", "message_id": 321}]
    assert "远程引用语音转写成功" in req.extra_user_content_parts[0].text


@pytest.mark.asyncio
async def test_routing_disabled_preserves_original_logic():
    context = _FakeContext()
    plugin = _plugin(
        {
            "multimodal_mode": "direct",
            "multimodal_provider_id": "mimo",
        },
        context=context,
    )
    event = _FakeEvent([Image(file="image.jpg")])

    await plugin.prepare_multimodal_routing(event)

    assert event.get_extra("selected_provider") is None
    assert plugin._routing_remaining == {}


def test_multimodal_mode_supports_new_values_and_legacy_routing_switch():
    assert _plugin()._multimodal_mode() == "direct"
    assert _plugin({"multimodal_mode": "route"})._multimodal_mode() == "route"
    assert _plugin({"multimodal_mode": "caption"})._multimodal_mode() == "caption"
    assert _plugin({"multimodal_routing_enabled": True})._multimodal_mode() == "route"


@pytest.mark.asyncio
async def test_caption_mode_injects_media_caption_without_routing_images(monkeypatch):
    context = _FakeContext()
    plugin = _plugin(
        {
            "multimodal_mode": "caption",
            "multimodal_provider_id": "mimo",
            "media_caption_prompt": "转述这些媒体",
        },
        context=context,
    )
    event = _FakeEvent([Image(file="official.jpg"), Video(file="video.mp4")])
    req = ProviderRequest(
        image_urls=["official.jpg"],
        extra_user_content_parts=[
            TextPart(text="[Video Attachment: name video.mp4, path video.mp4]")
        ],
    )

    async def fake_handle(event, caption_req, **kwargs):
        caption_req.extra_user_content_parts.append(
            VideoURLPart(
                video_url=VideoURLPart.VideoURL(url="data:video/mp4;base64,AAAA")
            )
        )

    monkeypatch.setattr(plugin, "_handle", fake_handle)
    monkeypatch.setattr(main, "Provider", _FakeProvider)

    await plugin._caption_media(event, req, process_audio=True)

    assert req.image_urls == ["official.jpg"]
    caption_call = context.providers["mimo"].chat_calls[0]
    assert "extra_user_content_parts" not in caption_call
    caption_context = caption_call["contexts"][0]
    assert isinstance(caption_context, Message)
    assert caption_context.role == "user"
    assert [part.type for part in caption_context.content] == ["text", "video_url"]
    assert caption_context.content[0].text == "转述这些媒体"
    assert "_no_save" not in caption_context.model_dump()["content"][1]
    assert all(
        not isinstance(part, VideoURLPart) for part in req.extra_user_content_parts
    )
    assert [part.text for part in req.extra_user_content_parts] == [
        "<media_caption>媒体转述结果</media_caption>"
    ]


@pytest.mark.asyncio
async def test_caption_mode_does_not_route_image_only_messages():
    context = _FakeContext()
    plugin = _plugin(
        {
            "multimodal_mode": "caption",
            "multimodal_provider_id": "mimo",
        },
        context=context,
    )
    event = _FakeEvent([Image(file="image.jpg")])

    await plugin.prepare_multimodal_routing(event)

    assert event.get_extra("selected_provider") is None


@pytest.mark.asyncio
async def test_cancelled_multimodal_llm_does_not_start_routing_window():
    context = _FakeContext()
    plugin = _plugin(
        {
            "multimodal_mode": "route",
            "multimodal_provider_id": "mimo",
            "multimodal_route_turns": 3,
        },
        context=context,
    )
    cancelled_event = _FakeEvent([Image(file="image.jpg")])
    next_llm_event = _FakeEvent([])

    await plugin.prepare_multimodal_routing(cancelled_event)
    # Simulate cancellation before on_llm_request is emitted.
    await plugin.prepare_multimodal_routing(next_llm_event)

    assert cancelled_event.get_extra("selected_provider") == "mimo"
    assert next_llm_event.get_extra("selected_provider") is None
    assert plugin._routing_remaining == {}


@pytest.mark.asyncio
async def test_default_routing_only_selects_mimo_for_triggered_llm():
    context = _FakeContext()
    plugin = _plugin(
        {
            "multimodal_mode": "route",
            "multimodal_provider_id": "mimo",
        },
        context=context,
    )
    media_event = _FakeEvent([Image(file="image.jpg")])
    follow_up = _FakeEvent([])

    await plugin.prepare_multimodal_routing(media_event)
    await plugin.on_llm_request(media_event, ProviderRequest(prompt="test"))
    await plugin.prepare_multimodal_routing(follow_up)

    assert media_event.get_extra("selected_provider") == "mimo"
    assert media_event.get_extra("mimo_media_routed") is True
    assert follow_up.get_extra("selected_provider") is None
    assert plugin._routing_remaining == {}


@pytest.mark.asyncio
async def test_configured_routing_turns_then_returns_to_original_provider():
    context = _FakeContext()
    plugin = _plugin(
        {
            "multimodal_mode": "route",
            "multimodal_provider_id": "mimo",
            "multimodal_route_turns": 3,
        },
        context=context,
    )
    events = [
        _FakeEvent([Image(file="image.jpg")]),
        _FakeEvent([]),
        _FakeEvent([]),
        _FakeEvent([]),
    ]

    for event in events[:3]:
        await plugin.prepare_multimodal_routing(event)
        await plugin.on_llm_request(event, ProviderRequest(prompt="test"))
    await plugin.prepare_multimodal_routing(events[3])

    assert [event.get_extra("selected_provider") for event in events] == [
        "mimo",
        "mimo",
        "mimo",
        None,
    ]
    assert plugin._routing_remaining == {}


@pytest.mark.asyncio
async def test_message_without_llm_does_not_consume_active_routing_window():
    context = _FakeContext()
    plugin = _plugin(
        {
            "multimodal_mode": "route",
            "multimodal_provider_id": "mimo",
            "multimodal_route_turns": 2,
        },
        context=context,
    )
    media_event = _FakeEvent([Image(file="image.jpg")])
    intercepted_event = _FakeEvent([])
    llm_event = _FakeEvent([])
    restored_event = _FakeEvent([])

    await plugin.prepare_multimodal_routing(media_event)
    await plugin.on_llm_request(media_event, ProviderRequest(prompt="test"))
    assert plugin._routing_remaining == {"test:user@test": 1}

    # A later waiting hook may still cancel the request before on_llm_request.
    await plugin.prepare_multimodal_routing(intercepted_event)
    assert intercepted_event.get_extra("selected_provider") == "mimo"
    assert plugin._routing_remaining == {"test:user@test": 1}

    await plugin.prepare_multimodal_routing(llm_event)
    await plugin.on_llm_request(llm_event, ProviderRequest(prompt="test"))
    await plugin.prepare_multimodal_routing(restored_event)

    assert llm_event.get_extra("selected_provider") == "mimo"
    assert restored_event.get_extra("selected_provider") is None
    assert plugin._routing_remaining == {}


def test_selected_routing_provider_is_used_for_mimo_detection():
    context = _FakeContext()
    plugin = _plugin(context=context)
    event = _FakeEvent([Record(file="audio.wav")])
    event.set_extra("selected_provider", "mimo")

    assert plugin._is_mimo_provider(event)


@pytest.mark.asyncio
async def test_llonebot_stt_runs_with_non_mimo_provider_and_keeps_video_hint():
    context = _FakeContext()
    plugin = _plugin({"audio_mode": "llonebot_stt"}, context=context)
    event = _FakeEvent(
        [
            Video(file="video.mp4"),
            Record(file="voice.amr"),
        ]
    )
    event.bot = _FakeBot("任意模型都能收到这段转写")
    event.message_obj.raw_message = {"message_id": 123}
    req = ProviderRequest(
        extra_user_content_parts=[
            TextPart(text="[Video Attachment: name v.mp4, path video.mp4]"),
            TextPart(text="[Audio Attachment: path voice.amr]"),
        ]
    )

    await plugin.on_llm_request(event, req)

    assert event.bot.calls == [{"action": "voice_msg_to_text", "message_id": 123}]
    texts = [
        part.text for part in req.extra_user_content_parts if hasattr(part, "text")
    ]
    assert "任意模型都能收到这段转写" in texts
    assert any(text.startswith("[Video Attachment:") for text in texts)
    assert all(not text.startswith("[Audio Attachment:") for text in texts)
    assert all(
        not isinstance(part, VideoURLPart) for part in req.extra_user_content_parts
    )


@pytest.mark.asyncio
async def test_handle_injects_video_and_audio(sample_video: Path, sample_wav: Path):
    plugin = _plugin({})
    video = Video.fromFileSystem(path=str(sample_video))
    record = Record.fromFileSystem(path=str(sample_wav))
    event = _FakeEvent([video, record], audio_urls=[str(sample_wav)])
    req = ProviderRequest(
        prompt="what is this?",
        audio_urls=[str(sample_wav)],
        extra_user_content_parts=[
            TextPart(text="[Video Attachment: name v.mp4, path x]"),
            TextPart(text="[Audio Attachment: path y]"),
        ],
    )
    await plugin._handle(event, req)

    # ?? URL ???????? Mimo ???????
    assert req.audio_urls == []
    # 占位符被移除
    assert all(
        "[Attachment" not in getattr(p, "text", "")
        for p in req.extra_user_content_parts
    )
    # 注入的媒体块存在
    types = [p.type for p in req.extra_user_content_parts]
    assert "video_url" in types
    assert "input_audio" in types
    video_part = next(p for p in req.extra_user_content_parts if p.type == "video_url")
    audio_part = next(
        p for p in req.extra_user_content_parts if p.type == "input_audio"
    )
    assert video_part.video_url.url.startswith("data:video/mp4;base64,")
    assert audio_part.input_audio.data.startswith("data:audio/wav;base64,")
    # 默认 mark_as_temp，不入库
    assert all(
        getattr(p, "_no_save", False)
        for p in req.extra_user_content_parts
        if p.type in ("video_url", "input_audio")
    )


@pytest.mark.asyncio
async def test_handle_uses_raw_record_source_for_audio(
    sample_mp3: Path, tmp_path: Path
):
    plugin = _plugin({})
    record = Record.fromFileSystem(path=str(sample_mp3))
    event = _FakeEvent([record], audio_urls=[str(tmp_path / "invalid.wav")])
    req = ProviderRequest(audio_urls=[str(tmp_path / "invalid.wav")])

    await plugin._handle(event, req)

    audio_parts = [
        part
        for part in req.extra_user_content_parts
        if isinstance(part, InputAudioPart)
    ]
    assert len(audio_parts) == 1
    assert audio_parts[0].input_audio.data.startswith("data:audio/wav;base64,")


@pytest.mark.asyncio
async def test_handle_adds_guide_text_when_prompt_empty(sample_video: Path):
    plugin = _plugin({})
    video = Video.fromFileSystem(path=str(sample_video))
    event = _FakeEvent([video], audio_urls=[])
    req = ProviderRequest(prompt="", extra_user_content_parts=[])
    await plugin._handle(event, req)
    texts = [p.text for p in req.extra_user_content_parts if hasattr(p, "text")]
    assert any("请分析这段视频" in t for t in texts)


@pytest.mark.asyncio
async def test_handle_noop_without_media():
    plugin = _plugin({})
    event = _FakeEvent([TextPart(text="hello")], audio_urls=[])
    req = ProviderRequest(prompt="hi")
    await plugin._handle(event, req)
    assert req.extra_user_content_parts == []
