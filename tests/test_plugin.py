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
from astrbot.core.message.components import Image, Record, Reply, Video
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

    def get_model(self):
        return self.model


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


@pytest.mark.asyncio
async def test_routing_disabled_preserves_original_logic():
    context = _FakeContext()
    plugin = _plugin(
        {
            "multimodal_routing_enabled": False,
            "multimodal_provider_id": "mimo",
        },
        context=context,
    )
    event = _FakeEvent([Image(file="image.jpg")])

    await plugin.prepare_multimodal_routing(event)

    assert event.get_extra("selected_provider") is None
    assert plugin._routing_remaining == {}


@pytest.mark.asyncio
async def test_default_routing_only_selects_mimo_for_trigger_turn():
    context = _FakeContext()
    plugin = _plugin(
        {
            "multimodal_routing_enabled": True,
            "multimodal_provider_id": "mimo",
        },
        context=context,
    )
    media_event = _FakeEvent([Image(file="image.jpg")])
    follow_up = _FakeEvent([])

    await plugin.prepare_multimodal_routing(media_event)
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
            "multimodal_routing_enabled": True,
            "multimodal_provider_id": "mimo",
            "multimodal_route_turns": 3,
        },
        context=context,
    )
    events = [
        _FakeEvent([Video(file="video.mp4")]),
        _FakeEvent([]),
        _FakeEvent([]),
        _FakeEvent([]),
    ]

    for event in events:
        await plugin.prepare_multimodal_routing(event)

    assert [event.get_extra("selected_provider") for event in events] == [
        "mimo",
        "mimo",
        "mimo",
        None,
    ]
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
