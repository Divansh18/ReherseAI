import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import ErrorFrame
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

from rehearse_ai.config import ConfigurationError, Settings
from rehearse_ai.pipeline import LiveSession
from rehearse_ai.server import create_app


def test_missing_credentials_name_only(tmp_path):
    with pytest.raises(ConfigurationError, match="DEEPGRAM_API_KEY, OPENAI_API_KEY"):
        Settings.load(env={}, env_file=tmp_path / "missing.env")


def test_bad_local_config_is_actionable(tmp_path):
    env = {
        "DEEPGRAM_API_KEY": "test-dg",
        "OPENAI_API_KEY": "test-oa",
        "CARTESIA_API_KEY": "test-ca",
    }
    with pytest.raises(ConfigurationError, match="REHEARSE_HOST"):
        Settings.load({**env, "REHEARSE_HOST": "0.0.0.0"}, tmp_path / "none")
    with pytest.raises(ConfigurationError, match="REHEARSE_TTS_PROVIDER"):
        Settings.load({**env, "REHEARSE_TTS_PROVIDER": "unknown"}, tmp_path / "none")
    with pytest.raises(ConfigurationError, match="REHEARSE_CARTESIA_TTS_MODEL"):
        Settings.load({**env, "REHEARSE_CARTESIA_TTS_MODEL": "unknown"}, tmp_path / "none")
    with pytest.raises(ConfigurationError, match="REHEARSE_CARTESIA_TTS_VOICE"):
        Settings.load({**env, "REHEARSE_CARTESIA_TTS_VOICE": "unknown"}, tmp_path / "none")
    with pytest.raises(ConfigurationError, match="REHEARSE_TURN_PAUSE_SECS"):
        Settings.load({**env, "REHEARSE_TURN_PAUSE_SECS": "0"}, tmp_path / "none")
    with pytest.raises(ConfigurationError, match="REHEARSE_OPENAI_TTS_SPEED"):
        Settings.load(
            {**env, "REHEARSE_TTS_PROVIDER": "openai", "REHEARSE_OPENAI_TTS_SPEED": "4.5"},
            tmp_path / "none",
        )
    with pytest.raises(
        ConfigurationError, match="REHEARSE_OPENAI_TTS_VOICE is unavailable with tts-1"
    ):
        Settings.load(
            {
                **env,
                "REHEARSE_TTS_PROVIDER": "openai",
                "REHEARSE_OPENAI_TTS_MODEL": "tts-1",
                "REHEARSE_OPENAI_TTS_VOICE": "marin",
            },
            tmp_path / "none",
        )


def test_only_selected_tts_provider_needs_its_credential(tmp_path):
    shared = {"DEEPGRAM_API_KEY": "test-dg", "OPENAI_API_KEY": "test-oa"}
    with pytest.raises(ConfigurationError, match="CARTESIA_API_KEY"):
        Settings.load(shared, tmp_path / "none")
    settings = Settings.load({**shared, "REHEARSE_TTS_PROVIDER": "openai"}, tmp_path / "none")
    assert settings.cartesia_api_key is None
    assert settings.tts_provider == "openai"


def test_local_browser_assets_and_health_load_without_provider_calls(tmp_path):
    settings = Settings(deepgram_api_key="test-dg", openai_api_key="test-oa", session_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        health_response = client.get("/healthz")
        health = health_response.json()
        assert health["status"] == "ready"
        assert health["build_id"].startswith("0.1.0+")
        assert health_response.headers["cache-control"] == "no-store"
        page = client.get("/")
        assert page.status_code == 200
        assert "Start interview" in page.text
        assert "Four project questions, then short feedback." in page.text
        assert "voice is AI-generated" in page.text
        assert f"Build {health['build_id']}" in page.text
        assert page.headers["cache-control"] == "no-store"
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/assets/styles.css").status_code == 200
        assert client.post("/api/offer", json={"type": "bad"}).status_code == 400


@pytest.mark.asyncio
async def test_locked_pipecat_pipeline_assembles_without_provider_calls(tmp_path):
    settings = Settings(
        deepgram_api_key="test-dg",
        openai_api_key="test-oa",
        cartesia_api_key="test-ca",
        session_dir=tmp_path,
    )
    connection = SmallWebRTCConnection()
    live = LiveSession(connection, settings)
    names = [type(processor).__name__ for processor in live.pipeline.processors]
    assert names.index("DeepgramSTTService") < names.index("InterviewInput")
    assert names.index("InterviewInput") < names.index("ObservedCartesiaTTSService")
    assert names.index("ObservedCartesiaTTSService") < names.index("DeliveryObserver")
    tts = next(p for p in live.pipeline.processors if isinstance(p, CartesiaTTSService))
    assert tts._settings.model == "sonic-3.6"
    assert tts._settings.voice == "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"
    assert tts._stop_frame_timeout_s == 3.0
    assert tts._max_buffer_delay_ms == 0
    await live.stop()
    await connection.disconnect()
    assert live.log.path.exists()


@pytest.mark.asyncio
async def test_tts_one_does_not_receive_unsupported_style_instructions(tmp_path):
    settings = Settings(
        deepgram_api_key="test-dg",
        openai_api_key="test-oa",
        session_dir=tmp_path,
        tts_provider="openai",
        openai_tts_model="tts-1",
    )
    connection = SmallWebRTCConnection()
    live = LiveSession(connection, settings)
    tts = next(p for p in live.pipeline.processors if isinstance(p, OpenAITTSService))
    assert tts._settings.instructions is None
    assert tts._stop_frame_timeout_s == 20.0
    await live.stop()
    await connection.disconnect()


@pytest.mark.asyncio
async def test_pipeline_provider_error_reaches_session_log(tmp_path):
    settings = Settings(
        deepgram_api_key="test-dg",
        openai_api_key="test-oa",
        cartesia_api_key="test-ca",
        session_dir=tmp_path,
    )
    connection = SmallWebRTCConnection()
    live = LiveSession(connection, settings)
    tts = next(p for p in live.pipeline.processors if isinstance(p, CartesiaTTSService))
    await live.worker._call_event_handler(
        "on_pipeline_error", ErrorFrame(error="synthetic TTS failure", processor=tts)
    )
    await asyncio.sleep(0)
    events = [json.loads(line) for line in live.log.path.read_text().splitlines()]
    assert events[-1]["event"] == "response_error"
    assert events[-1]["failure_stage"] == "pipeline"
    assert events[-1]["processor"] == "CartesiaTTSService"
    await live.stop()
    await connection.disconnect()
