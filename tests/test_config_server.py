import pytest
from fastapi.testclient import TestClient
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

from rehearse_ai.config import ConfigurationError, Settings
from rehearse_ai.pipeline import LiveSession
from rehearse_ai.server import create_app


def test_missing_credentials_name_only(tmp_path):
    with pytest.raises(ConfigurationError, match="DEEPGRAM_API_KEY, OPENAI_API_KEY"):
        Settings.load(env={}, env_file=tmp_path / "missing.env")


def test_bad_local_config_is_actionable(tmp_path):
    env = {"DEEPGRAM_API_KEY": "test-dg", "OPENAI_API_KEY": "test-oa"}
    with pytest.raises(ConfigurationError, match="REHEARSE_HOST"):
        Settings.load({**env, "REHEARSE_HOST": "0.0.0.0"}, tmp_path / "none")
    with pytest.raises(ConfigurationError, match="REHEARSE_TTS_MODEL"):
        Settings.load({**env, "REHEARSE_TTS_MODEL": "unknown"}, tmp_path / "none")
    with pytest.raises(ConfigurationError, match="REHEARSE_TURN_PAUSE_SECS"):
        Settings.load({**env, "REHEARSE_TURN_PAUSE_SECS": "0"}, tmp_path / "none")


def test_local_browser_assets_and_health_load_without_provider_calls(tmp_path):
    settings = Settings(deepgram_api_key="test-dg", openai_api_key="test-oa", session_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        assert client.get("/healthz").json() == {"status": "ready"}
        page = client.get("/")
        assert page.status_code == 200
        assert "Start interview" in page.text
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/assets/styles.css").status_code == 200
        assert client.post("/api/offer", json={"type": "bad"}).status_code == 400


@pytest.mark.asyncio
async def test_locked_pipecat_pipeline_assembles_without_provider_calls(tmp_path):
    settings = Settings(deepgram_api_key="test-dg", openai_api_key="test-oa", session_dir=tmp_path)
    connection = SmallWebRTCConnection()
    live = LiveSession(connection, settings)
    names = [type(processor).__name__ for processor in live.pipeline.processors]
    assert names.index("DeepgramSTTService") < names.index("InterviewInput")
    assert names.index("InterviewInput") < names.index("OpenAITTSService")
    assert names.index("OpenAITTSService") < names.index("DeliveryObserver")
    await live.stop()
    await connection.disconnect()
    assert live.log.path.exists()
