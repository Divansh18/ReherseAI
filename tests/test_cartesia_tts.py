"""Exercise Pipecat's Cartesia WebSocket context path without provider calls."""

import base64
import json

import pytest
from pipecat.processors.frame_processor import FrameDirection
from websockets.protocol import State

from rehearse_ai.session import InterviewSession
from rehearse_ai.session_log import SessionLog
from rehearse_ai.tts import ObservedCartesiaTTSService, UtteranceMarkerFrame


class RecordingSession:
    def __init__(self):
        self.events = []

    def provider_tts_requested(self, context_id, text_length):
        self.events.append(("requested", context_id, text_length))

    def provider_tts_audio_chunk(self, context_id, byte_count):
        self.events.append(("audio", context_id, byte_count))

    def provider_tts_finished(self, context_id):
        self.events.append(("finished", context_id))

    def provider_tts_cancelled(self, context_id):
        self.events.append(("cancelled", context_id))

    def provider_tts_error(self, context_id, error, status):
        self.events.append(("error", context_id, status))


class FakeSocket:
    state = State.OPEN

    def __init__(self):
        self.sent = []
        self.messages = []

    async def send(self, message):
        self.sent.append(json.loads(message))

    def __aiter__(self):
        async def messages():
            while self.messages:
                yield json.dumps(self.messages.pop(0))

        return messages()


@pytest.mark.asyncio
async def test_tts_request_binds_context_before_audio_started_frame(tmp_path):
    async def infer(_kind, _payload):
        raise AssertionError("No inference expected")

    async def speak(_utterance):
        pass

    log = SessionLog(tmp_path, "tts-correlation")
    session = InterviewSession(log, infer, speak, lambda _: None)
    tts = ObservedCartesiaTTSService(session, api_key="test-key")
    await session.start()
    utterance_id = session.controller.pending.id
    await tts.process_frame(UtteranceMarkerFrame(utterance_id), FrameDirection.DOWNSTREAM)
    await tts._start_audio_context("current")
    assert session.tts_context_utterance("current") == utterance_id
    session.provider_tts_requested("current", 13)
    assert any(
        row["event"] == "provider_tts_requested"
        and row["tts_context_id"] == "current"
        and row["utterance_id"] == utterance_id
        for row in (json.loads(line) for line in log.path.read_text().splitlines())
    )
    await session.user_started()
    await tts._start_audio_context("stale")
    assert session.tts_context_utterance("stale") is None
    await session.close()


@pytest.mark.asyncio
async def test_cancelled_cartesia_context_drops_late_audio_and_next_context_receives_it():
    session = RecordingSession()
    socket = FakeSocket()
    tts = ObservedCartesiaTTSService(
        session,
        api_key="test-key",
        settings=ObservedCartesiaTTSService.Settings(
            model="sonic-3.6", voice="db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"
        ),
    )
    tts._websocket = socket
    await tts.create_audio_context("old")
    async for _ in tts.run_tts("First question.", "old"):
        pass
    await tts.on_audio_context_interrupted("old")
    tts._audio_context_manager.remove("old")

    await tts.create_audio_context("new")
    async for _ in tts.run_tts("Fresh question.", "new"):
        pass
    socket.messages = [
        {"type": "chunk", "context_id": "old", "data": base64.b64encode(b"old").decode()},
        {"type": "done", "context_id": "old"},
        {"type": "chunk", "context_id": "new", "data": base64.b64encode(b"new").decode()},
        {"type": "done", "context_id": "new"},
    ]
    await tts._process_messages()

    assert {"context_id": "old", "cancel": True} in socket.sent
    assert ("cancelled", "old") in session.events
    assert ("audio", "old", 3) not in session.events
    assert ("audio", "new", 3) in session.events
    assert ("finished", "new") in session.events
