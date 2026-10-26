"""Pipecat assembly; interview policy remains in the domain controller."""

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    DataFrame,
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from rehearse_ai.config import Settings
from rehearse_ai.interview import Utterance
from rehearse_ai.prompts import load_prompt
from rehearse_ai.session import InterviewSession
from rehearse_ai.session_log import SessionLog

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["follow_up", "nudge", "clarification", "wait", "advance"],
        },
        "text": {"type": "string"},
    },
    "required": ["action", "text"],
    "additionalProperties": False,
}
FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {"type": "string"}}},
    "required": ["lines"],
    "additionalProperties": False,
}
OPENING_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


@dataclass
class UtteranceMarkerFrame(DataFrame):
    utterance_id: int


class InterviewInput(FrameProcessor):
    """Collect final STT segments within Pipecat's single user-turn boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.session: InterviewSession | None = None
        self._segments: list[str] = []
        self._turn_open = False

    async def speak(self, utterance: Utterance) -> None:
        if self.session is None or self.session.controller.pending != utterance:
            return
        await self.push_frame(UtteranceMarkerFrame(utterance_id=utterance.id))
        if self.session.controller.pending != utterance:
            return
        await self.push_frame(TTSSpeakFrame(text=utterance.text, append_to_context=False))

    async def cancel_audio(self) -> None:
        await self.broadcast_interruption()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM and self.session is not None:
            if isinstance(frame, UserStartedSpeakingFrame):
                if not self._turn_open:
                    self._turn_open = True
                    await self.session.user_started()
                    if self._segments:
                        await self.session.detect_stop_from_segments(" ".join(self._segments))
            elif isinstance(frame, VADUserStartedSpeakingFrame):
                self.session.turn_signal("vad_started", frame_id=frame.id)
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                self.session.turn_signal(
                    "vad_stopped",
                    frame_id=frame.id,
                    stop_secs=frame.stop_secs,
                    speech_end_basis="Silero VAD stop; estimate, not last audible phoneme",
                )
            elif isinstance(frame, TranscriptionFrame):
                if frame.text.strip():
                    self._segments.append(frame.text.strip())
                    self.session.transcript_segment(
                        frame.text.strip(),
                        frame_timestamp=frame.timestamp,
                        finalized=frame.finalized,
                        frame_id=frame.id,
                    )
                    if self._turn_open:
                        await self.session.detect_stop_from_segments(" ".join(self._segments))
            elif isinstance(frame, UserStoppedSpeakingFrame):
                if self._turn_open or self._segments:
                    await self.session.user_stopped(
                        " ".join(self._segments),
                        segment_count=len(self._segments),
                        boundary_frame_id=frame.id,
                    )
                    self._turn_open = False
                    self._segments.clear()
        await self.push_frame(frame, direction)


class DeliveryObserver(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.session: InterviewSession | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM and self.session is not None:
            if isinstance(frame, BotStartedSpeakingFrame):
                self.session.bot_started()
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self.session.bot_stopped()
        await self.push_frame(frame, direction)


class AudioObserver(FrameProcessor):
    """Correlate TTS context IDs with the utterance marker before output."""

    def __init__(self) -> None:
        super().__init__()
        self.session: InterviewSession | None = None
        self._utterance_id: int | None = None
        self._first_audio: set[str] = set()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM and self.session is not None:
            if isinstance(frame, UtteranceMarkerFrame):
                self._utterance_id = frame.utterance_id
                return
            if isinstance(frame, InterruptionFrame):
                self._utterance_id = None
            elif isinstance(frame, TTSStartedFrame) and self._utterance_id is not None:
                self.session.tts_started(self._utterance_id, frame.context_id)
            elif isinstance(frame, TTSAudioRawFrame) and frame.context_id not in self._first_audio:
                if frame.context_id:
                    self._first_audio.add(frame.context_id)
                self.session.tts_first_audio(frame.context_id)
            elif isinstance(frame, TTSStoppedFrame):
                self.session.tts_finished(frame.context_id)
        await self.push_frame(frame, direction)


def inference_adapter(llm: OpenAILLMService):
    async def infer(kind: str, payload: dict[str, Any]) -> Mapping[str, Any]:
        schema = {
            "decision": DECISION_SCHEMA,
            "opening_ack": OPENING_SCHEMA,
            "feedback": FEEDBACK_SCHEMA,
            "early_feedback": FEEDBACK_SCHEMA,
        }[kind]
        response = await llm.run_inference(
            LLMContext(messages=[{"role": "user", "content": json.dumps(payload)}]),
            max_tokens=220,
            system_instruction=load_prompt(kind),
            response_schema=schema,
        )
        if response is None:
            raise ValueError("Empty LLM response")
        parsed = json.loads(response)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response was not an object")
        return parsed

    return infer


class LiveSession:
    def __init__(self, connection: Any, settings: Settings, *, build_id: str | None = None) -> None:
        self.connection = connection
        self.log = SessionLog(settings.session_dir, str(uuid.uuid4()))
        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        )
        llm = OpenAILLMService(
            api_key=settings.openai_api_key,
            settings=OpenAILLMService.Settings(model=settings.llm_model),
        )
        input_adapter = InterviewInput()
        audio_observer = AudioObserver()
        delivery = DeliveryObserver()
        self.session = InterviewSession(
            self.log,
            inference_adapter(llm),
            input_adapter.speak,
            connection.send_app_message,
            turn_pause_secs=settings.turn_pause_secs,
            cancel_audio=input_adapter.cancel_audio,
            build_id=build_id,
        )
        input_adapter.session = self.session
        audio_observer.session = self.session
        delivery.session = self.session
        self.pipeline = Pipeline(
            [
                transport.input(),
                VADProcessor(
                    vad_analyzer=SileroVADAnalyzer(
                        params=VADParams(
                            confidence=settings.vad_confidence,
                            start_secs=settings.vad_start_secs,
                            stop_secs=settings.vad_stop_secs,
                        )
                    )
                ),
                DeepgramSTTService(
                    api_key=settings.deepgram_api_key,
                    settings=DeepgramSTTService.Settings(model=settings.stt_model),
                ),
                UserTurnProcessor(
                    user_turn_stop_timeout=30.0,
                    user_turn_strategies=UserTurnStrategies(
                        start=[VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()],
                        stop=[
                            SpeechTimeoutUserTurnStopStrategy(
                                user_speech_timeout=settings.turn_pause_secs
                            )
                        ],
                    ),
                ),
                input_adapter,
                OpenAITTSService(
                    api_key=settings.openai_api_key,
                    settings=OpenAITTSService.Settings(
                        model=settings.tts_model,
                        voice=settings.tts_voice,
                        speed=settings.tts_speed,
                        instructions=(
                            load_prompt("tts_style")
                            if settings.tts_model == "gpt-4o-mini-tts"
                            else None
                        ),
                    ),
                ),
                audio_observer,
                transport.output(),
                delivery,
            ]
        )
        self.worker = PipelineWorker(
            self.pipeline, params=PipelineParams(audio_in_sample_rate=16000)
        )
        self.runner = WorkerRunner(handle_sigint=False)

        @self.worker.event_handler("on_pipeline_error")
        async def pipeline_error(_worker: PipelineWorker, frame: Any) -> None:
            category = frame.category.value if frame.category is not None else None
            processor = type(frame.processor).__name__ if frame.processor is not None else None
            self.session.pipeline_error(processor=processor, category=category)

        @transport.event_handler("on_client_connected")
        async def connected(_transport: SmallWebRTCTransport, _client: Any) -> None:
            await self.session.start()

        @transport.event_handler("on_client_disconnected")
        async def disconnected(_transport: SmallWebRTCTransport, _client: Any) -> None:
            await self.stop()

    async def run(self) -> None:
        try:
            await self.runner.add_workers(self.worker)
            await self.runner.run()
        except Exception:
            self.session.pipeline_error(processor="PipelineWorker", category="unexpected_failure")
            raise
        finally:
            await self.session.close()

    async def stop(self) -> None:
        await self.session.close()
        await self.runner.cancel()
