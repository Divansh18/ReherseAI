"""Provider-specific TTS setup and provider-side audio observations."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from pipecat.frames.frames import (
    DataFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.tts_service import TTSService

from rehearse_ai.config import Settings
from rehearse_ai.prompts import load_prompt
from rehearse_ai.session import InterviewSession

# A real OpenAI request first yielded audio after 9.7 seconds. This override applies
# only to the OpenAI service; Cartesia uses Pipecat's own 3-second idle boundary.
OPENAI_CONTEXT_IDLE_SECS = 20.0


@dataclass
class UtteranceMarkerFrame(DataFrame):
    utterance_id: int


class _UtteranceCorrelation:
    _session: InterviewSession
    _utterance_id: int | None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, UtteranceMarkerFrame):
                self._utterance_id = frame.utterance_id
            elif isinstance(frame, InterruptionFrame):
                self._utterance_id = None
        await super().process_frame(frame, direction)

    async def _start_audio_context(self, context_id: str) -> None:
        # Pipecat calls this before emitting TTSStartedFrame or run_tts().
        # Its public on_tts_request handlers run in separate tasks and can race.
        if self._utterance_id is not None:
            self._session.tts_context_assigned(self._utterance_id, context_id)
        await super()._start_audio_context(context_id)


class ObservedOpenAITTSService(_UtteranceCorrelation, OpenAITTSService):
    """Record synchronous provider frames before Pipecat's playback queue."""

    def __init__(self, session: InterviewSession, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._session = session
        self._utterance_id = None

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        self._session.provider_tts_requested(context_id, len(text))
        try:
            async for frame in super().run_tts(text, context_id):
                if isinstance(frame, TTSAudioRawFrame):
                    self._session.provider_tts_audio_chunk(context_id, len(frame.audio))
                elif isinstance(frame, ErrorFrame):
                    self._session.provider_tts_error(context_id, frame.error, None)
                yield frame
        except asyncio.CancelledError:
            self._session.provider_tts_cancelled(context_id)
            raise
        except Exception as exc:
            self._session.provider_tts_error(
                context_id, type(exc).__name__, getattr(exc, "status_code", None)
            )
            raise
        finally:
            self._session.provider_tts_finished(context_id)


class ObservedCartesiaTTSService(_UtteranceCorrelation, CartesiaTTSService):
    """Observe WebSocket audio when Pipecat queues each provider message."""

    def __init__(self, session: InterviewSession, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._session = session
        self._utterance_id = None

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        # Cartesia's run_tts only sends the request. Audio arrives on a separate
        # WebSocket receive task, so completing this generator is not completion.
        self._session.provider_tts_requested(context_id, len(text))
        try:
            async for frame in super().run_tts(text, context_id):
                if isinstance(frame, ErrorFrame):
                    self._session.provider_tts_error(context_id, frame.error, None)
                yield frame
        except asyncio.CancelledError:
            self._session.provider_tts_cancelled(context_id)
            raise

    async def append_to_audio_context(self, context_id: str | None, frame: Any) -> None:
        available = bool(context_id and self.audio_context_available(context_id))
        if available and isinstance(frame, TTSAudioRawFrame):
            self._session.provider_tts_audio_chunk(context_id, len(frame.audio))
        elif available and isinstance(frame, TTSStoppedFrame):
            # The WebSocket 'done' event ends provider synthesis, before queued
            # audio is sent to the transport and before delivery acknowledgment.
            self._session.provider_tts_finished(context_id)
        await super().append_to_audio_context(context_id, frame)

    async def on_audio_context_interrupted(self, context_id: str) -> None:
        self._session.provider_tts_cancelled(context_id)
        self._session.provider_tts_finished(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def on_audio_context_completed(self, context_id: str) -> None:
        # Also closes stats for a timed-out or provider-error context with no done.
        self._session.provider_tts_finished(context_id)
        await super().on_audio_context_completed(context_id)


def build_tts_service(settings: Settings, session: InterviewSession) -> TTSService:
    if settings.tts_provider == "cartesia":
        if not settings.cartesia_api_key:
            raise ValueError("CARTESIA_API_KEY is required for Cartesia TTS")
        return ObservedCartesiaTTSService(
            session,
            api_key=settings.cartesia_api_key,
            settings=CartesiaTTSService.Settings(
                model=settings.cartesia_tts_model,
                voice=settings.cartesia_tts_voice,
            ),
        )
    if settings.tts_provider == "openai":
        return ObservedOpenAITTSService(
            session,
            api_key=settings.openai_api_key,
            stop_frame_timeout_s=OPENAI_CONTEXT_IDLE_SECS,
            settings=OpenAITTSService.Settings(
                model=settings.openai_tts_model,
                voice=settings.openai_tts_voice,
                speed=settings.openai_tts_speed,
                instructions=(
                    load_prompt("tts_style")
                    if settings.openai_tts_model == "gpt-4o-mini-tts"
                    else None
                ),
            ),
        )
    raise ValueError(f"Unsupported TTS provider: {settings.tts_provider}")
