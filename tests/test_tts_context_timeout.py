"""Reproduce Pipecat's context idle boundary without a paid TTS request."""

import asyncio

import pytest
from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai.tts import OpenAITTSService


class DelayedTTS(OpenAITTSService):
    async def run_tts(self, text, context_id):
        assert text.strip()
        await asyncio.sleep(0.12)
        yield TTSAudioRawFrame(b"\x00" * 4800, self.sample_rate, 1, context_id=context_id)


class AudioSink(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.audio_bytes = 0
        self.stopped = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            self.audio_bytes += len(frame.audio)
        elif isinstance(frame, TTSStoppedFrame):
            self.stopped += 1
        await self.push_frame(frame, direction)


async def run_delayed_context(timeout):
    sink = AudioSink()
    errors = []
    tts = DelayedTTS(api_key="unused", stop_frame_timeout_s=timeout)
    worker = PipelineWorker(Pipeline([tts, sink]))
    runner = WorkerRunner(handle_sigint=False)

    @worker.event_handler("on_pipeline_started")
    async def started(_worker, _frame):
        await worker.queue_frame(TTSSpeakFrame(text="Hello, this is a coach response."))

    @worker.event_handler("on_pipeline_error")
    async def failed(_worker, frame):
        errors.append(frame.error)

    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    try:
        await asyncio.sleep(0.38)
    finally:
        await runner.cancel()
        await task
    return sink, errors


@pytest.mark.asyncio
async def test_context_timeout_can_report_no_audio_before_delayed_audio_arrives():
    sink, errors = await run_delayed_context(0.05)
    assert any("completed with no audio" in error for error in errors)
    assert sink.stopped >= 1


@pytest.mark.asyncio
async def test_context_wait_longer_than_first_audio_avoids_false_silence():
    sink, errors = await run_delayed_context(0.30)
    assert sink.audio_bytes == 4800
    assert sink.stopped == 1
    assert not errors
