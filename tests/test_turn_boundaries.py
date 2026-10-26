import asyncio
import json

import pytest
from pipecat.frames.frames import (
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from rehearse_ai.interview import QUESTIONS, Stage
from rehearse_ai.pipeline import InterviewInput
from rehearse_ai.session import InterviewSession
from rehearse_ai.session_log import SessionLog


def deliver(session):
    utterance = session.controller.pending
    context = f"context-{utterance.id}"
    session.tts_started(utterance.id, context)
    session.tts_first_audio(context)
    session.bot_started()
    session.tts_finished(context)
    session.bot_stopped()


async def start_questions(session):
    await session.start()
    deliver(session)
    session.controller.submit_greeting("I'm doing well")
    await session._queue_speech(session.controller.prepare_readiness("Good to hear."))
    deliver(session)
    await session._queue_speech(session.controller.submit_readiness(True))


def begin_output(session):
    utterance = session.controller.pending
    context = f"context-{utterance.id}"
    session.tts_started(utterance.id, context)
    session.tts_first_audio(context)
    session.bot_started()


async def settle():
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def final(text, number):
    return TranscriptionFrame(
        text=text, user_id="candidate", timestamp=f"segment-{number}", finalized=False
    )


@pytest.mark.asyncio
async def test_multiple_final_stt_segments_one_turn_and_one_decision(tmp_path):
    calls = []
    sent = []
    input_adapter = InterviewInput()

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        sent.append((frame, direction))

    input_adapter.push_frame = push

    async def infer(kind, payload):
        calls.append((kind, payload))
        return {"action": "advance", "text": ""}

    log = SessionLog(tmp_path, "segments")
    session = InterviewSession(log, infer, input_adapter.speak, lambda _: None)
    input_adapter.session = session
    await start_questions(session)
    deliver(session)

    # Pipecat can forward a final STT segment before emitting the turn-start frame.
    for frame in (
        final("I built an API", 1),
        UserStartedSpeakingFrame(),
        final("for scheduling interviews", 2),
        final("using Python and Postgres", 3),
        UserStoppedSpeakingFrame(),
        UserStoppedSpeakingFrame(),  # stale duplicate must not cause another inference
    ):
        await input_adapter.process_frame(frame, FrameDirection.DOWNSTREAM)
    await settle()

    assert session.controller.answers[0] == (
        "I built an API for scheduling interviews using Python and Postgres",
    )
    assert len(calls) == 1
    assert session.controller.pending.text == QUESTIONS[1]
    await session.close()
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    segments = [row for row in rows if row["event"] == "stt_final_segment"]
    assert [row["frame_timestamp"] for row in segments] == [
        "segment-1",
        "segment-2",
        "segment-3",
    ]
    assert [row["segment_count"] for row in rows if row["event"] == "candidate_final"] == [3]
    assert len([row for row in rows if row["event"] == "inference_started"]) == 1


@pytest.mark.asyncio
async def test_fragment_after_follow_up_waits_for_continuation(tmp_path):
    actions = iter(
        [
            {"action": "follow_up", "text": "What did the API do?"},
            {"action": "wait", "text": ""},
            {"action": "advance", "text": ""},
        ]
    )
    calls = []
    spoken = []

    async def infer(kind, payload):
        calls.append((kind, payload))
        return next(actions)

    async def speak(utterance):
        spoken.append(utterance)

    session = InterviewSession(SessionLog(tmp_path, "fragment"), infer, speak, lambda _: None)
    await start_questions(session)
    deliver(session)
    await session.user_started()
    await session.user_stopped("I built an interview API")
    await settle()
    assert spoken[-1].text == "What did the API do?"
    deliver(session)

    await session.user_started()
    await session.user_stopped("So the idea", segment_count=1)
    await settle()
    assert session.controller.stage is Stage.AWAITING_PROPOSAL
    assert len(spoken) == 4

    await session.user_started()
    await session.user_stopped("was to schedule interviews using Python", segment_count=1)
    await settle()
    assert session.controller.answers[0][-1] == (
        "So the idea was to schedule interviews using Python"
    )
    assert spoken[-1].text == QUESTIONS[1]
    assert len(calls) == 3
    assert calls[1][1]["remaining_follow_ups"] == 0
    await session.close()


@pytest.mark.asyncio
async def test_spent_follow_up_does_not_force_advance(tmp_path):
    actions = iter(
        [
            {"action": "follow_up", "text": "What did it do?"},
            {"action": "follow_up", "text": "An impermissible second follow-up?"},
            {"action": "advance", "text": ""},
        ]
    )
    spoken = []

    async def infer(_kind, _payload):
        return next(actions)

    async def speak(utterance):
        spoken.append(utterance)

    session = InterviewSession(SessionLog(tmp_path, "limit"), infer, speak, lambda _: None)
    await start_questions(session)
    deliver(session)
    await session.user_started()
    await session.user_stopped("I built an API")
    await settle()
    deliver(session)
    await session.user_started()
    await session.user_stopped("So the idea")
    await settle()

    assert session.controller.stage is Stage.AWAITING_PROPOSAL
    assert len(spoken) == 4
    await session.user_started()
    await session.user_stopped("was to connect candidates to interviewers")
    await settle()
    assert spoken[-1].text == QUESTIONS[1]
    await session.close()


@pytest.mark.asyncio
async def test_barge_in_joins_several_segments_before_inference(tmp_path):
    calls = []
    input_adapter = InterviewInput()
    input_adapter.push_frame = lambda *_args, **_kwargs: asyncio.sleep(0)

    async def infer(kind, payload):
        calls.append((kind, payload))
        return {"action": "advance", "text": ""}

    session = InterviewSession(
        SessionLog(tmp_path, "barge-segments"), infer, input_adapter.speak, lambda _: None
    )
    input_adapter.session = session
    await start_questions(session)
    begin_output(session)
    for frame in (
        UserStartedSpeakingFrame(),
        final("I built", 1),
        final("a scheduling API", 2),
        final("with Python", 3),
        UserStoppedSpeakingFrame(),
    ):
        await input_adapter.process_frame(frame, FrameDirection.DOWNSTREAM)
    await settle()
    assert session.controller.answers[0] == ("I built a scheduling API with Python",)
    assert session.controller.pending.text == QUESTIONS[1]
    assert len(calls) == 1
    await session.close()


@pytest.mark.asyncio
async def test_stale_llm_result_after_continuation_cannot_advance(tmp_path):
    release_old = asyncio.Event()
    calls = 0
    spoken = []

    async def infer(_kind, _payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            try:
                await release_old.wait()
            except asyncio.CancelledError:
                await release_old.wait()
            return {"action": "advance", "text": ""}
        return {"action": "wait", "text": ""}

    async def speak(utterance):
        spoken.append(utterance)

    session = InterviewSession(SessionLog(tmp_path, "stale"), infer, speak, lambda _: None)
    await start_questions(session)
    deliver(session)
    await session.user_started()
    await session.user_stopped("I built an API")
    await settle()
    await session.user_started()
    await session.user_stopped("and owned its deployment")
    await settle()
    release_old.set()
    await settle()
    assert session.controller.stage is Stage.AWAITING_PROPOSAL
    assert len(spoken) == 3
    assert session.controller.answers[0] == ("I built an API and owned its deployment",)
    await session.close()
