import asyncio
import json

import pytest
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection

from rehearse_ai.interview import QUESTIONS, Stage
from rehearse_ai.pipeline import AudioObserver, UtteranceMarkerFrame
from rehearse_ai.prompts import load_prompt
from rehearse_ai.session import InterviewSession
from rehearse_ai.session_log import SessionLog


async def settle():
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def begin_output(session):
    utterance = session.controller.pending
    context = f"context-{utterance.id}"
    session.tts_started(utterance.id, context)
    session.tts_first_audio(context)
    session.bot_started()
    return context


def finish_output(session, context):
    session.tts_finished(context)
    session.bot_stopped()


def events(log):
    return [json.loads(line) for line in log.path.read_text().splitlines()]


@pytest.mark.asyncio
async def test_audio_observer_correlates_tts_frames_with_utterance_id(tmp_path):
    async def infer(_kind, _payload):
        raise AssertionError("No inference expected")

    async def speak(_utterance):
        pass

    log = SessionLog(tmp_path, "audio-ids")
    session = InterviewSession(log, infer, speak, lambda _: None)
    observer = AudioObserver()
    observer.session = session

    async def forward(_frame, _direction=FrameDirection.DOWNSTREAM):
        pass

    observer.push_frame = forward
    await session.start()
    utterance_id = session.controller.pending.id
    for frame in (
        UtteranceMarkerFrame(utterance_id=utterance_id),
        TTSStartedFrame(context_id="tts-context"),
        TTSAudioRawFrame(b"\0" * 32, 24000, 1, context_id="tts-context"),
        TTSStoppedFrame(context_id="tts-context"),
    ):
        await observer.process_frame(frame, FrameDirection.DOWNSTREAM)
    names = [row["event"] for row in events(log)]
    assert "tts_started" in names
    assert names.count("tts_first_audio_generated") == 1
    assert any(
        row["event"] == "tts_generation_finished" and row["utterance_id"] == utterance_id
        for row in events(log)
    )
    assert session.controller.stage is Stage.DELIVERING
    await session.close()


@pytest.mark.asyncio
async def test_spoken_stop_during_answer_ends_without_unasked_feedback(tmp_path):
    calls = []
    spoken = []

    async def infer(kind, payload):
        calls.append((kind, payload))
        raise AssertionError("No answer exists for feedback")

    async def speak(utterance):
        spoken.append(utterance)

    log = SessionLog(tmp_path, "stop-answer")
    session = InterviewSession(log, infer, speak, lambda _: None)
    await session.start()
    finish_output(session, begin_output(session))
    await session.user_started()
    await session.user_stopped("Can we end the interview?")
    assert session.controller.stage is Stage.DELIVERING
    assert spoken[-1].text == "We'll end here. Thanks for practicing."
    assert session.controller.answers == ((), (), (), ())
    assert not calls
    finish_output(session, begin_output(session))
    assert session.controller.stage is Stage.ENDED_EARLY
    await session.user_started()
    await session.user_stopped("More answers")
    assert len(spoken) == 2
    await session.close()
    assert events(log)[-1]["session_end_kind"] == "spoken_early_end"


@pytest.mark.parametrize(
    "spoken_request",
    [
        "Can we end the interview?",
        "Sounds a good question. But I would like to end the interview.",
        "Could we stop now?",
    ],
)
@pytest.mark.asyncio
async def test_explicit_stop_phrasings_are_not_inferred_as_answers(tmp_path, spoken_request):
    async def infer(_kind, _payload):
        raise AssertionError("No LLM call expected without collected answers")

    async def speak(_utterance):
        pass

    session = InterviewSession(SessionLog(tmp_path, "stop-phrasing"), infer, speak, lambda _: None)
    await session.start()
    finish_output(session, begin_output(session))
    await session.user_started()
    await session.user_stopped(spoken_request)
    assert session.controller.answers[0] == ()
    assert session.controller.pending.kind.name == "EARLY_FEEDBACK"
    await session.close()


@pytest.mark.asyncio
async def test_spoken_stop_cancels_inference_and_uses_only_collected_answer(tmp_path):
    held = asyncio.Event()
    calls = []
    spoken = []
    cancellations = []

    async def infer(kind, payload):
        calls.append((kind, payload))
        if kind == "decision":
            try:
                await held.wait()
            except asyncio.CancelledError:
                cancellations.append(kind)
                raise
        return {"lines": ["We'll stop here; you described building a parser."]}

    async def speak(utterance):
        spoken.append(utterance)

    async def cancel_audio():
        cancellations.append("audio")

    log = SessionLog(tmp_path, "stop-inference")
    session = InterviewSession(log, infer, speak, lambda _: None, cancel_audio=cancel_audio)
    await session.start()
    finish_output(session, begin_output(session))
    await session.user_started()
    await session.user_stopped("I built a parser")
    await settle()
    assert calls[0][0] == "decision"
    await session.user_started()
    await session.user_stopped("I would like to end the interview.")
    await settle()
    assert "decision" in cancellations
    assert "audio" in cancellations
    assert calls[-1][0] == "early_feedback"
    assert calls[-1][1]["answers_by_question"] == [["I built a parser"], [], [], []]
    assert spoken[-1].text.startswith("We'll stop here")
    finish_output(session, begin_output(session))
    assert session.controller.stage is Stage.ENDED_EARLY
    await session.close()


@pytest.mark.asyncio
async def test_stop_during_prepared_agent_output_discards_unplayed_text(tmp_path):
    spoken = []

    async def infer(kind, _payload):
        if kind == "decision":
            return {"action": "follow_up", "text": "Which parser did you build?"}
        return {"lines": ["We'll end early; you described building a parser."]}

    async def speak(utterance):
        spoken.append(utterance)

    log = SessionLog(tmp_path, "stop-output")
    session = InterviewSession(log, infer, speak, lambda _: None)
    await session.start()
    finish_output(session, begin_output(session))
    await session.user_started()
    await session.user_stopped("I built a parser")
    await settle()
    cancelled = spoken[-1]
    assert cancelled.text == "Which parser did you build?"
    await session.user_started()
    session.bot_stopped()  # Pipecat can emit this after cancellation.
    await session.user_stopped("Can we end the interview?")
    await settle()
    rows = events(log)
    assert not any(
        row["event"] == "agent_delivery_reported" and row["utterance_id"] == cancelled.id
        for row in rows
    )
    assert not any(
        row["event"] == "agent_output_started" and row["utterance_id"] == cancelled.id
        for row in rows
    )
    assert session.controller.stage is Stage.DELIVERING
    assert spoken[-1].text.startswith("We'll end early")
    await session.close()


@pytest.mark.asyncio
async def test_unheard_next_question_routes_continuation_to_previous_answer(tmp_path):
    payloads = []
    spoken = []

    async def infer(kind, payload):
        payloads.append((kind, payload))
        return {"action": "advance", "text": ""}

    async def speak(utterance):
        spoken.append(utterance)

    session = InterviewSession(SessionLog(tmp_path, "unheard"), infer, speak, lambda _: None)
    await session.start()
    finish_output(session, begin_output(session))
    await session.user_started()
    await session.user_stopped("I built an interview API")
    await settle()
    unheard = spoken[-1]
    assert unheard.text == QUESTIONS[1]
    await session.user_started()  # Before any TTS chunk for question two.
    session.bot_stopped()
    await session.user_stopped("It schedules interviews")
    await settle()
    assert session.controller.answers[0] == ("I built an interview API It schedules interviews",)
    assert session.controller.answers[1] == ()
    assert session.controller.current_question_number == 1
    assert spoken[-1].text == QUESTIONS[1]
    assert spoken[-1].id != unheard.id
    assert not session.controller.delivery_finished(unheard.id)
    assert payloads[-1][1]["current_question"] == QUESTIONS[0]
    assert len(payloads) == 1  # No new LLM decision from the unheard question.
    await session.close()


@pytest.mark.asyncio
async def test_barge_in_during_playback_and_late_stop_do_not_deliver(tmp_path):
    spoken = []

    async def infer(_kind, _payload):
        return {"action": "advance", "text": ""}

    async def speak(utterance):
        spoken.append(utterance)

    log = SessionLog(tmp_path, "barge-late")
    session = InterviewSession(log, infer, speak, lambda _: None)
    await session.start()
    interrupted = spoken[-1]
    context = begin_output(session)
    await session.user_started()
    finish_output(session, context)  # Old TTS and BotStopped arrive after interruption.
    await session.user_stopped("I built an API")
    await settle()
    assert session.controller.answers[0] == ("I built an API",)
    assert spoken[-1].text == QUESTIONS[1]
    assert not session.controller.delivery_finished(interrupted.id)
    assert not any(
        row["event"] == "agent_delivery_reported" and row["utterance_id"] == interrupted.id
        for row in events(log)
    )
    await session.close()


@pytest.mark.asyncio
async def test_question_two_payload_and_prompt_keep_follow_up_on_technical_challenge(tmp_path):
    payloads = []
    spoken = []

    async def infer(kind, payload):
        payloads.append((kind, payload))
        if payload["current_question_number"] == 1:
            return {"action": "advance", "text": ""}
        return {"action": "follow_up", "text": "What made the cache invalidation difficult?"}

    async def speak(utterance):
        spoken.append(utterance)

    session = InterviewSession(SessionLog(tmp_path, "q2"), infer, speak, lambda _: None)
    await session.start()
    finish_output(session, begin_output(session))
    await session.user_started()
    await session.user_stopped("I built a scheduling API")
    await settle()
    finish_output(session, begin_output(session))
    await session.user_started()
    await session.user_stopped("Cache invalidation was the hardest part")
    await settle()
    assert spoken[-1].text == "What made the cache invalidation difficult?"
    assert payloads[-1][1]["current_question"] == QUESTIONS[1]
    assert payloads[-1][1]["current_question_answers"] == [
        "Cache invalidation was the hardest part"
    ]
    assert payloads[-1][1]["last_delivered_prompt"] == QUESTIONS[1]
    assert "For question two" in load_prompt("decision")
    await session.close()
