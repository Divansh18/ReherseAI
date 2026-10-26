"""Exercise conversational controls through the Pipecat input adapter."""

import asyncio
import json

import pytest
from pipecat.frames.frames import (
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from rehearse_ai.interview import QUESTIONS, Stage, UtteranceKind
from rehearse_ai.pipeline import InterviewInput
from rehearse_ai.session import InterviewSession
from rehearse_ai.session_log import SessionLog


def harness(tmp_path, infer=None):
    spoken = []
    calls = []

    async def default_infer(kind, payload):
        calls.append((kind, payload))
        if kind == "opening_ack":
            return {"text": "Thanks for sharing."}
        if kind == "feedback":
            return {"lines": ["Thanks for practicing."]}
        return {"action": "wait", "text": "", "project_status": "unclear"}

    log = SessionLog(tmp_path, "routing")

    async def speak(utterance):
        spoken.append(utterance)

    session = InterviewSession(log, infer or default_infer, speak, lambda _: None)
    adapter = InterviewInput()
    adapter.session = session
    adapter.push_frame = lambda *_args, **_kwargs: asyncio.sleep(0)
    return session, adapter, spoken, calls, log


async def turn(adapter, *segments):
    await adapter.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    for index, text in enumerate(segments):
        await adapter.process_frame(
            TranscriptionFrame(
                text=text, user_id="candidate", timestamp=str(index), finalized=True
            ),
            FrameDirection.DOWNSTREAM,
        )
    await adapter.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)


async def settle():
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def deliver(session):
    utterance = session.controller.pending
    context = f"context-{utterance.id}"
    session.tts_started(utterance.id, context)
    session.tts_first_audio(context)
    session.bot_started()
    session.tts_finished(context)
    session.bot_stopped()


async def question_one(session, adapter):
    await session.start()
    deliver(session)
    await turn(adapter, "I'm good, let's start.")
    await settle()
    deliver(session)
    await settle()
    assert session.controller.pending.text == QUESTIONS[0]
    deliver(session)


@pytest.mark.asyncio
async def test_readiness_name_question_returns_to_readiness(tmp_path):
    session, adapter, spoken, calls, _ = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "Doing okay.")
    await settle()
    deliver(session)
    await turn(adapter, "Divanj.")
    deliver(session)
    await turn(adapter, "Can you tell me your name?")
    assert spoken[-1].kind is UtteranceKind.OPENING_COACH_REPLY
    assert "RehearseAI" in spoken[-1].text
    assert session.controller.preferred_name == "Divanj"
    deliver(session)
    await settle()
    assert spoken[-1].kind is UtteranceKind.READINESS
    assert all(kind != "decision" for kind, _ in calls)
    await session.close()


@pytest.mark.asyncio
async def test_name_and_coach_question_combined_during_interruption(tmp_path):
    session, adapter, spoken, _, _ = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "Doing okay.")
    await settle()
    old_prompt = session.controller.pending
    await turn(adapter, "You can call me Divanj.", "And what should I call you?")
    assert session.controller.preferred_name == "Divanj"
    assert "RehearseAI" in spoken[-1].text
    assert not session.controller.delivery_finished(old_prompt.id)
    assert session.controller.answers == ((), (), (), ())
    deliver(session)
    await settle()
    assert spoken[-1].kind is UtteranceKind.READINESS
    await session.close()


@pytest.mark.asyncio
async def test_exact_logged_name_combination_is_kept(tmp_path):
    session, adapter, spoken, _, _ = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "Doing okay.")
    await settle()
    deliver(session)
    await turn(adapter, "Name is Divanj. And what should I call you?")
    assert session.controller.preferred_name == "Divanj"
    assert "RehearseAI" in spoken[-1].text
    assert session.controller.answers == ((), (), (), ())
    await session.close()


@pytest.mark.asyncio
async def test_interrupted_coach_detour_retries_without_consuming_opening(tmp_path):
    session, adapter, spoken, _, _ = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "Doing okay.")
    await settle()
    deliver(session)
    await turn(adapter, "Can you tell me your name?")
    first_reply = session.controller.pending
    await turn(adapter, "Sorry. Can you repeat the question?")
    assert spoken[-1].kind is UtteranceKind.OPENING_COACH_REPLY
    assert spoken[-1].id != first_reply.id
    assert not session.controller.delivery_finished(first_reply.id)
    deliver(session)
    await settle()
    assert spoken[-1].kind is UtteranceKind.NAME_PROMPT
    assert session.controller.answers == ((), (), (), ())
    await session.close()


@pytest.mark.asyncio
async def test_repeat_request_repeats_active_follow_up_without_spending_follow_up(tmp_path):
    async def infer(kind, _payload):
        if kind == "opening_ack":
            return {"text": "Thanks for sharing."}
        return {
            "action": "follow_up",
            "text": "Which part did you build?",
            "project_status": "technical_example",
        }

    session, adapter, spoken, _, _ = harness(tmp_path, infer)
    await question_one(session, adapter)
    await turn(adapter, "I built a parser.")
    await settle()
    assert spoken[-1].kind is UtteranceKind.FOLLOW_UP
    deliver(session)
    await turn(adapter, "Sorry. Can you repeat the question?")
    assert spoken[-1].kind is UtteranceKind.REPEAT
    assert spoken[-1].text == "Which part did you build?"
    assert session.controller.remaining_follow_ups == 0
    assert session.controller.answers[0] == ("I built a parser.",)
    await session.close()


@pytest.mark.asyncio
async def test_no_project_then_football_never_advances_and_stop_latches(tmp_path):
    async def infer(kind, payload):
        if kind == "opening_ack":
            return {"text": "Thanks for sharing."}
        if kind == "feedback":
            return {"lines": ["We ended early, so there's no project answer to review."]}
        turn_text = payload["latest_candidate_turn"].lower()
        return {
            "action": "advance" if "project" in turn_text else "wait",
            "text": "",
            "project_status": "no_example" if "project" in turn_text else "nontechnical",
        }

    session, adapter, spoken, _, log = harness(tmp_path, infer)
    await question_one(session, adapter)
    await turn(adapter, "I didn't have any project.")
    await settle()
    assert spoken[-1].kind is UtteranceKind.NO_PROJECT_INVITE
    assert session.controller.current_question_number == 1
    deliver(session)
    await turn(adapter, "I played football.")
    await settle()
    assert spoken[-1].kind is UtteranceKind.NO_PROJECT_SCOPE
    assert QUESTIONS[1] not in [utterance.text for utterance in spoken]
    deliver(session)
    await turn(adapter, "Stop the interview.")
    await settle()
    assert session.controller.stage is Stage.DELIVERING
    assert spoken[-1].kind is UtteranceKind.EARLY_FEEDBACK
    assert QUESTIONS[1] not in [utterance.text for utterance in spoken]
    assert session.controller.answers[1] == ()
    await session.close()
    assert any(
        event["event"] == "early_end_requested"
        for event in (json.loads(line) for line in log.path.read_text().splitlines())
    )


@pytest.mark.asyncio
async def test_skip_without_project_enters_example_recovery(tmp_path):
    session, adapter, spoken, _, _ = harness(tmp_path)
    await question_one(session, adapter)
    await turn(adapter, "Skip this question.")
    assert spoken[-1].kind is UtteranceKind.NO_PROJECT_INVITE
    assert session.controller.current_question_number == 1
    assert session.controller.answers == ((), (), (), ())
    await session.close()


@pytest.mark.asyncio
async def test_gentle_project_recovery_and_real_example(tmp_path):
    async def infer(kind, payload):
        if kind == "opening_ack":
            return {"text": "Thanks for sharing."}
        if "nothing in my life" in payload["latest_candidate_turn"]:
            return {"action": "advance", "project_status": "no_example", "tone": "gentle"}
        return {"action": "advance", "project_status": "technical_example", "text": ""}

    session, adapter, spoken, _, _ = harness(tmp_path, infer)
    await question_one(session, adapter)
    await turn(adapter, "I'm proud of nothing in my life.")
    await settle()
    assert spoken[-1].kind is UtteranceKind.NO_PROJECT_INVITE
    assert "sorry" in spoken[-1].text.lower()
    deliver(session)
    await turn(adapter, "I wrote a small script to organize my notes.")
    await settle()
    assert spoken[-1].text == QUESTIONS[1]
    assert session.controller.project_established is True
    await session.close()


@pytest.mark.asyncio
async def test_tts_no_audio_error_is_correlated_and_not_mistaken_for_delivery(tmp_path):
    session, adapter, spoken, _, log = harness(tmp_path)
    await session.start()
    first = session.controller.pending
    context = "a31d57c2-7219-4345-9f5b-0e0286095643"
    session.tts_started(first.id, context)
    session.pipeline_error(
        processor="OpenAITTSService",
        category="unknown",
        error=f"TTS context {context} completed with no audio",
    )
    assert session.controller.pending == first
    assert session.controller.stage is Stage.DELIVERING
    await turn(adapter, "Can you repeat that?")
    assert spoken[-1].id != first.id
    session.pipeline_error(
        processor="OpenAITTSService",
        category="unknown",
        error=f"TTS context {context} completed with no audio",
    )
    await session.close()
    events = [json.loads(line) for line in log.path.read_text().splitlines()]
    errors = [event for event in events if event["event"] == "response_error"]
    assert errors[0]["error_code"] == "tts_context_no_audio"
    assert errors[0]["utterance_id"] == first.id
    assert errors[0]["recoverable"] is True
    assert any(event["event"] == "stale_pipeline_error_ignored" for event in events)
    assert not any(
        event["event"] == "agent_delivery_reported" and event["utterance_id"] == first.id
        for event in events
    )
