import asyncio
import json

import pytest
from pipecat.frames.frames import (
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from rehearse_ai.interview import GREETING, QUESTIONS, Stage, UtteranceKind
from rehearse_ai.pipeline import InterviewInput
from rehearse_ai.session import InterviewSession
from rehearse_ai.session_log import SessionLog


def make_session(tmp_path, infer=None):
    spoken = []
    calls = []

    async def default_infer(kind, payload):
        calls.append((kind, payload))
        if kind == "opening_ack":
            response = payload["greeting_response"].lower()
            return {"text": "Doing well, thanks!" if "how" in response else "Thanks for sharing."}
        if payload["phase"] == "continuation_choice":
            return {"action": "continue", "text": "", "project_fragment": ""}
        return {"action": "wait", "text": "", "project_fragment": ""}

    async def speak(utterance):
        spoken.append(utterance)

    log = SessionLog(tmp_path, "conversation")
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


def deliver(session):
    item = session.controller.pending
    context = f"context-{item.id}"
    session.tts_started(item.id, context)
    session.tts_first_audio(context)
    session.bot_started()
    session.tts_finished(context)
    session.bot_stopped()


async def settle():
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_full_name_opening_routes_to_first_question(tmp_path):
    session, adapter, spoken, calls, _log = make_session(tmp_path)
    await session.start()
    assert spoken[-1].text == GREETING
    deliver(session)
    await turn(adapter, "Good! How about you?")
    await settle()
    assert spoken[-1].text == "Doing well, thanks! What should I call you?"
    deliver(session)
    await turn(adapter, "Divansh.")
    assert session.controller.preferred_name == "Divansh"
    assert spoken[-1].text == "Nice to meet you, Divansh. Ready to jump in?"
    deliver(session)
    await turn(adapter, "Yeah, let's get into it.")
    assert spoken[-1].text == QUESTIONS[0]
    assert session.controller.answers == ((), (), (), ())
    assert session.controller.remaining_follow_ups == 1
    assert [kind for kind, _ in calls] == ["opening_ack"]
    await session.close()


@pytest.mark.parametrize(
    ("greeting", "expected_name", "expected_kind"),
    [
        ("I'm doing great. How are you?", None, UtteranceKind.NAME_PROMPT),
        ("A bit nervous.", None, UtteranceKind.NAME_PROMPT),
        ("I'm exhausted today.", None, UtteranceKind.NAME_PROMPT),
        ("I'm Divansh, doing well.", "Divansh", UtteranceKind.READINESS),
        ("I'm good, let's start.", None, UtteranceKind.OPENING_ACK),
        ("I'd rather not give my name.", None, UtteranceKind.READINESS),
    ],
)
@pytest.mark.asyncio
async def test_opening_adapts_to_greeting(tmp_path, greeting, expected_name, expected_kind):
    session, adapter, spoken, _calls, _log = make_session(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, greeting)
    await settle()
    assert session.controller.preferred_name == expected_name
    assert spoken[-1].kind is expected_kind
    if expected_kind is UtteranceKind.OPENING_ACK:
        deliver(session)
        await settle()
        assert spoken[-1].text == QUESTIONS[0]
    await session.close()


@pytest.mark.asyncio
async def test_name_refusal_and_deferred_readiness_then_start(tmp_path):
    session, adapter, spoken, _calls, _log = make_session(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "A bit nervous.")
    await settle()
    deliver(session)
    await turn(adapter, "I'd rather not give my name.")
    assert session.controller.name_declined
    assert spoken[-1].kind is UtteranceKind.READINESS
    deliver(session)
    await turn(adapter, "Not ready yet.")
    assert spoken[-1].kind is UtteranceKind.OPENING_WAIT
    deliver(session)
    await turn(adapter, "Start.")
    assert spoken[-1].text == QUESTIONS[0]
    await session.close()


async def at_question_one(tmp_path):
    session, adapter, spoken, calls, log = make_session(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm good, let's start.")
    await settle()
    deliver(session)
    await settle()
    assert spoken[-1].text == QUESTIONS[0]
    deliver(session)
    return session, adapter, spoken, calls, log


@pytest.mark.asyncio
async def test_name_detour_after_question_one_preserves_limits(tmp_path):
    session, adapter, spoken, calls, _log = await at_question_one(tmp_path)
    await turn(adapter, "Okay. So you'll not ask me my name?")
    assert spoken[-1].kind is UtteranceKind.NAME_PROMPT
    assert "What should I call you?" in spoken[-1].text
    assert session.controller.answers[0] == ()
    deliver(session)
    await turn(adapter, "Divansh.")
    assert spoken[-1].kind is UtteranceKind.DETOUR_ACK
    deliver(session)
    await settle()
    assert session.controller.stage is Stage.AWAITING_CONTINUE_CHOICE
    await turn(adapter, "Continue.")
    assert spoken[-1].text == QUESTIONS[0]
    assert session.controller.remaining_follow_ups == 1
    assert session.controller.remaining_clarifications == 1
    assert [kind for kind, _ in calls] == ["opening_ack", "decision"]
    await session.close()


@pytest.mark.asyncio
async def test_detour_after_partial_answer_keeps_answer_and_decides_once(tmp_path):
    session, adapter, spoken, calls, _log = await at_question_one(tmp_path)
    await turn(adapter, "I built a parser for risk reports.")
    await settle()
    assert session.controller.stage is Stage.AWAITING_PROPOSAL
    await turn(adapter, "Aren't you going to ask my name?")
    assert session.controller.answers[0] == ("I built a parser for risk reports.",)
    assert spoken[-1].kind is UtteranceKind.NAME_PROMPT
    deliver(session)
    await turn(adapter, "Divansh.")
    deliver(session)
    await settle()
    assert session.controller.stage is Stage.AWAITING_CONTINUE_CHOICE
    await turn(adapter, "Continue.")
    assert session.controller.answers[0] == ("I built a parser for risk reports.",)
    assert session.controller.remaining_follow_ups == 1
    assert session.controller.remaining_clarifications == 1
    assert [kind for kind, _ in calls] == ["opening_ack", "decision", "decision"]
    await session.close()


@pytest.mark.asyncio
async def test_name_detour_replays_unanswered_follow_up_without_spending_budget(tmp_path):
    calls = []

    async def infer(kind, payload):
        calls.append((kind, payload))
        if kind == "opening_ack":
            return {"text": "Sounds good."}
        if payload["phase"] == "continuation_choice":
            return {"action": "continue", "text": "", "project_fragment": ""}
        return {"action": "follow_up", "text": "What did you build?", "project_fragment": ""}

    session, adapter, spoken, _unused, _log = make_session(tmp_path, infer)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm good, let's start.")
    await settle()
    deliver(session)
    await settle()
    deliver(session)
    await turn(adapter, "I built a parser.")
    await settle()
    assert spoken[-1].kind is UtteranceKind.FOLLOW_UP
    deliver(session)
    await turn(adapter, "So you'll not ask me my name?")
    assert spoken[-1].kind is UtteranceKind.NAME_PROMPT
    deliver(session)
    await turn(adapter, "Divansh.")
    deliver(session)
    await settle()
    assert session.controller.stage is Stage.AWAITING_CONTINUE_CHOICE
    await turn(adapter, "Continue.")
    assert spoken[-1].text == "What did you build?"
    assert spoken[-1].kind is UtteranceKind.DETOUR_RESUME
    assert session.controller.answers[0] == ("I built a parser.",)
    assert session.controller.remaining_follow_ups == 0
    assert len([kind for kind, _ in calls if kind == "decision"]) == 2
    await session.close()


@pytest.mark.asyncio
async def test_structured_general_detour_restores_project_answer(tmp_path):
    calls = []

    async def infer(kind, payload):
        calls.append((kind, payload))
        if kind == "opening_ack":
            return {"text": "Sounds good."}
        if payload["latest_candidate_turn"].startswith("The parser"):
            return {
                "action": "detour",
                "text": "I'm an AI interview coach.",
                "project_fragment": "The parser checks PRs.",
            }
        return {"action": "wait", "text": "", "project_fragment": ""}

    session, adapter, spoken, _unused, _log = make_session(tmp_path, infer)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm good, let's start.")
    await settle()
    deliver(session)
    await settle()
    deliver(session)
    await turn(adapter, "The parser checks PRs. Are you a person?")
    await settle()
    assert spoken[-1].kind is UtteranceKind.DETOUR_ACK
    assert session.controller.answers[0] == ("The parser checks PRs.",)
    deliver(session)
    await settle()
    assert session.controller.stage is Stage.AWAITING_CONTINUE_CHOICE
    assert spoken[-1].kind is UtteranceKind.DETOUR_ACK
    assert session.controller.answers[0] == ("The parser checks PRs.",)
    assert session.controller.remaining_follow_ups == 1
    await session.close()


@pytest.mark.parametrize("phase", ["greeting", "name", "detour"])
@pytest.mark.asyncio
async def test_spoken_stop_latches_during_new_phases(tmp_path, phase):
    session, adapter, spoken, _calls, log = make_session(tmp_path)
    await session.start()
    if phase != "greeting":
        deliver(session)
        await turn(adapter, "Doing well.")
        await settle()
        if phase == "detour":
            deliver(session)
            await turn(adapter, "Start.")
            deliver(session)
            await settle()
            deliver(session)
            await turn(adapter, "Why didn't you ask my name?")
    stale = session.controller.pending.id
    await turn(adapter, "I wanna end the interview.")
    assert session.controller.stage is Stage.DELIVERING
    assert session.controller.pending.kind is UtteranceKind.EARLY_FEEDBACK
    assert not session.controller.delivery_finished(stale)
    count = len(spoken)
    await turn(adapter, "Start.")
    assert len(spoken) == count + 1
    assert spoken[-1].kind is UtteranceKind.EARLY_FEEDBACK
    assert all(item.kind is not UtteranceKind.QUESTION for item in spoken[count:])
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert sum(row["event"] == "early_end_requested" for row in rows) == 1
    await session.close()


@pytest.mark.asyncio
async def test_interrupted_name_prompt_accepts_explicit_name_without_stale_delivery(tmp_path):
    session, adapter, spoken, _calls, _log = make_session(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "Doing well.")
    await settle()
    first = session.controller.pending
    await turn(adapter, "Divansh.")
    assert spoken[-1].kind is UtteranceKind.READINESS
    assert spoken[-1].id != first.id
    assert not session.controller.delivery_finished(first.id)
    assert session.controller.preferred_name == "Divansh"
    await session.close()


@pytest.mark.asyncio
async def test_interrupted_readiness_after_name_can_start_without_stale_delivery(tmp_path):
    session, adapter, spoken, _calls, _log = make_session(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "Doing well.")
    await settle()
    deliver(session)
    await turn(adapter, "Divansh.")
    readiness = session.controller.pending
    context = f"context-{readiness.id}"
    session.tts_started(readiness.id, context)
    session.tts_first_audio(context)
    session.bot_started()
    await turn(adapter, "Start.")
    assert spoken[-1].text == QUESTIONS[0]
    assert not session.controller.delivery_finished(readiness.id)
    assert session.controller.preferred_name == "Divansh"
    await session.close()
