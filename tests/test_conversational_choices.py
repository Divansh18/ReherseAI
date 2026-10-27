"""Conversation choices through the same Pipecat input entry point as live STT."""

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


def harness(tmp_path, infer):
    spoken = []
    calls = []

    async def tracked_infer(kind, payload):
        calls.append((kind, payload))
        return await infer(kind, payload)

    async def speak(utterance):
        spoken.append(utterance)

    log = SessionLog(tmp_path, "choices")
    session = InterviewSession(log, tracked_infer, speak, lambda _: None)
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
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def deliver(session):
    item = session.controller.pending
    context = f"context-{item.id}"
    session.tts_started(item.id, context)
    session.tts_first_audio(context)
    session.bot_started()
    session.tts_finished(context)
    session.bot_stopped()


async def question_one(session, adapter):
    await session.start()
    deliver(session)
    await turn(adapter, "I'm good, let's start.")
    deliver(session)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert session.controller.pending.text == QUESTIONS[0]
    deliver(session)


@pytest.mark.asyncio
async def test_feedback_request_waits_for_choice_without_replaying_follow_up(tmp_path):
    async def infer(kind, payload):
        if kind == "opening_ack":
            return {"text": "Thanks."}
        if payload["phase"] == "continuation_choice":
            return {"action": "continue", "text": ""}
        if "feedback" in payload["latest_candidate_turn"].lower():
            assert payload["answers_by_question_before_latest_turn"][0] == [
                "I built a study planner in Python."
            ]
            return {
                "action": "feedback_request",
                "text": (
                    "Your planner has a clear purpose. I would explain its results more concretely."
                ),
            }
        return {
            "action": "follow_up",
            "text": "What part did you build yourself?",
            "project_status": "technical_example",
        }

    session, adapter, spoken, calls, log = harness(tmp_path, infer)
    await question_one(session, adapter)
    await turn(adapter, "I built a study planner in Python.")
    assert spoken[-1].kind is UtteranceKind.FOLLOW_UP
    deliver(session)
    await turn(adapter, "Can I get the feedback of the interview?")
    assert spoken[-1].kind is UtteranceKind.PROVISIONAL_FEEDBACK
    assert "call me" not in spoken[-1].text.lower()
    assert session.controller.answers[0] == ("I built a study planner in Python.",)
    before_delivery = len(spoken)
    deliver(session)
    await asyncio.sleep(0)
    assert session.controller.stage is Stage.AWAITING_CONTINUE_CHOICE
    assert len(spoken) == before_delivery
    assert any(
        json.loads(line)["event"] == "provisional_feedback_prepared"
        for line in log.path.read_text().splitlines()
    )
    await turn(adapter, "Continue with the project.")
    assert spoken[-1].kind is UtteranceKind.DETOUR_RESUME
    assert spoken[-1].text == "What part did you build yourself?"
    assert [kind for kind, _ in calls].count("decision") == 3
    await session.close()


@pytest.mark.asyncio
async def test_off_topic_barge_in_preserves_unheard_q3_and_waits_for_choice(tmp_path):
    async def infer(kind, payload):
        if kind == "opening_ack":
            return {"text": "Thanks."}
        if payload["phase"] == "continuation_choice":
            return {
                "action": "off_topic"
                if "love life" in payload["latest_candidate_turn"]
                else "continue",
                "text": "",
            }
        if payload["phase"] == "interrupted_turn":
            assert payload["interrupted_output_started"] is True
            assert payload["current_question_number"] == 3
            return {"action": "off_topic", "text": ""}
        return {"action": "advance", "text": "", "project_status": "technical_example"}

    session, adapter, spoken, calls, log = harness(tmp_path, infer)
    await question_one(session, adapter)
    await turn(adapter, "I built a scheduling API.")
    assert spoken[-1].text == QUESTIONS[1]
    deliver(session)
    await turn(adapter, "The hardest part was handling duplicate requests.")
    old = session.controller.pending
    assert old.text == QUESTIONS[2]
    context = f"context-{old.id}"
    session.tts_started(old.id, context)
    session.tts_first_audio(context)
    session.bot_started()
    await turn(adapter, "Can we talk about real life?")
    assert spoken[-1].kind is UtteranceKind.TOPIC_BOUNDARY
    assert spoken[-1].question_number == 3
    assert session.controller.current_question_number == 2
    assert session.controller.answers[2] == ()
    assert not session.controller.delivery_finished(old.id)
    before_stale = len(spoken)
    await session._queue_speech(old)
    session.bot_stopped()  # Cancelled output-stop cannot acknowledge the new boundary.
    assert len(spoken) == before_stale
    deliver(session)
    assert session.controller.stage is Stage.AWAITING_CONTINUE_CHOICE
    assert len(spoken) == before_stale
    await turn(adapter, "I wanna discuss about love life.")
    assert spoken[-1].kind is UtteranceKind.TOPIC_BOUNDARY
    deliver(session)
    assert session.controller.stage is Stage.AWAITING_CONTINUE_CHOICE
    await turn(adapter, "Continue with my project.")
    assert spoken[-1].kind is UtteranceKind.DETOUR_RESUME
    assert spoken[-1].text == QUESTIONS[2]
    assert all(item.text != QUESTIONS[3] for item in spoken)
    assert len([kind for kind, _ in calls if kind == "decision"]) == 5
    deliver(session)
    assert session.controller.current_question_number == 3
    await turn(adapter, "I checked it against expected outputs.")
    assert spoken[-1].text == QUESTIONS[3]
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert any(
        row["event"] == "decision_proposed" and row.get("action") == "off_topic" for row in rows
    )
    await session.close()


@pytest.mark.asyncio
async def test_q4_off_topic_request_does_not_append_question_replay(tmp_path):
    async def infer(kind, payload):
        if kind == "opening_ack":
            return {"text": "Thanks."}
        if payload["phase"] == "continuation_choice":
            return {"action": "off_topic", "text": ""}
        if "real life" in payload["latest_candidate_turn"]:
            return {"action": "off_topic", "text": ""}
        return {"action": "advance", "text": "", "project_status": "technical_example"}

    session, adapter, spoken, _calls, _log = harness(tmp_path, infer)
    await question_one(session, adapter)
    for answer in (
        "I built a scheduling API.",
        "The hardest problem was duplicate requests.",
        "I tested it with sample bookings.",
    ):
        await turn(adapter, answer)
        deliver(session)
    assert session.controller.current_question_number == 4
    await turn(adapter, "We talk about real life?")
    assert spoken[-1].kind is UtteranceKind.TOPIC_BOUNDARY
    count = len(spoken)
    deliver(session)
    await asyncio.sleep(0)
    assert len(spoken) == count
    assert session.controller.stage is Stage.AWAITING_CONTINUE_CHOICE
    await turn(adapter, "I wanna discuss about love life.")
    assert spoken[-1].kind is UtteranceKind.TOPIC_BOUNDARY
    count = len(spoken)
    deliver(session)
    await asyncio.sleep(0)
    assert len(spoken) == count
    assert session.controller.answers[3] == ()
    await session.close()


@pytest.mark.asyncio
async def test_detour_after_collected_answer_continues_decision_without_question_replay(tmp_path):
    first_decision_started = asyncio.Event()
    pending = asyncio.Event()

    async def infer(kind, payload):
        if kind == "opening_ack":
            return {"text": "Thanks."}
        if payload["phase"] == "continuation_choice":
            return {"action": "continue", "text": ""}
        if "real life" in payload["latest_candidate_turn"].lower():
            return {"action": "off_topic", "text": ""}
        if not first_decision_started.is_set():
            first_decision_started.set()
            await pending.wait()
        assert payload["current_question_answer"] == "I built a planner."
        assert payload["latest_candidate_turn"] == "I built a planner."
        return {
            "action": "follow_up",
            "text": "What did you implement yourself?",
            "project_status": "technical_example",
        }

    session, adapter, spoken, _calls, _log = harness(tmp_path, infer)
    await question_one(session, adapter)
    await turn(adapter, "I built a planner.")
    await first_decision_started.wait()
    await turn(adapter, "Can we talk about real life?")
    assert spoken[-1].kind is UtteranceKind.TOPIC_BOUNDARY
    assert session.controller.answers[0] == ("I built a planner.",)
    deliver(session)
    count = len(spoken)
    await turn(adapter, "Continue with the interview.")
    assert len(spoken) == count + 1
    assert spoken[-1].kind is UtteranceKind.FOLLOW_UP
    assert spoken[-1].text == "What did you implement yourself?"
    assert session.controller.answers[0] == ("I built a planner.",)
    pending.set()
    await session.close()


@pytest.mark.asyncio
async def test_spoken_stop_during_continuation_choice_prevents_late_replay(tmp_path):
    async def infer(kind, payload):
        if kind == "opening_ack":
            return {"text": "Thanks."}
        return {"action": "off_topic", "text": ""}

    session, adapter, spoken, _calls, _log = harness(tmp_path, infer)
    await question_one(session, adapter)
    await turn(adapter, "Can we talk about real life?")
    choice = session.controller.pending
    deliver(session)
    await turn(adapter, "Stop the interview.")
    assert session.controller.pending.kind is UtteranceKind.EARLY_FEEDBACK
    assert all(item.kind is not UtteranceKind.DETOUR_RESUME for item in spoken)
    assert not session.controller.delivery_finished(choice.id)
    await session.close()


@pytest.mark.asyncio
async def test_pending_choice_presence_then_repeated_detours_close_cleanly(tmp_path):
    async def infer(kind, payload):
        if kind == "opening_ack":
            return {"text": "Thanks."}
        if payload["phase"] == "continuation_choice":
            return {"action": "wait", "text": ""}
        return {"action": "off_topic", "text": ""}

    session, adapter, spoken, _calls, log = harness(tmp_path, infer)
    await question_one(session, adapter)
    await turn(adapter, "Can we talk about real life?")
    deliver(session)
    await turn(adapter, "You are so sweet.")
    assert spoken[-1].kind is UtteranceKind.TOPIC_BOUNDARY
    deliver(session)
    await turn(adapter, "Hello? Where you going between?")
    assert spoken[-1].text.startswith("I'm here.")
    deliver(session)
    await turn(adapter, "Tell me about your weekend instead.")
    assert spoken[-1].kind is UtteranceKind.EARLY_FEEDBACK
    assert session.controller.answers == ((), (), (), ())
    assert not any(
        item.kind is UtteranceKind.QUESTION and item.question_number == 2 for item in spoken
    )
    assert any(
        json.loads(line)["event"] == "continuation_choice_exhausted"
        for line in log.path.read_text().splitlines()
    )
    await session.close()


@pytest.mark.asyncio
async def test_no_project_choice_recovers_from_latest_real_sequence(tmp_path):
    async def infer(kind, payload):
        if kind == "opening_ack":
            return {"text": "Thanks."}
        if kind == "early_feedback":
            return {"lines": ["We'll stop here. Thanks for sharing what you could."]}
        if payload["phase"] == "interview":
            if payload["latest_candidate_turn"] == "I'm proud of":
                return {"action": "wait", "project_status": "unclear", "text": ""}
            return {"action": "wait", "project_status": "no_example", "tone": "gentle", "text": ""}
        return {"action": "wait", "project_status": "nontechnical", "text": ""}

    session, adapter, spoken, _calls, log = harness(tmp_path, infer)
    await question_one(session, adapter)
    await turn(adapter, "I'm proud of")
    await turn(adapter, "my love life.")
    assert spoken[-1].kind is UtteranceKind.NO_PROJECT_INVITE
    assert "I'm sorry" not in spoken[-1].text
    deliver(session)
    await turn(adapter, "I wanna talk about love life.")
    assert spoken[-1].kind is UtteranceKind.NO_PROJECT_SCOPE
    deliver(session)
    await turn(adapter, "You are so sweet.")
    assert spoken[-1].kind is UtteranceKind.NO_PROJECT_SCOPE
    deliver(session)
    await turn(adapter, "Hello? Where you going between?")
    assert spoken[-1].text.startswith("I'm here.")
    deliver(session)
    await turn(adapter, "Can we talk about something else?")
    assert spoken[-1].kind is UtteranceKind.EARLY_FEEDBACK
    assert session.controller.answers[1:] == ((), (), ())
    assert all(item.text != QUESTIONS[1] for item in spoken)
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert any(row["event"] == "no_project_choice_exhausted" for row in rows)
    assert not any(row["event"] == "response_error" for row in rows)
    await session.close()


@pytest.mark.asyncio
async def test_project_self_reference_is_a_possible_technical_example(tmp_path):
    async def infer(kind, payload):
        if kind == "opening_ack":
            return {"text": "Thanks."}
        assert "RehearseAI" in payload["latest_candidate_turn"]
        return {
            "action": "follow_up",
            "text": "Which parts of its implementation did you own?",
            "project_status": "technical_example",
        }

    session, adapter, spoken, _calls, _log = harness(tmp_path, infer)
    await question_one(session, adapter)
    await turn(adapter, "I'm proud of this app you're interviewing me on—RehearseAI.")
    assert spoken[-1].kind is UtteranceKind.FOLLOW_UP
    assert session.controller.project_established is True
    assert session.controller.answers[0] == (
        "I'm proud of this app you're interviewing me on—RehearseAI.",
    )
    await session.close()


@pytest.mark.asyncio
async def test_identity_misclassification_is_retried_once_for_feedback_request(tmp_path):
    decisions = 0

    async def infer(kind, payload):
        nonlocal decisions
        if kind == "opening_ack":
            return {"text": "Thanks."}
        decisions += 1
        if decisions == 1:
            return {
                "action": "detour",
                "text": "You can call me RehearseAI.",
                "project_fragment": "",
            }
        assert payload["invalid_prior_decision"]
        return {"action": "feedback_request", "text": "", "project_fragment": ""}

    session, adapter, spoken, _calls, log = harness(tmp_path, infer)
    await question_one(session, adapter)
    await turn(adapter, "Can I get the feedback of the interview?")
    assert decisions == 2
    assert spoken[-1].kind is UtteranceKind.PROVISIONAL_FEEDBACK
    assert "don't have enough project detail" in spoken[-1].text
    assert session.controller.answers[0] == ()
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert any(row.get("reason") == "coach_identity_without_question" for row in rows)
    await session.close()


@pytest.mark.asyncio
async def test_actual_coach_identity_question_waits_for_continuation_choice(tmp_path):
    async def infer(kind, payload):
        if kind == "opening_ack":
            return {"text": "Thanks."}
        return {
            "action": "detour",
            "text": "You can call me RehearseAI.",
            "project_fragment": "",
        }

    session, adapter, spoken, _calls, log = harness(tmp_path, infer)
    await question_one(session, adapter)
    await turn(adapter, "What is your name?")
    assert spoken[-1].kind is UtteranceKind.DETOUR_ACK
    assert spoken[-1].text.startswith("You can call me RehearseAI.")
    deliver(session)
    assert session.controller.stage is Stage.AWAITING_CONTINUE_CHOICE
    assert session.controller.answers == ((), (), (), ())
    assert any(
        json.loads(line)["event"] == "decision_normalized"
        for line in log.path.read_text().splitlines()
    )
    await session.close()
