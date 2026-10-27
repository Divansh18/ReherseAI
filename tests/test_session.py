import asyncio
import json

import pytest

from rehearse_ai.interview import QUESTIONS, Stage, UtteranceKind
from rehearse_ai.session import InterviewSession
from rehearse_ai.session_log import SessionLog


@pytest.fixture
def harness(tmp_path):
    spoken = []
    published = []
    decisions = []

    async def infer(kind, payload):
        decisions.append((kind, payload))
        if kind == "feedback":
            return {
                "lines": [
                    "Your parser story showed clear ownership.",
                    "Add a concrete result next time.",
                ]
            }
        return {"action": "advance", "text": ""}

    async def speak(utterance):
        spoken.append(utterance)

    session = InterviewSession(SessionLog(tmp_path, "test-session"), infer, speak, published.append)
    return session, spoken, published, decisions


def deliver(session):
    utterance = session.controller.pending
    context = f"context-{utterance.id}"
    session.tts_started(utterance.id, context)
    session.tts_first_audio(context)
    session.bot_started()
    session.tts_finished(context)
    session.bot_stopped()


def begin_output(session):
    utterance = session.controller.pending
    context = f"context-{utterance.id}"
    session.tts_started(utterance.id, context)
    session.tts_first_audio(context)
    session.bot_started()


async def start_questions(session):
    await session.start()
    deliver(session)
    session.controller.submit_greeting("I'm doing well")
    await session._queue_speech(session.controller.prepare_readiness("Good to hear."))
    deliver(session)
    await session._queue_speech(session.controller.submit_readiness(True))


async def settle():
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_full_mocked_session_and_log(harness):
    session, spoken, published, decisions = harness
    await start_questions(session)
    for number, question in enumerate(QUESTIONS, start=1):
        assert spoken[-1].text == question
        deliver(session)
        assert session.controller.current_question_number == number
        await session.user_started()
        await session.user_stopped(f"Answer {number}: I built a parser")
        await settle()
    assert spoken[-1].text.startswith("Your parser story")
    assert [kind for kind, _ in decisions] == ["decision"] * 4 + ["feedback"]
    assert session.controller.stage is Stage.DELIVERING
    deliver(session)
    assert session.controller.stage is Stage.COMPLETE
    assert any(message["event"]["event"] == "interview_complete" for message in published)
    await session.close()
    lines = [json.loads(line) for line in session.log.path.read_text().splitlines()]
    assert [line["sequence"] for line in lines] == list(range(1, len(lines) + 1))
    assert all(line["session_id"] == "test-session" for line in lines)
    assert [line["text"] for line in lines if line["event"] == "candidate_final"] == [
        f"Answer {n}: I built a parser" for n in range(1, 5)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("preferred_name", [None, "Divansh"])
async def test_normal_completion_signs_off_after_q4_follow_up(tmp_path, preferred_name):
    spoken = []
    published = []
    calls = []

    async def infer(kind, payload):
        calls.append(kind)
        if kind == "feedback":
            assert payload["answers_by_question"][3] == [
                "I would simplify the parser API.",
                "I would document the error cases first.",
            ]
            return {
                "lines": [
                    "Your parser story showed clear ownership.",
                    "Next time, describe a concrete result from its use.",
                ]
            }
        if payload["current_question_number"] == 4 and len(payload["answers_by_question"][3]) == 1:
            return {"action": "follow_up", "text": "What would you simplify first?"}
        return {"action": "advance", "text": ""}

    async def speak(utterance):
        spoken.append(utterance)

    session = InterviewSession(
        SessionLog(tmp_path, "normal-completion"), infer, speak, published.append
    )
    if preferred_name:
        session.controller.remember_name(preferred_name)
    await start_questions(session)
    for answer in (
        "I built a parser for reports.",
        "Handling malformed input was hardest.",
        "I tested it with sample files.",
        "I would simplify the parser API.",
    ):
        deliver(session)
        await session.user_started()
        await session.user_stopped(answer)
        await settle()
    assert spoken[-1].kind is UtteranceKind.FOLLOW_UP
    assert spoken[-1].text == "What would you simplify first?"
    deliver(session)
    await session.user_started()
    await session.user_stopped("I would document the error cases first.")
    await settle()

    feedback = spoken[-1]
    assert feedback.kind is UtteranceKind.FEEDBACK
    lines = feedback.text.splitlines()
    assert len(lines) == 3
    assert lines[:2] == [
        "Your parser story showed clear ownership.",
        "Next time, describe a concrete result from its use.",
    ]
    assert lines[2].startswith("That wraps up our practice interview")
    if preferred_name:
        assert preferred_name in lines[2]
    else:
        assert "Divansh" not in lines[2]
    assert "?" not in feedback.text
    assert "end the session" not in feedback.text.lower()
    assert session.controller.stage is Stage.DELIVERING
    assert not any(message["event"]["event"] == "interview_complete" for message in published)
    assert not session.controller.delivery_finished(feedback.id + 1)
    deliver(session)
    assert session.controller.stage is Stage.COMPLETE
    assert calls.count("feedback") == 1
    assert sum(message["event"]["event"] == "interview_complete" for message in published) == 1
    await session.user_started()
    await session.user_stopped("Continue.")
    assert len(spoken) == 8  # Greeting, readiness, four questions, follow-up, feedback.
    assert not any(message["event"]["event"] == "interview_ended_early" for message in published)
    await session.close()


@pytest.mark.asyncio
async def test_barge_in_cancels_obsolete_inference_and_keeps_answer(tmp_path):
    future = asyncio.Event()
    spoken = []

    async def infer(_kind, _payload):
        await future.wait()
        return {"action": "follow_up", "text": "Obsolete question?"}

    async def speak(utterance):
        spoken.append(utterance)

    session = InterviewSession(SessionLog(tmp_path, "cancel"), infer, speak, lambda _: None)
    await start_questions(session)
    deliver(session)
    await session.user_started()
    await session.user_stopped("I built a compiler")
    await settle()
    pending = session.response_task
    await session.user_started()
    assert pending.cancelling()
    await session.user_stopped("and measured its compile time")
    assert session.controller.answers[0] == ("I built a compiler and measured its compile time",)
    assert len(spoken) == 3  # Greeting, readiness, and question one only.
    await session.close()


@pytest.mark.asyncio
async def test_interrupted_question_accepts_answer_without_stale_delivery(harness):
    session, spoken, published, _decisions = harness
    await start_questions(session)
    original = spoken[-1]
    begin_output(session)
    await session.user_started()
    session.bot_stopped()
    assert session.controller.current_question_number is None
    await session.user_stopped("I built a parser")
    await settle()
    assert session.controller.answers[0] == ("I built a parser",)
    assert spoken[-1].text == QUESTIONS[1]
    assert any(m["event"]["event"] == "agent_interrupted" for m in published)
    assert not session.controller.delivery_finished(original.id)
    await session.close()


@pytest.mark.asyncio
async def test_empty_barge_in_retries_with_new_id(harness):
    session, spoken, _published, _decisions = harness
    await start_questions(session)
    first = spoken[-1]
    await session.user_started()
    await session.user_stopped("")
    assert spoken[-1].id != first.id
    assert spoken[-1].text == first.text
    assert not session.controller.delivery_finished(first.id)
    await session.close()
