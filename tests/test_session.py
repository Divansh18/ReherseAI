import asyncio
import json

import pytest

from rehearse_ai.interview import QUESTIONS, Stage
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
