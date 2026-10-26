import asyncio
import json

import pytest
from pipecat.frames.frames import (
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from rehearse_ai.interview import (
    GREETING,
    QUESTIONS,
    READINESS_CLARIFICATION,
    READINESS_PROMPT,
    READINESS_WAIT,
    InterviewController,
    InvalidTransition,
    LimitReached,
    Readiness,
    Stage,
    UtteranceKind,
)
from rehearse_ai.pipeline import InterviewInput, inference_adapter
from rehearse_ai.prompts import load_prompt
from rehearse_ai.session import InterviewSession
from rehearse_ai.session_log import SessionLog


def output_started(session):
    utterance = session.controller.pending
    context = f"tts-{utterance.id}"
    session.tts_started(utterance.id, context)
    session.tts_first_audio(context)
    session.bot_started()
    return context


def deliver(session):
    context = output_started(session)
    session.tts_finished(context)
    session.bot_stopped()


async def settle():
    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def turn(adapter, *segments):
    await adapter.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    for number, text in enumerate(segments, 1):
        await adapter.process_frame(
            TranscriptionFrame(
                text=text, user_id="candidate", timestamp=f"segment-{number}", finalized=False
            ),
            FrameDirection.DOWNSTREAM,
        )
    await adapter.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)


def harness(tmp_path, infer=None):
    adapter = InterviewInput()
    adapter.push_frame = lambda *_args, **_kwargs: asyncio.sleep(0)
    spoken = []

    async def default_infer(kind, payload):
        assert kind == "opening_ack"
        assert payload["greeting_response"]
        return {"text": "Thanks for telling me."}

    async def speak(utterance):
        spoken.append(utterance)

    log = SessionLog(tmp_path, "opening")
    session = InterviewSession(log, infer or default_infer, speak, lambda _: None)
    adapter.session = session
    return session, adapter, spoken, log


def test_controller_requires_opening_delivery_and_bounds_prompts():
    controller = InterviewController()
    greeting = controller.start()
    assert greeting.text == GREETING
    assert greeting.question_number is None
    with pytest.raises(InvalidTransition):
        controller.submit_greeting("Doing well")
    assert controller.delivery_finished(greeting.id)
    assert controller.submit_greeting("A bit nervous") is None
    readiness = controller.prepare_readiness("That's understandable.")
    assert readiness.text.endswith(READINESS_PROMPT)
    with pytest.raises(InvalidTransition):
        controller.submit_readiness(True)
    assert controller.delivery_finished(readiness.id)
    wait = controller.submit_readiness(False)
    assert wait.kind is UtteranceKind.OPENING_WAIT
    assert controller.delivery_finished(wait.id)
    assert controller.submit_readiness(False) is None
    repeat = controller.request_opening_repeat()
    assert repeat.text == READINESS_PROMPT
    assert controller.delivery_finished(repeat.id)
    with pytest.raises(LimitReached):
        controller.request_opening_repeat()
    question = controller.submit_readiness(True)
    assert question.text == QUESTIONS[0]
    assert controller.current_question_number is None
    assert controller.answers == ((), (), (), ())
    assert controller.remaining_follow_ups == 1
    assert controller.remaining_clarifications == 1
    assert controller.delivery_finished(question.id)
    assert controller.current_question_number == 1


def test_ready_greeting_requires_ack_delivery_before_question_one():
    controller = InterviewController()
    greeting = controller.start()
    assert controller.delivery_finished(greeting.id)
    controller.submit_greeting("I'm good, let's start.")
    acknowledgement = controller.prepare_readiness("Sounds good.", Readiness.READY)
    assert acknowledgement.kind is UtteranceKind.OPENING_ACK
    assert acknowledgement.text == "Sounds good."
    assert controller.current_question_number is None
    assert not controller.delivery_finished(greeting.id)
    assert controller.delivery_finished(acknowledgement.id)
    assert controller.stage is Stage.AWAITING_QUESTION_START
    question = controller.begin_question_one()
    assert question.text == QUESTIONS[0]
    assert controller.current_question_number is None
    assert controller.delivery_finished(question.id)
    assert controller.current_question_number == 1


@pytest.mark.parametrize(
    ("greeting_response", "acknowledgement", "classification", "kind", "expected_text"),
    [
        (
            "I'm doing great. How are you?",
            "Doing well, thanks!",
            "unclear",
            UtteranceKind.READINESS,
            "Doing well, thanks! Ready to jump in?",
        ),
        (
            "A bit nervous.",
            "That makes sense.",
            "unclear",
            UtteranceKind.READINESS,
            "That makes sense. Ready to jump in?",
        ),
        (
            "I'm good, let's start.",
            "Sounds good.",
            "ready",
            UtteranceKind.OPENING_ACK,
            "Sounds good.",
        ),
        (
            "Not ready yet.",
            "Of course.",
            "deferred",
            UtteranceKind.OPENING_WAIT,
            f"Of course. {READINESS_WAIT}",
        ),
    ],
)
@pytest.mark.asyncio
async def test_short_contextual_opening_routes_greeting(
    tmp_path, greeting_response, acknowledgement, classification, kind, expected_text
):
    calls = []

    async def infer(purpose, payload):
        calls.append((purpose, payload))
        return {"text": acknowledgement}

    session, adapter, spoken, log = harness(tmp_path, infer)
    await session.start()
    assert spoken[-1].text == GREETING
    deliver(session)
    await turn(adapter, greeting_response)
    await settle()
    assert calls[0][0] == "opening_ack"
    assert calls[0][1]["greeting_response"] == greeting_response
    assert calls[0][1]["greeting_readiness"] == classification
    assert spoken[-1].kind is kind
    assert spoken[-1].text == expected_text
    assert "four questions" not in spoken[-1].text.lower()
    assert session.controller.current_question_number is None
    assert session.controller.answers == ((), (), (), ())
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [
        row["classification"] for row in rows if row["event"] == "greeting_readiness_classified"
    ] == [classification]
    if kind is UtteranceKind.OPENING_ACK:
        deliver(session)
        await settle()
        assert spoken[-1].text == QUESTIONS[0]
        assert session.controller.current_question_number is None
    elif kind is UtteranceKind.OPENING_WAIT:
        deliver(session)
        await turn(adapter, "Start.")
        assert spoken[-1].text == QUESTIONS[0]
    await session.close()


@pytest.mark.asyncio
async def test_ready_greeting_ack_interruption_preserves_delivery_order(tmp_path):
    session, adapter, spoken, _log = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm good, let's start.")
    await settle()
    acknowledgement = session.controller.pending
    assert acknowledgement.kind is UtteranceKind.OPENING_ACK
    await turn(adapter, "Hello?")
    assert spoken[-1].kind is UtteranceKind.OPENING_ACK
    assert spoken[-1].id != acknowledgement.id
    assert not any(item.kind is UtteranceKind.QUESTION for item in spoken)
    retried = session.controller.pending
    output_started(session)
    await turn(adapter, "Go ahead.")
    assert not session.controller.delivery_finished(retried.id)
    assert spoken[-1].text == QUESTIONS[0]
    await session.close()


@pytest.mark.asyncio
async def test_end_after_ready_ack_drops_queued_question(tmp_path):
    session, adapter, spoken, _log = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm good, let's start.")
    await settle()
    assert session.controller.pending.kind is UtteranceKind.OPENING_ACK
    deliver(session)
    await session.close()
    await settle()
    assert not any(item.kind is UtteranceKind.QUESTION for item in spoken)


@pytest.mark.asyncio
async def test_ready_greeting_remains_ready_after_later_stt_segment(tmp_path):
    session, adapter, spoken, log = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm good, let's start.", "I have a project in mind.")
    await settle()
    assert (
        session.controller.greeting_response == "I'm good, let's start. I have a project in mind."
    )
    assert spoken[-1].kind is UtteranceKind.OPENING_ACK
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [
        row["classification"] for row in rows if row["event"] == "greeting_readiness_classified"
    ] == ["ready"]
    await session.close()


@pytest.mark.asyncio
async def test_spoken_stop_during_greeting_preempts_opening_inference(tmp_path):
    calls = []

    async def infer(kind, payload):
        calls.append((kind, payload))
        return {"text": "This should not run."}

    session, adapter, spoken, log = harness(tmp_path, infer)
    await session.start()
    await turn(adapter, "Can we end the interview?")
    assert calls == []
    assert session.controller.pending.kind is UtteranceKind.EARLY_FEEDBACK
    assert not any(item.kind is UtteranceKind.QUESTION for item in spoken)
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert any(row["event"] == "early_end_requested" for row in rows)
    await session.close()


@pytest.mark.asyncio
async def test_spoken_stop_during_ready_greeting_ack_blocks_question_one(tmp_path):
    session, adapter, spoken, _log = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm good, let's start.")
    await settle()
    assert session.controller.pending.kind is UtteranceKind.OPENING_ACK
    await turn(adapter, "Stop the interview.")
    assert session.controller.pending.kind is UtteranceKind.EARLY_FEEDBACK
    assert not any(item.kind is UtteranceKind.QUESTION for item in spoken)
    await session.close()


@pytest.mark.asyncio
async def test_greeting_ack_not_yet_repeat_then_ready_through_session_route(tmp_path):
    session, adapter, spoken, log = harness(tmp_path)
    await session.start()
    assert spoken[-1].text == GREETING
    deliver(session)
    await turn(adapter, "I'm doing okay, a little nervous.")
    await settle()
    assert spoken[-1].kind is UtteranceKind.READINESS
    assert "Thanks for telling me." in spoken[-1].text
    assert spoken[-1].text.endswith(READINESS_PROMPT)
    assert session.controller.current_question_number is None
    deliver(session)

    await turn(adapter, "Not yet.")
    assert spoken[-1].kind is UtteranceKind.OPENING_WAIT
    deliver(session)
    count = len(spoken)
    await turn(adapter, "No, I'm not ready.")
    assert len(spoken) == count
    await turn(adapter, "Can you repeat that?")
    assert spoken[-1].text == READINESS_PROMPT
    deliver(session)
    await turn(adapter, "Yeah, let's do it.")
    assert spoken[-1].text == QUESTIONS[0]
    assert session.controller.current_question_number is None
    assert session.controller.answers == ((), (), (), ())
    deliver(session)
    assert session.controller.current_question_number == 1
    assert session.controller.remaining_follow_ups == 1
    assert session.controller.remaining_clarifications == 1
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [row["ready"] for row in rows if row["event"] == "readiness_response"] == [
        False,
        False,
        True,
    ]
    assert any(
        row["event"] == "opening_waiting_for_readiness" and row["classification"] == "deferred"
        for row in rows
    )
    await session.close()


@pytest.mark.parametrize(
    "response",
    [
        "Yes. Let's let's go into it.",
        "Let's get into it.",
        "Start.",
        "I'm ready.",
        "Go ahead.",
    ],
)
@pytest.mark.asyncio
async def test_natural_affirmative_readiness_routes_to_question_one(tmp_path, response):
    session, adapter, spoken, log = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm doing great. How are you?")
    await settle()
    deliver(session)
    await turn(adapter, response)
    assert spoken[-1].text == QUESTIONS[0]
    assert spoken[-1].kind is UtteranceKind.QUESTION
    assert session.controller.current_question_number is None
    assert session.controller.answers == ((), (), (), ())
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [row["classification"] for row in rows if row["event"] == "readiness_response"] == [
        "ready"
    ]
    await session.close()


@pytest.mark.parametrize(
    "response", ["I'm not ready yet.", "Don't start.", "Yes, but give me a moment."]
)
@pytest.mark.asyncio
async def test_deferred_readiness_does_not_start_question_one(tmp_path, response):
    session, adapter, spoken, log = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "Doing fine.")
    await settle()
    deliver(session)
    await turn(adapter, response)
    assert spoken[-1].kind is UtteranceKind.OPENING_WAIT
    assert session.controller.current_question_number is None
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [row["classification"] for row in rows if row["event"] == "readiness_response"] == [
        "deferred"
    ]
    await session.close()


@pytest.mark.asyncio
async def test_real_log_sequence_and_segmented_start_recovery(tmp_path):
    session, adapter, spoken, _log = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm doing great. How are you?")
    await settle()
    deliver(session)
    await turn(adapter, "Yes. Let's let's go into it.")
    assert spoken[-1].text == QUESTIONS[0]
    await session.close()

    recovered, adapter, spoken, log = harness(tmp_path / "recovery")
    await recovered.start()
    deliver(recovered)
    await turn(adapter, "I'm doing great. How are you?")
    await settle()
    deliver(recovered)
    await turn(adapter, "Not yet.")
    deliver(recovered)
    await turn(adapter, "I'm saying let's get into it.", "Start.")
    assert spoken[-1].text == QUESTIONS[0]
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [row["text"] for row in rows if row["event"] == "stt_final_segment"][-2:] == [
        "I'm saying let's get into it.",
        "Start.",
    ]
    assert [row["classification"] for row in rows if row["event"] == "readiness_response"] == [
        "deferred",
        "ready",
    ]
    await recovered.close()


@pytest.mark.asyncio
async def test_unclear_opening_and_exhausted_allowances_remain_actionable(tmp_path):
    published = []
    session, adapter, spoken, log = harness(tmp_path)
    session._publish = published.append
    await session.start()
    deliver(session)
    await turn(adapter, "I'm fine.")
    await settle()
    deliver(session)
    await turn(adapter, "Hey. Where you are?")
    assert spoken[-1].text == READINESS_CLARIFICATION
    deliver(session)
    count = len(spoken)
    await turn(adapter, "Hello. What's up?")
    assert len(spoken) == count
    assert published[-1]["event"]["event"] == "opening_waiting_for_readiness"
    await turn(adapter, "Can you repeat that?")
    assert spoken[-1].text == READINESS_PROMPT
    deliver(session)
    await turn(adapter, "Can you repeat that?")
    assert published[-1]["event"]["event"] == "opening_repeat_limit"
    assert session.controller.stage is Stage.AWAITING_READINESS
    count = len(spoken)
    await turn(adapter, "I'm not ready yet.")
    assert spoken[-1].kind is UtteranceKind.OPENING_WAIT
    deliver(session)
    await turn(adapter, "Don't start.")
    assert len(spoken) == count + 1
    assert published[-1]["event"]["event"] == "opening_waiting_for_readiness"
    await turn(adapter, "Start.")
    assert spoken[-1].text == QUESTIONS[0]
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [row["classification"] for row in rows if row["event"] == "readiness_response"] == [
        "unclear",
        "unclear",
        "deferred",
        "deferred",
        "ready",
    ]
    await session.close()


@pytest.mark.asyncio
async def test_greeting_repeat_is_bounded_through_session_route(tmp_path):
    session, adapter, spoken, log = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "Can you repeat that?")
    assert spoken[-1].text == GREETING
    deliver(session)
    count = len(spoken)
    await turn(adapter, "Can you repeat that?")
    assert len(spoken) == count
    assert session.controller.stage is Stage.AWAITING_GREETING
    await turn(adapter, "I'm doing fine.")
    await settle()
    assert spoken[-1].kind is UtteranceKind.READINESS
    assert any(
        json.loads(line)["event"] == "opening_repeat_limit"
        for line in log.path.read_text().splitlines()
    )
    await session.close()


@pytest.mark.asyncio
async def test_repeat_during_opening_ack_cancels_inference_and_reasks_greeting(tmp_path):
    started = asyncio.Event()
    cancelled = []

    async def infer(kind, _payload):
        assert kind == "opening_ack"
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    session, adapter, spoken, _log = harness(tmp_path, infer)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm doing fine.")
    await started.wait()
    await turn(adapter, "Can you repeat that?")
    await settle()
    assert cancelled
    assert spoken[-1].text == GREETING
    assert session.controller.greeting_response == ""
    assert session.controller.answers == ((), (), (), ())
    await session.close()


@pytest.mark.asyncio
async def test_exhausted_greeting_repeat_resumes_cancelled_ack_generation(tmp_path):
    started = asyncio.Event()
    cancelled = []
    calls = 0

    async def infer(kind, _payload):
        nonlocal calls
        assert kind == "opening_ack"
        calls += 1
        if calls == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
        return {"text": "Good to hear."}

    session, adapter, spoken, log = harness(tmp_path, infer)
    await session.start()
    deliver(session)
    await turn(adapter, "Can you repeat that?")
    deliver(session)
    await turn(adapter, "Doing well.")
    await started.wait()
    await turn(adapter, "Can you repeat that?")
    await settle()
    assert cancelled
    assert calls == 2
    assert spoken[-1].kind is UtteranceKind.READINESS
    assert any(
        json.loads(line)["event"] == "opening_repeat_limit"
        for line in log.path.read_text().splitlines()
    )
    await session.close()


@pytest.mark.asyncio
async def test_speech_queue_and_pipeline_failures_are_published(tmp_path):
    published = []

    async def broken_speak(_utterance):
        raise RuntimeError("synthetic transport failure")

    session = InterviewSession(
        SessionLog(tmp_path, "errors"),
        lambda *_args: asyncio.sleep(0),
        broken_speak,
        published.append,
    )
    await session.start()
    session.pipeline_error(processor="OpenAITTSService", category="provider_error")
    errors = [
        message["event"] for message in published if message["event"]["event"] == "response_error"
    ]
    assert [error["failure_stage"] for error in errors] == ["speech_queue", "pipeline"]
    assert errors[0]["utterance_id"] == session.controller.pending.id
    assert errors[1]["processor"] == "OpenAITTSService"
    await session.close()


@pytest.mark.asyncio
async def test_opening_inference_failure_reports_error_and_uses_safe_ack(tmp_path):
    async def infer(_kind, _payload):
        raise RuntimeError("synthetic provider failure")

    session, adapter, spoken, log = harness(tmp_path, infer)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm doing well.")
    await settle()
    assert spoken[-1].kind is UtteranceKind.READINESS
    assert spoken[-1].text.startswith("Thanks for telling me.")
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert any(
        row["event"] == "response_error" and row["failure_stage"] == "opening_inference"
        for row in rows
    )
    await session.close()


@pytest.mark.asyncio
async def test_how_are_you_fallback_answers_briefly_if_inference_fails(tmp_path):
    async def infer(_kind, _payload):
        raise RuntimeError("synthetic provider failure")

    session, adapter, spoken, _log = harness(tmp_path, infer)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm doing great. How are you?")
    await settle()
    assert spoken[-1].text == "Doing well, thanks. Ready to jump in?"
    await session.close()


@pytest.mark.asyncio
async def test_unheard_greeting_retries_and_heard_barge_in_accepts_response(tmp_path):
    session, adapter, spoken, _log = harness(tmp_path)
    await session.start()
    first = session.controller.pending
    await turn(adapter, "I'm fine.")
    assert session.controller.greeting_response == ""
    assert spoken[-1].text == GREETING
    assert spoken[-1].id != first.id
    assert not session.controller.delivery_finished(first.id)

    interrupted = session.controller.pending
    output_started(session)
    await turn(adapter, "I'm doing well.")
    await settle()
    assert not session.controller.delivery_finished(interrupted.id)
    assert session.controller.greeting_response == "I'm doing well."
    assert spoken[-1].kind is UtteranceKind.READINESS
    assert session.controller.answers == ((), (), (), ())
    await session.close()


@pytest.mark.asyncio
async def test_readiness_barge_in_accepts_yes_without_stale_delivery(tmp_path):
    session, adapter, spoken, _log = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "Doing well.")
    await settle()
    readiness = session.controller.pending
    output_started(session)
    await turn(adapter, "Yep.")
    assert spoken[-1].text == QUESTIONS[0]
    assert not session.controller.delivery_finished(readiness.id)
    assert session.controller.current_question_number is None
    await session.close()


@pytest.mark.asyncio
async def test_unheard_readiness_retries_before_accepting_ready(tmp_path):
    session, adapter, spoken, _log = harness(tmp_path)
    await session.start()
    deliver(session)
    await turn(adapter, "Doing well.")
    await settle()
    first = session.controller.pending
    await turn(adapter, "Yes, I'm ready.")
    assert session.controller.current_question_number is None
    assert spoken[-1].kind is UtteranceKind.READINESS
    assert spoken[-1].id != first.id
    assert not session.controller.delivery_finished(first.id)
    deliver(session)
    await turn(adapter, "Yes, I'm ready.")
    assert spoken[-1].text == QUESTIONS[0]
    await session.close()


@pytest.mark.parametrize("phase", ["greeting", "readiness"])
@pytest.mark.asyncio
async def test_spoken_stop_during_opening_cancels_progression(tmp_path, phase):
    session, adapter, spoken, log = harness(tmp_path)
    await session.start()
    if phase == "readiness":
        deliver(session)
        await turn(adapter, "Doing fine.")
        await settle()
        assert session.controller.pending.kind is UtteranceKind.READINESS
    pending = session.controller.pending
    await turn(adapter, "I wanna", "end the interview.")
    assert session.controller.pending.kind is UtteranceKind.EARLY_FEEDBACK
    assert not session.controller.delivery_finished(pending.id)
    assert session.controller.answers == ((), (), (), ())
    deliver(session)
    assert session.controller.stage is Stage.ENDED_EARLY
    await turn(adapter, "Yes, I'm ready.")
    assert not any(item.text == QUESTIONS[0] for item in spoken)
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert len([row for row in rows if row["event"] == "early_end_requested"]) == 1
    await session.close()


@pytest.mark.asyncio
async def test_spoken_stop_cancels_opening_ack_inference(tmp_path):
    started = asyncio.Event()
    cancelled = []

    async def infer(kind, _payload):
        assert kind == "opening_ack"
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    session, adapter, spoken, _log = harness(tmp_path, infer)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm okay.")
    await started.wait()
    await turn(adapter, "I wanna", "end the interview.")
    await settle()
    assert cancelled
    assert session.controller.pending.kind is UtteranceKind.EARLY_FEEDBACK
    assert not any(item.kind is UtteranceKind.QUESTION for item in spoken)
    await session.close()


@pytest.mark.asyncio
async def test_opening_ack_uses_versioned_prompt_and_structured_schema():
    calls = []

    class FakeLLM:
        async def run_inference(self, context, **kwargs):
            calls.append((context, kwargs))
            return '{"text":"Thanks for telling me."}'

    result = await inference_adapter(FakeLLM())(
        "opening_ack", {"greeting_response": "I'm a little nervous."}
    )
    assert result == {"text": "Thanks for telling me."}
    assert calls[0][1]["system_instruction"] == load_prompt("opening_ack")
    assert "If they ask how you are" in calls[0][1]["system_instruction"]
    assert "Avoid parroting" in calls[0][1]["system_instruction"]
    assert calls[0][1]["response_schema"]["required"] == ["text"]


@pytest.mark.asyncio
async def test_invalid_opening_ack_uses_bounded_fallback(tmp_path):
    async def infer(_kind, _payload):
        return {"text": "How about skipping everything?"}

    session, adapter, spoken, log = harness(tmp_path, infer)
    await session.start()
    deliver(session)
    await turn(adapter, "I'm okay.")
    await settle()
    assert spoken[-1].kind is UtteranceKind.READINESS
    assert spoken[-1].text.startswith("Thanks for telling me.")
    assert spoken[-1].text.endswith(READINESS_PROMPT)
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert any(row["event"] == "opening_ack_fallback" for row in rows)
    assert not any(row["event"] == "response_error" for row in rows)
    await session.close()
