import asyncio
import json

import pytest
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

from rehearse_ai.interview import QUESTIONS, Stage
from rehearse_ai.pipeline import AudioObserver, InterviewInput, UtteranceMarkerFrame
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


async def start_questions(session):
    await session.start()
    finish_output(session, begin_output(session))
    session.controller.submit_greeting("I'm doing well")
    await session._queue_speech(session.controller.prepare_readiness("Good to hear."))
    finish_output(session, begin_output(session))
    await session._queue_speech(session.controller.submit_readiness(True))


def events(log):
    return [json.loads(line) for line in log.path.read_text().splitlines()]


def final_segment(text, number):
    return TranscriptionFrame(
        text=text, user_id="candidate", timestamp=f"segment-{number}", finalized=False
    )


async def feed_turn(adapter, segments):
    await adapter.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    for number, segment in enumerate(segments, start=1):
        await adapter.process_frame(final_segment(segment, number), FrameDirection.DOWNSTREAM)
    await adapter.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)


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
    session.tts_context_assigned(utterance_id, "tts-context")
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
        row["event"] == "tts_generation_finished"
        and row["utterance_id"] == utterance_id
        and row["audio_bytes"] == 32
        and row["audio_chunks"] == 1
        for row in events(log)
    )
    assert session.controller.stage is Stage.DELIVERING
    await session.close()


@pytest.mark.asyncio
async def test_audio_from_cancelled_tts_context_is_not_forwarded(tmp_path):
    async def infer(_kind, _payload):
        raise AssertionError("No inference expected")

    async def speak(_utterance):
        pass

    log = SessionLog(tmp_path, "late-audio")
    session = InterviewSession(log, infer, speak, lambda _: None)
    observer = AudioObserver()
    observer.session = session
    forwarded = []

    async def forward(frame, _direction=FrameDirection.DOWNSTREAM):
        forwarded.append(frame)

    observer.push_frame = forward
    await session.start()
    first = session.controller.pending
    session.tts_context_assigned(first.id, "old")
    await observer.process_frame(UtteranceMarkerFrame(first.id), FrameDirection.DOWNSTREAM)
    await observer.process_frame(TTSStartedFrame(context_id="old"), FrameDirection.DOWNSTREAM)
    await session.user_started()
    await observer.process_frame(
        TTSAudioRawFrame(b"\x01" * 32, 24000, 1, context_id="old"),
        FrameDirection.DOWNSTREAM,
    )
    assert not any(isinstance(frame, TTSAudioRawFrame) for frame in forwarded)
    assert not any(
        row["event"] == "tts_first_audio_generated" and row["utterance_id"] == first.id
        for row in events(log)
    )
    assert any(row["event"] == "stale_tts_audio_discarded" for row in events(log))
    await session.close()


@pytest.mark.asyncio
async def test_late_cancelled_tts_stop_cannot_finish_fresh_output(tmp_path):
    async def infer(_kind, _payload):
        raise AssertionError("No inference expected")

    async def speak(_utterance):
        pass

    log = SessionLog(tmp_path, "late-stop")
    session = InterviewSession(log, infer, speak, lambda _: None)
    observer = AudioObserver()
    observer.session = session
    forwarded_stops = []

    class Output:
        async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
            if direction is FrameDirection.DOWNSTREAM and isinstance(
                frame, BotStoppedSpeakingFrame
            ):
                session.bot_stopped()

    sender = BaseOutputTransport.MediaSender(
        Output(),
        destination=None,
        sample_rate=24000,
        audio_chunk_size=4800,
        params=TransportParams(),
    )

    async def forward(frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, TTSStoppedFrame):
            forwarded_stops.append(frame.context_id)
            await sender._handle_frame(frame)

    observer.push_frame = forward
    await session.start()
    first = session.controller.pending
    session.tts_context_assigned(first.id, "old")
    await observer.process_frame(TTSStartedFrame(context_id="old"), FrameDirection.DOWNSTREAM)
    await observer.process_frame(
        TTSAudioRawFrame(b"\x01" * 32, 24000, 1, context_id="old"),
        FrameDirection.DOWNSTREAM,
    )
    session.bot_started()
    await session.user_started()
    session.bot_stopped()  # Pipecat's interruption stop retires A's stop debt.

    second = session.controller.retry_interrupted()
    await session._queue_speech(second)
    session.tts_context_assigned(second.id, "new")
    await observer.process_frame(TTSStartedFrame(context_id="new"), FrameDirection.DOWNSTREAM)
    await observer.process_frame(
        TTSAudioRawFrame(b"\x02" * 32, 24000, 1, context_id="new"),
        FrameDirection.DOWNSTREAM,
    )
    session.bot_started()
    sender._bot_speaking = True
    sender._tts_audio_received = True

    await observer.process_frame(
        TTSAudioRawFrame(b"\x03" * 32, 24000, 1, context_id="old"),
        FrameDirection.DOWNSTREAM,
    )
    await observer.process_frame(TTSStoppedFrame(context_id="old"), FrameDirection.DOWNSTREAM)
    await observer.process_frame(TTSStoppedFrame(context_id="new"), FrameDirection.DOWNSTREAM)
    await observer.process_frame(TTSStoppedFrame(context_id="new"), FrameDirection.DOWNSTREAM)

    deliveries = [row for row in events(log) if row["event"] == "agent_delivery_reported"]
    assert forwarded_stops == ["new"]
    assert [row["utterance_id"] for row in deliveries] == [second.id]
    assert session.controller.stage is Stage.AWAITING_GREETING
    assert not any(row["utterance_id"] == first.id for row in deliveries)
    assert any(row["event"] == "stale_tts_audio_discarded" for row in events(log))
    assert len([row for row in events(log) if row["event"] == "stale_tts_stop_discarded"]) == 2
    await session.close()


@pytest.mark.asyncio
async def test_fresh_output_retires_missing_cancelled_stop_debt(tmp_path):
    async def infer(_kind, _payload):
        raise AssertionError("No inference expected")

    async def speak(_utterance):
        pass

    log = SessionLog(tmp_path, "stop-debt")
    session = InterviewSession(log, infer, speak, lambda _: None)
    await session.start()
    first = session.controller.pending
    session.tts_started(first.id, "old")
    session.tts_first_audio("old")
    session.bot_started()
    await session.user_started()  # No cancellation BotStopped callback is observed.

    second = session.controller.retry_interrupted()
    await session._queue_speech(second)
    session.tts_started(second.id, "new")
    session.tts_first_audio("new")
    session.bot_started()
    session.tts_finished("new")
    session.bot_stopped()

    deliveries = [row for row in events(log) if row["event"] == "agent_delivery_reported"]
    assert [row["utterance_id"] for row in deliveries] == [second.id]
    assert any(row["event"] == "cancelled_output_stop_debt_retired" for row in events(log))
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
    await start_questions(session)
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
    assert len(spoken) == 4
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
    await start_questions(session)
    finish_output(session, begin_output(session))
    await session.user_started()
    await session.user_stopped(spoken_request)
    assert session.controller.answers[0] == ()
    assert session.controller.pending.kind.name == "EARLY_FEEDBACK"
    await session.close()


@pytest.mark.parametrize(
    "segments",
    [
        ["Can I", "end this interview?"],
        ["Can we end", "the interview?"],
        ["I want you to", "end the interview right now."],
        ["Want you to end the interview.", "Right now."],
        ["Stop the interview."],
        ["Stop the interview, please."],
        ["I'd like to stop."],
        ["I wanna", "end the interview."],
        ["I don't wanna", "proceed further."],
    ],
)
@pytest.mark.asyncio
async def test_spoken_stop_segments_enter_early_closing_before_turn_completion(tmp_path, segments):
    input_adapter = InterviewInput()
    input_adapter.push_frame = lambda *_args, **_kwargs: asyncio.sleep(0)
    calls = []
    cancelled = []

    async def infer(kind, payload):
        calls.append((kind, payload))
        raise AssertionError("Stop request must not reach ordinary inference")

    async def cancel_audio():
        cancelled.append(True)

    log = SessionLog(tmp_path, "segmented-stop")
    session = InterviewSession(
        log, infer, input_adapter.speak, lambda _: None, cancel_audio=cancel_audio
    )
    input_adapter.session = session
    await start_questions(session)
    finish_output(session, begin_output(session))

    await input_adapter.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    for number, segment in enumerate(segments, start=1):
        await input_adapter.process_frame(final_segment(segment, number), FrameDirection.DOWNSTREAM)
    assert session.controller.stage is Stage.EARLY_CLOSING
    await input_adapter.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert session.controller.pending.kind.name == "EARLY_FEEDBACK"
    assert session.controller.answers == ((), (), (), ())
    assert not calls
    assert cancelled == []  # Pipecat's speech-onset interruption owns audio cancellation.
    rows = events(log)
    early = next(row["sequence"] for row in rows if row["event"] == "early_end_requested")
    assert (
        next(row["detection_source"] for row in rows if row["event"] == "early_end_requested")
        == "final_stt_segments"
    )
    completed = next(row["sequence"] for row in rows if row["event"] == "candidate_final")
    assert early < completed
    assert [row["text"] for row in rows if row["event"] == "stt_final_segment"] == segments
    await session.close()


@pytest.mark.parametrize(
    "speech",
    [
        "I ended the interview yesterday.",
        "Don't end the interview.",
        "I don't want to end the interview.",
        "I don't wanna end the interview.",
        "I don't wanna proceed further with this project.",
    ],
)
@pytest.mark.asyncio
async def test_non_requests_reach_answer_route_without_early_end(tmp_path, speech):
    input_adapter = InterviewInput()
    input_adapter.push_frame = lambda *_args, **_kwargs: asyncio.sleep(0)
    calls = []

    async def infer(kind, _payload):
        calls.append(kind)
        return {"action": "wait", "text": ""}

    log = SessionLog(tmp_path, "not-a-stop")
    session = InterviewSession(log, infer, input_adapter.speak, lambda _: None)
    input_adapter.session = session
    await start_questions(session)
    finish_output(session, begin_output(session))
    await feed_turn(input_adapter, [speech])
    await settle()
    assert calls == ["decision"]
    assert session.controller.answers[0] == (speech,)
    assert not any(row["event"] == "early_end_requested" for row in events(log))
    await session.close()


@pytest.mark.asyncio
async def test_stop_invalidates_pending_next_question_before_it_can_play(tmp_path):
    input_adapter = InterviewInput()
    input_adapter.push_frame = lambda *_args, **_kwargs: asyncio.sleep(0)
    calls = []
    cancellations = []

    async def infer(kind, payload):
        calls.append(kind)
        if kind == "decision":
            return {"action": "advance", "text": ""}
        assert payload["answers_by_question"] == [["I built an API"], [], [], []]
        return {"lines": ["We'll stop early; you described building an API."]}

    async def cancel_audio():
        cancellations.append(True)

    log = SessionLog(tmp_path, "stop-before-next-question")
    session = InterviewSession(
        log, infer, input_adapter.speak, lambda _: None, cancel_audio=cancel_audio
    )
    input_adapter.session = session
    await start_questions(session)
    finish_output(session, begin_output(session))
    await feed_turn(input_adapter, ["I built an API"])
    await settle()
    pending_question = session.controller.pending
    assert pending_question.text == QUESTIONS[1]

    await input_adapter.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    for number, segment in enumerate(["Want you to end the interview.", "Right now."], start=1):
        await input_adapter.process_frame(final_segment(segment, number), FrameDirection.DOWNSTREAM)
    assert session.controller.stage is Stage.EARLY_CLOSING
    session.tts_started(pending_question.id, "late-context")
    session.tts_first_audio("late-context")
    session.tts_finished("late-context")
    session.bot_started()
    session.bot_stopped()
    await input_adapter.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await settle()
    assert calls == ["decision", "early_feedback"]
    assert cancellations == []  # UserTurnProcessor interrupts audio on speech onset.
    assert session.controller.answers[1] == ()
    assert session.controller.pending.kind.name == "EARLY_FEEDBACK"
    assert not session.controller.delivery_finished(pending_question.id)
    rows = events(log)
    assert not any(
        row["event"] in {"agent_output_started", "agent_delivery_reported"}
        and row["utterance_id"] == pending_question.id
        for row in rows
    )
    await session.close()


@pytest.mark.asyncio
async def test_latest_stop_request_cancels_pending_question_and_latches(tmp_path):
    input_adapter = InterviewInput()
    input_adapter.push_frame = lambda *_args, **_kwargs: asyncio.sleep(0)
    calls = []
    spoken = []

    async def infer(kind, payload):
        calls.append(kind)
        if kind == "decision":
            return {"action": "advance", "text": ""}
        assert payload["answers_by_question"] == [["I built an API"], [], [], []]
        return {"lines": ["We'll end early; you described building an API."]}

    async def speak(utterance):
        spoken.append(utterance)

    log = SessionLog(tmp_path, "wanna-stop")
    session = InterviewSession(log, infer, speak, lambda _: None)
    input_adapter.session = session
    await start_questions(session)
    finish_output(session, begin_output(session))
    await feed_turn(input_adapter, ["I built an API"])
    await settle()
    pending_question = session.controller.pending
    assert pending_question.text == QUESTIONS[1]

    await input_adapter.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await input_adapter.process_frame(final_segment("I wanna", 1), FrameDirection.DOWNSTREAM)
    await input_adapter.process_frame(
        final_segment("end the interview.", 2), FrameDirection.DOWNSTREAM
    )
    assert session.controller.stage is Stage.EARLY_CLOSING
    session.tts_started(pending_question.id, "late-question")
    session.tts_first_audio("late-question")
    session.tts_finished("late-question")
    session.bot_started()
    session.bot_stopped()
    await input_adapter.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await settle()
    assert calls == ["decision", "early_feedback"]
    assert session.controller.pending.kind.name == "EARLY_FEEDBACK"
    assert not any(
        row["event"] in {"agent_output_started", "agent_delivery_reported"}
        and row["utterance_id"] == pending_question.id
        for row in events(log)
    )

    # Further candidate speech may interrupt the closing, but cannot reopen a question.
    await input_adapter.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await input_adapter.process_frame(final_segment("I mean, now", 3), FrameDirection.DOWNSTREAM)
    await input_adapter.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert session.controller.pending.kind.name == "EARLY_FEEDBACK"
    assert calls == ["decision", "early_feedback"]
    finish_output(session, begin_output(session))
    assert session.controller.stage is Stage.ENDED_EARLY
    await feed_turn(input_adapter, ["No."])
    assert session.controller.stage is Stage.ENDED_EARLY
    assert not any(utterance.text == QUESTIONS[2] for utterance in spoken)
    await session.close()


@pytest.mark.asyncio
async def test_withdrawal_in_noisy_segmented_answer_bypasses_decision(tmp_path):
    input_adapter = InterviewInput()
    input_adapter.push_frame = lambda *_args, **_kwargs: asyncio.sleep(0)
    calls = []

    async def infer(kind, payload):
        calls.append(kind)
        if kind == "decision":
            return {"action": "advance", "text": ""}
        assert payload["answers_by_question"] == [["I built an API"], [], [], []]
        return {"lines": ["We'll end early; thanks for describing your API."]}

    log = SessionLog(tmp_path, "withdrawal")
    session = InterviewSession(log, infer, input_adapter.speak, lambda _: None)
    input_adapter.session = session
    await start_questions(session)
    finish_output(session, begin_output(session))
    await feed_turn(input_adapter, ["I built an API"])
    await settle()
    finish_output(session, begin_output(session))
    assert session.controller.current_question_number == 2

    await input_adapter.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await input_adapter.process_frame(
        final_segment("I want to enter entity because I don't wanna", 1),
        FrameDirection.DOWNSTREAM,
    )
    assert session.controller.stage is Stage.AWAITING_ANSWER
    await input_adapter.process_frame(
        final_segment("proceed further. There's no technical", 2),
        FrameDirection.DOWNSTREAM,
    )
    assert session.controller.stage is Stage.EARLY_CLOSING
    await input_adapter.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await settle()
    assert calls == ["decision", "early_feedback"]
    assert session.controller.answers[1] == ()
    assert session.controller.pending.kind.name == "EARLY_FEEDBACK"
    assert not any(
        row["event"] == "agent_prepared" and row["text"] == QUESTIONS[2] for row in events(log)
    )
    await session.close()


@pytest.mark.asyncio
async def test_stop_latch_discards_cancellation_resistant_decision(tmp_path):
    input_adapter = InterviewInput()
    input_adapter.push_frame = lambda *_args, **_kwargs: asyncio.sleep(0)
    decision_started = asyncio.Event()
    release_decision = asyncio.Event()
    calls = []

    async def infer(kind, payload):
        calls.append(kind)
        if kind == "decision":
            decision_started.set()
            while not release_decision.is_set():
                try:
                    await release_decision.wait()
                except asyncio.CancelledError:
                    pass
            return {"action": "advance", "text": ""}
        assert payload["answers_by_question"] == [["I built an API"], [], [], []]
        return {"lines": ["We'll end early; thanks for describing your API."]}

    log = SessionLog(tmp_path, "late-decision")
    session = InterviewSession(log, infer, input_adapter.speak, lambda _: None)
    input_adapter.session = session
    await start_questions(session)
    finish_output(session, begin_output(session))
    await feed_turn(input_adapter, ["I built an API"])
    await decision_started.wait()
    await feed_turn(input_adapter, ["I wanna", "end the interview."])
    release_decision.set()
    await settle()
    assert calls == ["decision", "early_feedback"]
    assert session.controller.pending.kind.name == "EARLY_FEEDBACK"
    assert not any(
        row["event"] == "agent_prepared" and row["text"] == QUESTIONS[1] for row in events(log)
    )
    await session.close()


@pytest.mark.asyncio
async def test_speech_before_first_output_retries_unheard_question(tmp_path):
    input_adapter = InterviewInput()
    input_adapter.push_frame = lambda *_args, **_kwargs: asyncio.sleep(0)

    async def infer(_kind, payload):
        assert payload["phase"] == "interrupted_turn"
        return {"action": "wait", "text": ""}

    log = SessionLog(tmp_path, "startup")
    session = InterviewSession(log, infer, input_adapter.speak, lambda _: None)
    input_adapter.session = session
    await start_questions(session)
    first_id = session.controller.pending.id
    await feed_turn(input_adapter, ["I am proud of a project I built."])
    await settle()
    assert session.controller.answers == ((), (), (), ())
    assert session.controller.pending.id != first_id
    assert session.controller.pending.text == QUESTIONS[0]
    assert not any(
        row["event"] == "agent_output_started" and row["utterance_id"] == first_id
        for row in events(log)
    )
    await session.close()


@pytest.mark.asyncio
async def test_cancel_between_utterance_marker_and_tts_discards_question(tmp_path):
    input_adapter = InterviewInput()
    marker_sent = asyncio.Event()
    release_marker = asyncio.Event()
    frames = []

    async def push(frame, _direction=FrameDirection.DOWNSTREAM):
        frames.append(frame)
        if isinstance(frame, UtteranceMarkerFrame) and frame.utterance_id == 4:
            marker_sent.set()
            try:
                await release_marker.wait()
            except asyncio.CancelledError:
                await release_marker.wait()

    input_adapter.push_frame = push

    async def infer(kind, _payload):
        if kind == "decision":
            return {"action": "advance", "text": ""}
        return {"lines": ["We'll stop early. Thanks for sharing your project."]}

    session = InterviewSession(
        SessionLog(tmp_path, "tts-race"), infer, input_adapter.speak, lambda _: None
    )
    input_adapter.session = session
    await start_questions(session)
    finish_output(session, begin_output(session))
    await session.user_started()
    await session.user_stopped("I built an API")
    await marker_sent.wait()
    await session.user_started()
    await session.detect_stop_from_segments("Can I end this interview?")
    release_marker.set()
    await session.user_stopped("Can I end this interview?")
    await settle()
    assert not any(
        isinstance(frame, TTSSpeakFrame) and frame.text == QUESTIONS[1] for frame in frames
    )
    assert session.controller.pending.kind.name == "EARLY_FEEDBACK"
    await session.close()


@pytest.mark.asyncio
async def test_decision_receives_complete_accumulated_answer_after_nudge(tmp_path):
    input_adapter = InterviewInput()
    input_adapter.push_frame = lambda *_args, **_kwargs: asyncio.sleep(0)
    payloads = []

    async def infer(_kind, payload):
        payloads.append(payload)
        if len(payloads) == 1:
            return {"action": "nudge", "text": "Tell me more about that project."}
        return {"action": "advance", "text": ""}

    session = InterviewSession(
        SessionLog(tmp_path, "complete-answer"), infer, input_adapter.speak, lambda _: None
    )
    input_adapter.session = session
    await start_questions(session)
    finish_output(session, begin_output(session))
    await feed_turn(input_adapter, ["I'm proud of a first-semester project."])
    await settle()
    finish_output(session, begin_output(session))
    segments = [
        "It posts a structured risk report",
        "as a PR comment",
        "in under sixty seconds.",
    ]
    await feed_turn(input_adapter, segments)
    await settle()
    current = payloads[-1]["current_question_answer"]
    assert current == ("I'm proud of a first-semester project. " + " ".join(segments))
    assert payloads[-1]["current_question_answers"][-1] == " ".join(segments)
    assert "one missing concrete detail" in load_prompt("decision")
    assert session.controller.pending.text == QUESTIONS[1]
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
    await start_questions(session)
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
    await start_questions(session)
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
    await start_questions(session)
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
    assert payloads[-1][1]["current_question"] == QUESTIONS[1]
    assert payloads[-1][1]["last_delivered_prompt"] == QUESTIONS[0]
    assert len(payloads) == 2  # One interpretation for each completed candidate turn.
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
    await start_questions(session)
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
    await start_questions(session)
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
