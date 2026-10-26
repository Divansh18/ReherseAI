"""Connect one interview controller to speech, inference, and session events."""

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from rehearse_ai.interview import (
    InterviewController,
    LimitReached,
    Readiness,
    ReplyKind,
    Stage,
    Utterance,
    UtteranceKind,
)
from rehearse_ai.session_log import SessionLog

Infer = Callable[[str, dict[str, Any]], Awaitable[Mapping[str, Any]]]
Speak = Callable[[Utterance], Awaitable[None]]
Publish = Callable[[dict[str, Any]], None]
CancelAudio = Callable[[], Awaitable[None]]

_REPEAT = re.compile(
    r"^(?:please\s+)?(?:can you\s+|could you\s+|would you\s+)?"
    r"(?:repeat(?: that| the (?:question|last question))?"
    r"|say (?:that|the question) again)\??[.!]?$",
    re.IGNORECASE,
)
_SKIP = re.compile(r"^(?:please\s+)?(?:skip(?: this question)?|next question)\??[.!]?$", re.I)
_END_CLAUSE = re.compile(
    r"(?:please\s+)?(?:"
    r"(?:can|could|may|would)\s+(?:i|we|you)\s+"
    r"|(?:i|we)\s+(?:want|would like|need)\s+(?:you\s+)?to\s+"
    r"|(?:i'd|we'd)\s+like\s+to\s+"
    r"|want\s+you\s+to\s+"
    r"|let's\s+"
    r")?"
    r"(?:end|stop|finish|quit|exit|wrap up)"
    r"(?:\s+(?:(?:the|this|our)\s+)?(?:interview|session|practice)"
    r"|\s+(?:now|here|it))?"
    r"(?:\s+right\s+now|\s+please)?",
    re.I,
)
_DONE_CLAUSE = re.compile(r"(?:i'm|i am|we're|we are)\s+(?:done|finished)", re.I)
_WITHDRAW_CLAUSE = re.compile(
    r"(?:i|we)\s+(?:don't|do not|dont)\s+want\s+to\s+"
    r"(?:proceed|continue|go on|carry on|keep going)"
    r"(?:\s+(?:further|anymore|now|with (?:this|the) (?:interview|session|practice)))?",
    re.I,
)


_DEFER_READINESS = re.compile(
    r"\b(?:not (?:ready|yet|now)|do not (?:start|begin)|do not want to"
    r"|give me (?:a |one )?(?:moment|minute|second)|need (?:a |one )?(?:moment|minute|second)"
    r"|hold on|wait|maybe later|not quite|later)\b"
)
_AFFIRM_READINESS = re.compile(
    r"\b(?:i am ready(?: to (?:start|begin|go))?|i am good to go"
    r"|let us (?:get|go) into it|let us (?:start|begin|go|do it)"
    r"|we can (?:start|begin)|go ahead|ready to (?:start|begin|go))"
    r"(?: now| please)?$"
)
_BARE_AFFIRM_READINESS = re.compile(
    r"(?:yes|yeah|yep|sure|okay|ok|absolutely|of course|sounds good|ready"
    r"|start|begin|let us go|let us do it)(?: now| please)?"
)
_FINAL_START = re.compile(r"\b(?:start|begin)(?: now| please)?$")


def _classify_readiness(text: str) -> Readiness:
    normalized = text.lower().replace("’", "'")
    for old, new in {"i'm": "i am", "let's": "let us", "don't": "do not"}.items():
        normalized = normalized.replace(old, new)
    words = re.findall(r"[a-z]+", normalized)
    words = [word for index, word in enumerate(words) if index == 0 or word != words[index - 1]]
    normalized = " ".join(words)
    if _DEFER_READINESS.search(normalized):
        return Readiness.DEFERRED
    if (
        _AFFIRM_READINESS.search(normalized)
        or _BARE_AFFIRM_READINESS.fullmatch(normalized)
        or _FINAL_START.search(normalized)
    ):
        return Readiness.READY
    return Readiness.UNCLEAR


def _classify_greeting_readiness(text: str) -> Readiness:
    classification = _classify_readiness(text)
    if classification is not Readiness.UNCLEAR:
        return classification
    clauses = [_classify_readiness(clause) for clause in re.split(r"[.!?]+", text)]
    if Readiness.DEFERRED in clauses:
        return Readiness.DEFERRED
    if Readiness.READY in clauses:
        return Readiness.READY
    return Readiness.UNCLEAR


def _fallback_opening_ack(text: str) -> str:
    if re.search(r"\bhow (?:are|r) you\b|\bhow's it going\b", text, re.I):
        return "Doing well, thanks."
    return "Thanks for telling me."


def _is_stop_request(text: str) -> bool:
    """Match a complete, affirmative request clause, including STT sentence splits."""
    normalized = re.sub(r"\bwanna\b", "want to", text.replace("’", "'"), flags=re.I)
    for sentence in re.split(r"[.!?]+", normalized):
        for clause in re.split(r"\b(?:but|because)\b", sentence, flags=re.I):
            clause = " ".join(re.sub(r"[,;:]", " ", clause).split())
            if (
                _END_CLAUSE.fullmatch(clause)
                or _DONE_CLAUSE.fullmatch(clause)
                or _WITHDRAW_CLAUSE.fullmatch(clause)
            ):
                return True
    return False


class InterviewSession:
    def __init__(
        self,
        log: SessionLog,
        infer: Infer,
        speak: Speak,
        publish: Publish,
        *,
        turn_pause_secs: float = 1.1,
        cancel_audio: CancelAudio | None = None,
        build_id: str | None = None,
    ) -> None:
        self.controller = InterviewController()
        self.log = log
        self._infer = infer
        self._speak = speak
        self._publish = publish
        self._cancel_audio = cancel_audio
        self._response_task: asyncio.Task[None] | None = None
        self._generation = 0
        self._speaking_id: int | None = None
        self._active_prompt_id: int | None = None
        self._last_delivered_prompt: Utterance | None = None
        self._output_started_ids: set[int] = set()
        self._tts_completed_ids: set[int] = set()
        self._tts_contexts: dict[str, int] = {}
        self._first_audio_ids: set[int] = set()
        self._interrupted_output_id: int | None = None
        self._cancelled_output_stop_debt = 0
        self._last_turn_end: float | None = None
        self._turn_started_at: float | None = None
        self._turn_number = 0
        self._answer_number = 0
        self._turn_pause_secs = turn_pause_secs
        self._build_id = build_id
        self._stop_latched = False
        self._closed = False

    @property
    def response_task(self) -> asyncio.Task[None] | None:
        return self._response_task

    def _record(self, event: str, **details: Any) -> None:
        entry = self.log.record(event, **details)
        self._publish({"type": "rehearseai", "event": entry})

    def turn_signal(self, signal: str, **details: Any) -> None:
        if not self._closed:
            self._record(signal, speaker="candidate", turn_number=self._turn_number, **details)

    def transcript_segment(
        self, text: str, *, frame_timestamp: str, finalized: bool, frame_id: int
    ) -> None:
        if not self._closed:
            self._record(
                "stt_final_segment",
                speaker="candidate",
                text=text,
                turn_number=self._turn_number,
                frame_timestamp=frame_timestamp,
                finalized=finalized,
                frame_id=frame_id,
            )

    async def detect_stop_from_segments(self, text: str) -> bool:
        if (
            self._closed
            or self._stop_latched
            or self.controller.stage
            in {
                Stage.COMPLETE,
                Stage.ENDED_EARLY,
                Stage.EARLY_CLOSING,
            }
        ):
            return False
        if not _is_stop_request(text):
            return False
        await self._end_early(text, defer_closing=True)
        return True

    async def start(self) -> None:
        if self._closed or self.controller.stage is not Stage.NEW:
            return
        self._record("session_started", speaker=None, build_id=self._build_id)
        await self._queue_speech(self.controller.start())

    async def _queue_speech(self, utterance: Utterance) -> None:
        if (
            self._closed
            or self.controller.pending != utterance
            or (self._stop_latched and utterance.kind is not UtteranceKind.EARLY_FEEDBACK)
        ):
            return
        self._active_prompt_id = utterance.id
        self._record(
            "agent_prepared",
            speaker="agent",
            utterance_id=utterance.id,
            text=utterance.text,
            interrupted=False,
            question_number=utterance.question_number,
            utterance_kind=utterance.kind.name.lower(),
            generation=self._generation,
            spoken=False,
            browser_playback_onset_ms=None,
        )
        self._record(
            "tts_requested", speaker=None, utterance_id=utterance.id, generation=self._generation
        )
        try:
            await self._speak(utterance)
        except Exception as exc:
            self._record(
                "response_error",
                speaker=None,
                error_type=type(exc).__name__,
                failure_stage="speech_queue",
                utterance_id=utterance.id,
            )

    async def user_started(self) -> None:
        if self._closed or self.controller.stage in {Stage.COMPLETE, Stage.ENDED_EARLY}:
            return
        self._generation += 1
        task = self._response_task
        if task and not task.done():
            self._record(
                "inference_cancel_requested", speaker=None, generation=self._generation - 1
            )
            task.cancel()
        self._response_task = None
        pending = self.controller.pending
        if pending is not None and self.controller.interrupt(pending.id):
            self._interrupted_output_id = (
                pending.id if pending.id in self._output_started_ids else None
            )
            if self._interrupted_output_id is not None:
                self._cancelled_output_stop_debt += 1
            self._speaking_id = None
            self._tts_contexts = {
                context: identifier
                for context, identifier in self._tts_contexts.items()
                if identifier != pending.id
            }
            self._record(
                "agent_interrupted",
                speaker="agent",
                utterance_id=pending.id,
                text=pending.text,
                interrupted=True,
                output_started=self._interrupted_output_id is not None,
                cancellation_reason="candidate_speech_onset",
                generation=self._generation - 1,
            )
            self._record(
                "agent_cancellation",
                speaker=None,
                utterance_id=pending.id,
                generation=self._generation - 1,
            )
        self._turn_number += 1
        self._turn_started_at = time.perf_counter()
        self._record(
            "candidate_started",
            speaker="candidate",
            turn_number=self._turn_number,
            question_number=self.controller.current_question_number,
        )

    async def user_stopped(
        self,
        text: str,
        *,
        segment_count: int | None = None,
        boundary_frame_id: int | None = None,
    ) -> None:
        if self._closed or self.controller.stage in {Stage.COMPLETE, Stage.ENDED_EARLY}:
            return
        answer = text.strip()
        self._last_turn_end = time.perf_counter()
        continuing = self.controller.stage in {
            Stage.AWAITING_OPENING_ACK,
            Stage.AWAITING_PROPOSAL,
            Stage.AWAITING_FEEDBACK,
        }
        if not continuing:
            self._answer_number += 1
        duration = (
            round((self._last_turn_end - self._turn_started_at) * 1000, 1)
            if self._turn_started_at is not None
            else None
        )
        self._record(
            "candidate_final",
            speaker="candidate",
            text=answer,
            turn_number=self._turn_number,
            answer_number=self._answer_number,
            question_number=self.controller.current_question_number,
            prompt_utterance_id=self._active_prompt_id,
            segment_count=segment_count,
            turn_duration_ms=duration,
            boundary="Pipecat UserStoppedSpeakingFrame",
            boundary_frame_id=boundary_frame_id,
            configured_pause_secs=self._turn_pause_secs,
            speech_end_estimate="Pipecat VAD stop before speech-timeout boundary",
        )
        self._record(
            "candidate_turn_completed",
            speaker="candidate",
            turn_number=self._turn_number,
            boundary_frame_id=boundary_frame_id,
            boundary_method="Pipecat SpeechTimeoutUserTurnStopStrategy",
            configured_pause_secs=self._turn_pause_secs,
        )
        self._turn_started_at = None

        if not self._stop_latched and _is_stop_request(answer):
            await self._end_early(answer)
            return
        if self._stop_latched:
            if self.controller.stage is Stage.EARLY_CLOSING:
                if any(self.controller.answers):
                    self._start_response()
                else:
                    await self._queue_speech(
                        self.controller.prepare_early_feedback(
                            "We'll end here. Thanks for practicing."
                        )
                    )
            elif self.controller.interrupted_utterance is not None:
                await self._queue_speech(self.controller.retry_interrupted())
            return

        if await self._route_opening(answer):
            return

        if _SKIP.fullmatch(answer) and self.controller.interrupted_utterance is not None:
            try:
                utterance = self.controller.skip_interrupted()
            except ValueError:
                return
            self._record("candidate_skipped_question", speaker="candidate")
            if utterance is not None:
                await self._queue_speech(utterance)
            else:
                self._start_response()
            return

        interrupted = self.controller.interrupted_utterance
        if interrupted is not None:
            if interrupted.kind.name == "EARLY_FEEDBACK":
                await self._queue_speech(self.controller.retry_interrupted())
                return
            if not answer or _REPEAT.fullmatch(answer):
                await self._queue_speech(self.controller.retry_interrupted())
                return
            if self._interrupted_output_id != interrupted.id:
                self._record(
                    "unheard_prompt_recovery",
                    speaker=None,
                    utterance_id=interrupted.id,
                    candidate_text=answer,
                )
                if interrupted.kind.name == "QUESTION":
                    retry = self.controller.resolve_unheard_question(answer)
                    await self._queue_speech(retry)
                    return
                elif interrupted.kind.name in {"FOLLOW_UP", "NUDGE", "CLARIFICATION", "REPEAT"}:
                    await self._queue_speech(self.controller.resolve_unheard_follow_up(answer))
                    return
                else:
                    await self._queue_speech(self.controller.retry_interrupted())
                    return
            else:
                self.controller.accept_barge_in_answer(answer)
                self._record("barge_in_answer_accepted", speaker="candidate", text=answer)
            self._interrupted_output_id = None
        elif self.controller.stage is Stage.AWAITING_ANSWER:
            try:
                if _SKIP.fullmatch(answer):
                    utterance = self.controller.skip_current()
                    self._record("candidate_skipped_question", speaker="candidate")
                elif _REPEAT.fullmatch(answer):
                    utterance = self.controller.request_repeat()
                else:
                    utterance = self.controller.submit_answer(answer)
            except LimitReached:
                self._record("clarification_limit", speaker=None)
                return
            if utterance is not None:
                await self._queue_speech(utterance)
                return
            if _SKIP.fullmatch(answer):
                self._start_response()
                return
        elif self.controller.stage is Stage.AWAITING_PROPOSAL:
            if not answer:
                return
            self.controller.add_answer_continuation(answer)
        elif self.controller.stage is Stage.AWAITING_FEEDBACK:
            if not answer:
                return
            self.controller.add_closing_remarks(answer)
        else:
            return

        self._start_response()

    async def _route_opening(self, answer: str) -> bool:
        interrupted = self.controller.interrupted_utterance
        opening_kinds = {
            UtteranceKind.GREETING,
            UtteranceKind.OPENING_ACK,
            UtteranceKind.READINESS,
            UtteranceKind.OPENING_WAIT,
            UtteranceKind.OPENING_CLARIFICATION,
        }
        if interrupted is not None and interrupted.kind in opening_kinds:
            if not answer or _REPEAT.fullmatch(answer):
                await self._queue_speech(self.controller.retry_interrupted())
                return True
            if (
                interrupted.kind
                not in {
                    UtteranceKind.OPENING_WAIT,
                    UtteranceKind.OPENING_CLARIFICATION,
                }
                and self._interrupted_output_id != interrupted.id
            ):
                self._record(
                    "unheard_prompt_recovery",
                    speaker=None,
                    utterance_id=interrupted.id,
                    candidate_text=answer,
                )
                await self._queue_speech(self.controller.retry_interrupted())
                return True
            self.controller.accept_barge_in_answer(answer)
            self._interrupted_output_id = None
            self._record("barge_in_answer_accepted", speaker="candidate", text=answer)

        if self.controller.stage is Stage.AWAITING_GREETING:
            try:
                utterance = (
                    self.controller.request_opening_repeat()
                    if not answer or _REPEAT.fullmatch(answer)
                    else self.controller.submit_greeting(answer)
                )
            except LimitReached:
                self._record("opening_repeat_limit", speaker=None, opening_phase="greeting")
                return True
            if utterance is not None:
                await self._queue_speech(utterance)
            else:
                self._start_response()
            return True

        if self.controller.stage is Stage.AWAITING_QUESTION_START:
            await self._queue_speech(self.controller.begin_question_one())
            return True

        if self.controller.stage is Stage.AWAITING_OPENING_ACK:
            if _REPEAT.fullmatch(answer):
                try:
                    await self._queue_speech(self.controller.request_opening_repeat())
                except LimitReached:
                    self._record("opening_repeat_limit", speaker=None, opening_phase="greeting")
                    self._start_response()
                return True
            if answer and interrupted is None:
                self.controller.add_greeting_continuation(answer)
            self._start_response()
            return True

        if self.controller.stage is Stage.AWAITING_READINESS:
            if not answer or _REPEAT.fullmatch(answer):
                try:
                    await self._queue_speech(self.controller.request_opening_repeat())
                except LimitReached:
                    self._record("opening_repeat_limit", speaker=None, opening_phase="readiness")
                return True
            classification = _classify_readiness(answer)
            self._record(
                "readiness_response",
                speaker="candidate",
                classification=classification.name.lower(),
                ready=classification is Readiness.READY,
            )
            utterance = (
                self.controller.clarify_readiness()
                if classification is Readiness.UNCLEAR
                else self.controller.submit_readiness(classification is Readiness.READY)
            )
            if utterance is not None:
                await self._queue_speech(utterance)
            else:
                self._record(
                    "opening_waiting_for_readiness",
                    speaker=None,
                    classification=classification.name.lower(),
                    guidance="Say start when ready, not yet to wait, or end the interview to stop.",
                )
            return True

        return False

    async def _end_early(self, request: str, *, defer_closing: bool = False) -> None:
        if self._stop_latched:
            return
        self._stop_latched = True
        self._generation += 1
        task = self._response_task
        if task and not task.done():
            task.cancel()
        self._response_task = None
        pending = self.controller.pending
        if self.controller.end_early():
            self._speaking_id = None
            self._record(
                "early_end_requested",
                speaker="candidate",
                text=request,
                utterance_id=pending.id if pending else None,
                detection_source="final_stt_segments" if defer_closing else "completed_turn",
            )
            if pending:
                self._record(
                    "agent_cancellation",
                    speaker=None,
                    utterance_id=pending.id,
                    cancellation_reason="early_end",
                )
            # Pipecat already broadcasts interruption on UserStartedSpeakingFrame.
            # A second broadcast while processing an STT frame could reset the
            # queue before UserStoppedSpeakingFrame reaches this adapter.
            if self._cancel_audio and not defer_closing:
                await self._cancel_audio()
            if defer_closing:
                return
            if any(self.controller.answers):
                self._start_response()
            else:
                await self._queue_speech(
                    self.controller.prepare_early_feedback("We'll end here. Thanks for practicing.")
                )

    def _start_response(self) -> None:
        if self._stop_latched and self.controller.stage is not Stage.EARLY_CLOSING:
            return
        self._generation += 1
        generation = self._generation
        self._response_task = asyncio.create_task(self._respond(generation))

    def _payload(self) -> dict[str, Any]:
        index = self.controller.current_question_number
        current_answers = list(self.controller.answers[index - 1]) if index else []
        return {
            "current_question_number": self.controller.current_question_number,
            "current_question": self.controller.current_question,
            "greeting_response": self.controller.greeting_response,
            "answers_by_question": [list(turns) for turns in self.controller.answers],
            "current_question_answers": current_answers,
            "current_question_answer": " ".join(current_answers),
            "last_delivered_prompt": (
                self._last_delivered_prompt.text if self._last_delivered_prompt else None
            ),
            "last_delivered_prompt_id": (
                self._last_delivered_prompt.id if self._last_delivered_prompt else None
            ),
            "interrupted_prompt": (
                self.controller.interrupted_utterance.text
                if self.controller.interrupted_utterance
                else None
            ),
            "closing_remarks": list(self.controller.closing_remarks),
            "max_extra_prompts_per_question": {"follow_up_or_nudge": 1, "clarification": 1},
            "remaining_follow_ups": self.controller.remaining_follow_ups,
            "remaining_clarifications": self.controller.remaining_clarifications,
        }

    async def _respond(self, generation: int) -> None:
        try:
            if not self._stop_latched and self.controller.stage is Stage.AWAITING_OPENING_ACK:
                greeting_readiness = _classify_greeting_readiness(self.controller.greeting_response)
                self._record(
                    "greeting_readiness_classified",
                    speaker=None,
                    classification=greeting_readiness.name.lower(),
                )
                self._record(
                    "inference_started", speaker=None, generation=generation, purpose="opening_ack"
                )
                result = await self._infer(
                    "opening_ack",
                    {**self._payload(), "greeting_readiness": greeting_readiness.name.lower()},
                )
                if generation != self._generation or self._closed:
                    self._record("inference_stale_discarded", speaker=None, generation=generation)
                    return
                acknowledgement = result.get("text")
                if not isinstance(acknowledgement, str):
                    raise ValueError("Opening response did not contain text")
                try:
                    utterance = self.controller.prepare_readiness(
                        acknowledgement, greeting_readiness
                    )
                except ValueError:
                    self._record("opening_ack_fallback", speaker=None, error_type="InvalidText")
                    utterance = self.controller.prepare_readiness(
                        _fallback_opening_ack(self.controller.greeting_response), greeting_readiness
                    )
                self._record(
                    "inference_completed",
                    speaker=None,
                    generation=generation,
                    purpose="opening_ack",
                    validated=True,
                )
                await self._queue_speech(utterance)
                return

            if not self._stop_latched and self.controller.stage is Stage.AWAITING_PROPOSAL:
                started_at = time.perf_counter()
                self._record(
                    "inference_started",
                    speaker=None,
                    generation=generation,
                    purpose="decision",
                    answer_number=self._answer_number,
                    llm_first_token_ms=None,
                    llm_first_token_basis="unavailable from structured run_inference",
                )
                result = await self._infer("decision", self._payload())
                if generation != self._generation or self._closed:
                    self._record("inference_stale_discarded", speaker=None, generation=generation)
                    return
                self._record(
                    "inference_completed",
                    speaker=None,
                    generation=generation,
                    purpose="decision",
                    duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
                )
                action = str(result.get("action", "")).upper()
                text = result.get("text", "")
                self._record(
                    "decision_proposed",
                    speaker=None,
                    generation=generation,
                    action=action.lower(),
                    text=text if isinstance(text, str) else None,
                    remaining_follow_ups=self.controller.remaining_follow_ups,
                    remaining_clarifications=self.controller.remaining_clarifications,
                    validated=True,
                )
                if action == "WAIT":
                    self._record(
                        "awaiting_continuation", speaker=None, answer_number=self._answer_number
                    )
                    return
                try:
                    kind = ReplyKind[action]
                    speech = text if kind is not ReplyKind.ADVANCE else None
                    utterance = self.controller.propose(kind, speech)
                except (KeyError, ValueError) as exc:
                    self._record(
                        "proposal_rejected",
                        speaker=None,
                        reason=type(exc).__name__,
                        action=action.lower(),
                    )
                    self._record(
                        "awaiting_continuation", speaker=None, answer_number=self._answer_number
                    )
                    return
                if utterance is not None:
                    await self._queue_speech(utterance)
                    return

            if not self._stop_latched and self.controller.stage is Stage.AWAITING_FEEDBACK:
                started_at = time.perf_counter()
                self._record(
                    "inference_started",
                    speaker=None,
                    generation=generation,
                    purpose="feedback",
                    answer_number=self._answer_number,
                    llm_first_token_ms=None,
                    llm_first_token_basis="unavailable from structured run_inference",
                )
                result = await self._infer("feedback", self._payload())
                if generation != self._generation or self._closed:
                    self._record("inference_stale_discarded", speaker=None, generation=generation)
                    return
                self._record(
                    "inference_completed",
                    speaker=None,
                    generation=generation,
                    purpose="feedback",
                    duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
                    validated=True,
                )
                lines = result.get("lines")
                if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
                    raise ValueError("Feedback response did not contain lines")
                await self._queue_speech(self.controller.prepare_feedback("\n".join(lines)))

            if self.controller.stage is Stage.EARLY_CLOSING:
                self._record(
                    "inference_started",
                    speaker=None,
                    generation=generation,
                    purpose="early_feedback",
                    llm_first_token_ms=None,
                    llm_first_token_basis="unavailable from structured run_inference",
                )
                result = await self._infer("early_feedback", self._payload())
                if generation != self._generation or self._closed:
                    self._record("inference_stale_discarded", speaker=None, generation=generation)
                    return
                lines = result.get("lines")
                if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
                    raise ValueError("Early feedback response did not contain lines")
                self._record(
                    "inference_completed",
                    speaker=None,
                    generation=generation,
                    purpose="early_feedback",
                    validated=True,
                )
                await self._queue_speech(self.controller.prepare_early_feedback("\n".join(lines)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if (
                self.controller.stage is Stage.AWAITING_OPENING_ACK
                and generation == self._generation
            ):
                self._record("opening_ack_fallback", speaker=None, error_type=type(exc).__name__)
                self._record(
                    "response_error",
                    speaker=None,
                    error_type=type(exc).__name__,
                    failure_stage="opening_inference",
                )
                greeting_readiness = _classify_greeting_readiness(self.controller.greeting_response)
                await self._queue_speech(
                    self.controller.prepare_readiness(
                        _fallback_opening_ack(self.controller.greeting_response), greeting_readiness
                    )
                )
            else:
                self._record("response_error", speaker=None, error_type=type(exc).__name__)
                if self.controller.stage is Stage.EARLY_CLOSING and generation == self._generation:
                    await self._queue_speech(
                        self.controller.prepare_early_feedback(
                            "We'll stop here. Thanks for sharing what you have so far."
                        )
                    )

    def tts_started(self, utterance_id: int, context_id: str | None) -> None:
        pending = self.controller.pending
        if pending is None or pending.id != utterance_id or context_id is None:
            return
        self._tts_contexts[context_id] = utterance_id
        self._record(
            "tts_started", speaker=None, utterance_id=utterance_id, tts_context_id=context_id
        )

    def pipeline_error(self, *, processor: str | None, category: str | None) -> None:
        if not self._closed:
            self._record(
                "response_error",
                speaker=None,
                failure_stage="pipeline",
                processor=processor,
                category=category,
                utterance_id=self.controller.pending.id if self.controller.pending else None,
            )

    def tts_first_audio(self, context_id: str | None) -> None:
        utterance_id = self._tts_contexts.get(context_id or "")
        pending = self.controller.pending
        if (
            utterance_id is not None
            and pending is not None
            and pending.id == utterance_id
            and utterance_id not in self._first_audio_ids
        ):
            self._first_audio_ids.add(utterance_id)
            self._record(
                "tts_first_audio_generated",
                speaker=None,
                utterance_id=utterance_id,
                tts_context_id=context_id,
                browser_playback_onset_ms=None,
            )

    def tts_finished(self, context_id: str | None) -> None:
        utterance_id = self._tts_contexts.pop(context_id or "", None)
        if utterance_id is not None:
            self._tts_completed_ids.add(utterance_id)
            self._record(
                "tts_generation_finished",
                speaker=None,
                utterance_id=utterance_id,
                tts_context_id=context_id,
            )

    def bot_started(self) -> None:
        pending = self.controller.pending
        if pending is None or self._closed or self._speaking_id == pending.id:
            return
        if pending.id not in self._first_audio_ids:
            return
        self._speaking_id = pending.id
        self._output_started_ids.add(pending.id)
        elapsed = (
            round((time.perf_counter() - self._last_turn_end) * 1000, 1)
            if self._last_turn_end is not None
            else None
        )
        self._record(
            "agent_output_started",
            speaker="agent",
            utterance_id=pending.id,
            latency_ms=elapsed,
            latency_basis=(
                "user turn stopped → server output started (approximate)"
                if elapsed is not None
                else None
            ),
            browser_playback_onset_ms=None,
        )

    def bot_stopped(self) -> None:
        if self._cancelled_output_stop_debt:
            self._cancelled_output_stop_debt -= 1
            self._record("cancelled_output_stop_ignored", speaker=None)
            return
        pending = self.controller.pending
        if (
            pending is None
            or pending.id != self._speaking_id
            or self._closed
            or pending.id not in self._tts_completed_ids
        ):
            return
        self._speaking_id = None
        if self.controller.delivery_finished(pending.id):
            self._last_delivered_prompt = pending
            self._record(
                "agent_delivery_reported",
                speaker="agent",
                utterance_id=pending.id,
                text=pending.text,
                interrupted=False,
                delivery_basis="Pipecat BotStoppedSpeakingFrame at output transport",
                browser_playback_confirmed=False,
            )
            if self.controller.stage is Stage.COMPLETE:
                self._record("interview_complete", speaker=None)
            elif self.controller.stage is Stage.ENDED_EARLY:
                self._record("interview_ended_early", speaker=None)
            elif self.controller.stage is Stage.AWAITING_QUESTION_START:
                next_question = self.controller.begin_question_one()
                asyncio.create_task(self._queue_speech(next_question))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        pending = self.controller.pending
        if pending is not None:
            self._record(
                "agent_cancellation",
                speaker=None,
                utterance_id=pending.id,
                cancellation_reason="session_disconnected",
            )
        task = self._response_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self.controller.stage is Stage.COMPLETE:
            end_kind = "normal_completion"
        elif self.controller.stage is Stage.ENDED_EARLY:
            end_kind = "spoken_early_end"
        else:
            end_kind = "disconnected_before_completion"
        self._record(
            "session_ended",
            speaker=None,
            interview_stage=self.controller.stage.name.lower(),
            session_end_kind=end_kind,
        )
        self.log.close()
