"""Connect one interview controller to speech, inference, and session events."""

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from rehearse_ai.interview import (
    InterviewController,
    LimitReached,
    ReplyKind,
    Stage,
    Utterance,
)
from rehearse_ai.session_log import SessionLog

Infer = Callable[[str, dict[str, Any]], Awaitable[Mapping[str, Any]]]
Speak = Callable[[Utterance], Awaitable[None]]
Publish = Callable[[dict[str, Any]], None]

_REPEAT = re.compile(
    r"^(?:please\s+)?(?:can you\s+|could you\s+|would you\s+)?"
    r"(?:repeat(?: the (?:question|last question))?|say (?:that|the question) again)\??[.!]?$",
    re.IGNORECASE,
)


class InterviewSession:
    def __init__(
        self,
        log: SessionLog,
        infer: Infer,
        speak: Speak,
        publish: Publish,
        *,
        turn_pause_secs: float = 1.1,
    ) -> None:
        self.controller = InterviewController()
        self.log = log
        self._infer = infer
        self._speak = speak
        self._publish = publish
        self._response_task: asyncio.Task[None] | None = None
        self._generation = 0
        self._speaking_id: int | None = None
        self._active_prompt_id: int | None = None
        self._last_turn_end: float | None = None
        self._turn_started_at: float | None = None
        self._turn_number = 0
        self._answer_number = 0
        self._turn_pause_secs = turn_pause_secs
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

    async def start(self) -> None:
        if self._closed or self.controller.stage is not Stage.NEW:
            return
        self._record("session_started", speaker=None)
        await self._queue_speech(self.controller.start())

    async def _queue_speech(self, utterance: Utterance) -> None:
        self._active_prompt_id = utterance.id
        self._record(
            "agent_prepared",
            speaker="agent",
            utterance_id=utterance.id,
            text=utterance.text,
            interrupted=False,
            question_number=utterance.question_number,
            utterance_kind=utterance.kind.name.lower(),
        )
        await self._speak(utterance)

    async def user_started(self) -> None:
        if self._closed:
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
            self._speaking_id = None
            self._record(
                "agent_interrupted",
                speaker="agent",
                utterance_id=pending.id,
                text=pending.text,
                interrupted=True,
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
        if self._closed or self.controller.stage is Stage.COMPLETE:
            return
        answer = text.strip()
        self._last_turn_end = time.perf_counter()
        continuing = self.controller.stage in {Stage.AWAITING_PROPOSAL, Stage.AWAITING_FEEDBACK}
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
        )
        self._turn_started_at = None

        interrupted = self.controller.interrupted_utterance
        if interrupted is not None:
            if not answer or _REPEAT.fullmatch(answer):
                await self._queue_speech(self.controller.retry_interrupted())
                return
            self.controller.accept_barge_in_answer(answer)
            self._record("barge_in_answer_accepted", speaker="candidate", text=answer)
        elif self.controller.stage is Stage.AWAITING_ANSWER:
            try:
                utterance = (
                    self.controller.request_repeat()
                    if _REPEAT.fullmatch(answer)
                    else self.controller.submit_answer(answer)
                )
            except LimitReached:
                self._record("clarification_limit", speaker=None)
                return
            if utterance is not None:
                await self._queue_speech(utterance)
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

    def _start_response(self) -> None:
        self._generation += 1
        generation = self._generation
        self._response_task = asyncio.create_task(self._respond(generation))

    def _payload(self) -> dict[str, Any]:
        return {
            "current_question_number": self.controller.current_question_number,
            "current_question": self.controller.current_question,
            "answers_by_question": [list(turns) for turns in self.controller.answers],
            "closing_remarks": list(self.controller.closing_remarks),
            "max_extra_prompts_per_question": {"follow_up_or_nudge": 1, "clarification": 1},
            "remaining_follow_ups": self.controller.remaining_follow_ups,
            "remaining_clarifications": self.controller.remaining_clarifications,
        }

    async def _respond(self, generation: int) -> None:
        try:
            if self.controller.stage is Stage.AWAITING_PROPOSAL:
                started_at = time.perf_counter()
                self._record(
                    "inference_started",
                    speaker=None,
                    generation=generation,
                    purpose="decision",
                    answer_number=self._answer_number,
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

            if self.controller.stage is Stage.AWAITING_FEEDBACK:
                started_at = time.perf_counter()
                self._record(
                    "inference_started",
                    speaker=None,
                    generation=generation,
                    purpose="feedback",
                    answer_number=self._answer_number,
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
                )
                lines = result.get("lines")
                if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
                    raise ValueError("Feedback response did not contain lines")
                await self._queue_speech(self.controller.prepare_feedback("\n".join(lines)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record("response_error", speaker=None, error_type=type(exc).__name__)

    def bot_started(self) -> None:
        pending = self.controller.pending
        if pending is None or self._closed or self._speaking_id == pending.id:
            return
        self._speaking_id = pending.id
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
        )

    def bot_stopped(self) -> None:
        pending = self.controller.pending
        if pending is None or pending.id != self._speaking_id or self._closed:
            return
        self._speaking_id = None
        if self.controller.delivery_finished(pending.id):
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

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        task = self._response_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._record("session_ended", speaker=None)
        self.log.close()
