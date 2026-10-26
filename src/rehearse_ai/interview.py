"""Provider-independent interview policy and speech delivery state."""

from dataclasses import dataclass
from enum import Enum, auto

QUESTIONS = (
    "Tell me about a project you're proud of.",
    "What was the hardest technical problem in it?",
    "How did you know it was working well?",
    "What would you do differently now?",
)
GREETING = "Hey, how are you doing today?"
READINESS_PROMPT = (
    "We'll go through four questions about your project, then I'll give you short feedback. "
    "Are you ready to begin?"
)
READINESS_WAIT = "No rush. Tell me when you're ready to begin."

MAX_FOLLOW_UPS_PER_QUESTION = 1
MAX_CLARIFICATIONS_PER_QUESTION = 1
MAX_OPENING_REPEATS = 1


class Stage(Enum):
    NEW = auto()
    DELIVERING = auto()
    AWAITING_GREETING = auto()
    AWAITING_OPENING_ACK = auto()
    AWAITING_READINESS = auto()
    AWAITING_ANSWER = auto()
    AWAITING_PROPOSAL = auto()
    AWAITING_FEEDBACK = auto()
    EARLY_CLOSING = auto()
    COMPLETE = auto()
    ENDED_EARLY = auto()


class ReplyKind(Enum):
    FOLLOW_UP = auto()
    NUDGE = auto()
    CLARIFICATION = auto()
    ADVANCE = auto()


class UtteranceKind(Enum):
    GREETING = auto()
    READINESS = auto()
    OPENING_WAIT = auto()
    QUESTION = auto()
    FOLLOW_UP = auto()
    NUDGE = auto()
    CLARIFICATION = auto()
    REPEAT = auto()
    FEEDBACK = auto()
    EARLY_FEEDBACK = auto()


@dataclass(frozen=True)
class Utterance:
    id: int
    kind: UtteranceKind
    text: str
    question_number: int | None


class InvalidTransition(ValueError):
    """An event is not valid in the current interview stage."""


class LimitReached(InvalidTransition):
    """The current question's follow-up or clarification budget is spent."""


class InterviewController:
    """One session's question order, reply budgets, and delivery accounting.

    IDs are scoped to this controller. A delivery acknowledgment means the
    transport reports playback completed, not that a human heard every word.
    """

    def __init__(self) -> None:
        self._stage = Stage.NEW
        self._question_index = -1
        self._next_utterance_id = 1
        self._pending: Utterance | None = None
        self._interrupted: Utterance | None = None
        self._answers: list[list[str]] = [[] for _ in QUESTIONS]
        self._closing_remarks: list[str] = []
        self._follow_ups = [0] * len(QUESTIONS)
        self._clarifications = [0] * len(QUESTIONS)
        self._greeting_response = ""
        self._opening_repeats = {Stage.AWAITING_GREETING: 0, Stage.AWAITING_READINESS: 0}
        self._readiness_nudges = 0

    @property
    def stage(self) -> Stage:
        return self._stage

    @property
    def current_question_number(self) -> int | None:
        return self._question_index + 1 if self._question_index >= 0 else None

    @property
    def current_question(self) -> str | None:
        return QUESTIONS[self._question_index] if self._question_index >= 0 else None

    @property
    def pending(self) -> Utterance | None:
        return self._pending

    @property
    def interrupted_utterance(self) -> Utterance | None:
        return self._interrupted

    @property
    def answers(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(turns) for turns in self._answers)

    @property
    def closing_remarks(self) -> tuple[str, ...]:
        return tuple(self._closing_remarks)

    @property
    def greeting_response(self) -> str:
        return self._greeting_response

    @property
    def remaining_follow_ups(self) -> int:
        if self._question_index < 0:
            return MAX_FOLLOW_UPS_PER_QUESTION
        return MAX_FOLLOW_UPS_PER_QUESTION - self._follow_ups[self._question_index]

    @property
    def remaining_clarifications(self) -> int:
        if self._question_index < 0:
            return MAX_CLARIFICATIONS_PER_QUESTION
        return MAX_CLARIFICATIONS_PER_QUESTION - self._clarifications[self._question_index]

    def start(self) -> Utterance:
        self._require(Stage.NEW)
        return self._prepare(UtteranceKind.GREETING, GREETING, None)

    def submit_greeting(self, text: str) -> Utterance | None:
        self._require(Stage.AWAITING_GREETING)
        response = text.strip()
        if not response:
            return self.request_opening_repeat()
        self._greeting_response = response
        self._stage = Stage.AWAITING_OPENING_ACK
        return None

    def add_greeting_continuation(self, text: str) -> None:
        self._require(Stage.AWAITING_OPENING_ACK)
        if continuation := text.strip():
            self._greeting_response = f"{self._greeting_response} {continuation}"

    def prepare_readiness(self, acknowledgement: str) -> Utterance:
        self._require(Stage.AWAITING_OPENING_ACK)
        acknowledgement = acknowledgement.strip()
        body = acknowledgement.rstrip(".! ")
        if (
            not body
            or len(acknowledgement) > 120
            or "\n" in acknowledgement
            or "?" in acknowledgement
            or "." in body
            or "!" in body
        ):
            raise ValueError("Opening acknowledgement must be one short spoken line")
        if acknowledgement[-1] not in ".!?":
            acknowledgement += "."
        return self._prepare(UtteranceKind.READINESS, f"{acknowledgement} {READINESS_PROMPT}", None)

    def submit_readiness(self, ready: bool) -> Utterance | None:
        self._require(Stage.AWAITING_READINESS)
        if ready:
            return self._prepare(UtteranceKind.QUESTION, QUESTIONS[0], 0)
        if self._readiness_nudges >= 1:
            return None
        self._readiness_nudges += 1
        return self._prepare(UtteranceKind.OPENING_WAIT, READINESS_WAIT, None)

    def request_opening_repeat(self) -> Utterance:
        phase = (
            Stage.AWAITING_GREETING if self._stage is Stage.AWAITING_OPENING_ACK else self._stage
        )
        if phase not in self._opening_repeats:
            raise InvalidTransition("There is no opening prompt to repeat")
        if self._opening_repeats[phase] >= MAX_OPENING_REPEATS:
            raise LimitReached("Opening repeat limit reached")
        self._opening_repeats[phase] += 1
        if phase is Stage.AWAITING_GREETING:
            self._greeting_response = ""
            return self._prepare(UtteranceKind.GREETING, GREETING, None)
        return self._prepare(UtteranceKind.READINESS, READINESS_PROMPT, None)

    def submit_answer(self, text: str) -> Utterance | None:
        """Record a candidate turn; an empty turn requests one bounded repeat."""
        self._require(Stage.AWAITING_ANSWER)
        answer = text.strip()
        if not answer:
            return self.request_repeat()
        self._answers[self._question_index].append(answer)
        self._stage = Stage.AWAITING_PROPOSAL
        return None

    def request_repeat(self) -> Utterance:
        """Repeat the current main question, spending its clarification budget."""
        self._require(Stage.AWAITING_ANSWER)
        index = self._question_index
        if self._clarifications[index] >= MAX_CLARIFICATIONS_PER_QUESTION:
            raise LimitReached("Clarification limit reached for this question")
        self._clarifications[index] += 1
        return self._prepare(UtteranceKind.REPEAT, QUESTIONS[index], index)

    def propose(self, kind: ReplyKind, text: str | None = None) -> Utterance | None:
        """Accept a bounded LLM proposal; Python selects the next question."""
        self._require(Stage.AWAITING_PROPOSAL)
        if not isinstance(kind, ReplyKind):
            raise ValueError("Unknown reply kind")
        index = self._question_index
        if kind is ReplyKind.ADVANCE:
            if text is not None:
                raise ValueError("ADVANCE cannot supply speech text")
            if index == len(QUESTIONS) - 1:
                self._stage = Stage.AWAITING_FEEDBACK
                return None
            return self._prepare(UtteranceKind.QUESTION, QUESTIONS[index + 1], index + 1)

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Spoken proposal text must be nonempty")
        if kind is ReplyKind.CLARIFICATION:
            if self._clarifications[index] >= MAX_CLARIFICATIONS_PER_QUESTION:
                raise LimitReached("Clarification limit reached for this question")
            self._clarifications[index] += 1
            utterance_kind = UtteranceKind.CLARIFICATION
        else:
            if self._follow_ups[index] >= MAX_FOLLOW_UPS_PER_QUESTION:
                raise LimitReached("Follow-up limit reached for this question")
            self._follow_ups[index] += 1
            utterance_kind = (
                UtteranceKind.FOLLOW_UP if kind is ReplyKind.FOLLOW_UP else UtteranceKind.NUDGE
            )
        return self._prepare(utterance_kind, text.strip(), index)

    def prepare_feedback(self, text: str) -> Utterance:
        """Queue actual session feedback supplied by the future LLM adapter."""
        self._require(Stage.AWAITING_FEEDBACK)
        lines = text.splitlines()
        if not 2 <= len(lines) <= 3 or any(not line.strip() for line in lines):
            raise ValueError("Closing feedback must have 2 or 3 nonempty lines")
        return self._prepare(UtteranceKind.FEEDBACK, text, None)

    def end_early(self) -> bool:
        """Stop all question progression, including an undelivered question."""
        if self._stage in {Stage.COMPLETE, Stage.ENDED_EARLY, Stage.EARLY_CLOSING}:
            return False
        self._pending = None
        self._interrupted = None
        self._stage = Stage.EARLY_CLOSING
        return True

    def prepare_early_feedback(self, text: str) -> Utterance:
        self._require(Stage.EARLY_CLOSING)
        lines = text.splitlines()
        if not 1 <= len(lines) <= 2 or any(not line.strip() for line in lines):
            raise ValueError("Early closing must have 1 or 2 nonempty lines")
        return self._prepare(UtteranceKind.EARLY_FEEDBACK, text, None)

    def skip_current(self) -> Utterance | None:
        """Advance on an explicit skip without recording a fabricated answer."""
        self._require(Stage.AWAITING_ANSWER)
        if self._question_index == len(QUESTIONS) - 1:
            self._stage = Stage.AWAITING_FEEDBACK
            return None
        return self._prepare(
            UtteranceKind.QUESTION, QUESTIONS[self._question_index + 1], self._question_index + 1
        )

    def skip_interrupted(self) -> Utterance | None:
        """Apply an explicit skip during interrupted question speech."""
        self._require(Stage.DELIVERING)
        interrupted = self._interrupted
        if interrupted is None or interrupted.question_number is None:
            raise InvalidTransition("There is no interrupted question to skip")
        if interrupted.kind is UtteranceKind.QUESTION:
            self._question_index = interrupted.question_number - 1
        self._interrupted = None
        self._stage = Stage.AWAITING_ANSWER
        return self.skip_current()

    def resolve_unheard_question(self, text: str) -> Utterance:
        """Route barge-in before audio to the previously delivered question."""
        self._require(Stage.DELIVERING)
        interrupted = self._interrupted
        if interrupted is None or interrupted.kind is not UtteranceKind.QUESTION:
            raise InvalidTransition("There is no interrupted main question")
        self._interrupted = None
        if self._question_index < 0:
            return self._prepare(UtteranceKind.QUESTION, interrupted.text, 0)
        answer = text.strip()
        if answer:
            previous = self._answers[self._question_index]
            if previous:
                previous[-1] = f"{previous[-1]} {answer}"
            else:
                previous.append(answer)
        return self._prepare(
            UtteranceKind.QUESTION, interrupted.text, interrupted.question_number - 1
        )

    def resolve_unheard_follow_up(self, text: str) -> Utterance:
        """Preserve the earlier answer; an unheard extra prompt cannot consume a reply."""
        self._require(Stage.DELIVERING)
        interrupted = self._interrupted
        if interrupted is None or interrupted.kind not in {
            UtteranceKind.FOLLOW_UP,
            UtteranceKind.NUDGE,
            UtteranceKind.CLARIFICATION,
            UtteranceKind.REPEAT,
        }:
            raise InvalidTransition("There is no interrupted extra prompt")
        if text.strip() and self._answers[self._question_index]:
            answers = self._answers[self._question_index]
            answers[-1] = f"{answers[-1]} {text.strip()}"
        return self.retry_interrupted()

    def delivery_finished(self, utterance_id: int) -> bool:
        """Apply only the current pending delivery; ignore duplicate or stale IDs."""
        utterance = self._pending
        if utterance is None or utterance.id != utterance_id:
            return False
        self._pending = None
        if utterance.kind is UtteranceKind.GREETING:
            self._stage = Stage.AWAITING_GREETING
        elif utterance.kind in {UtteranceKind.READINESS, UtteranceKind.OPENING_WAIT}:
            self._stage = Stage.AWAITING_READINESS
        elif utterance.kind is UtteranceKind.QUESTION:
            if utterance.question_number is None:
                raise RuntimeError("Question utterance has no question number")
            self._question_index = utterance.question_number - 1
            self._stage = Stage.AWAITING_ANSWER
        elif utterance.kind is UtteranceKind.FEEDBACK:
            self._stage = Stage.COMPLETE
        elif utterance.kind is UtteranceKind.EARLY_FEEDBACK:
            self._stage = Stage.ENDED_EARLY
        else:
            self._stage = Stage.AWAITING_ANSWER
        return True

    def interrupt(self, utterance_id: int) -> bool:
        """Keep interrupted speech recoverable and invalidate its old ID."""
        utterance = self._pending
        if utterance is None or utterance.id != utterance_id:
            return False
        self._pending = None
        self._interrupted = utterance
        return True

    def retry_interrupted(self) -> Utterance:
        """Requeue interrupted text with a fresh ID after barge-in has settled."""
        self._require(Stage.DELIVERING)
        utterance = self._interrupted
        if utterance is None:
            raise InvalidTransition("There is no interrupted utterance to retry")
        self._interrupted = None
        return self._prepare(utterance.kind, utterance.text, self._index_for(utterance))

    def accept_barge_in_answer(self, text: str) -> bool:
        """Use a substantive answer to resolve interrupted speech without replaying it.

        An answer to a partly heard main question explicitly commits that
        question. Empty transcription leaves the utterance recoverable.
        """
        self._require(Stage.DELIVERING)
        interrupted = self._interrupted
        if interrupted is None:
            raise InvalidTransition("There is no interrupted utterance")
        answer = text.strip()
        if not answer:
            return False
        if interrupted.kind is UtteranceKind.GREETING:
            self._greeting_response = answer
            self._stage = Stage.AWAITING_OPENING_ACK
        elif interrupted.kind in {UtteranceKind.READINESS, UtteranceKind.OPENING_WAIT}:
            self._stage = Stage.AWAITING_READINESS
        elif interrupted.kind is UtteranceKind.FEEDBACK:
            self._closing_remarks.append(answer)
            self._stage = Stage.AWAITING_FEEDBACK
        else:
            if interrupted.kind is UtteranceKind.QUESTION:
                if interrupted.question_number is None:
                    raise RuntimeError("Question utterance has no question number")
                self._question_index = interrupted.question_number - 1
            self._answers[self._question_index].append(answer)
            self._stage = Stage.AWAITING_PROPOSAL
        self._interrupted = None
        return True

    def add_answer_continuation(self, text: str) -> None:
        """Extend the latest answer when speech resumes during LLM planning."""
        self._require(Stage.AWAITING_PROPOSAL)
        continuation = text.strip()
        if continuation:
            latest = self._answers[self._question_index]
            latest[-1] = f"{latest[-1]} {continuation}"

    def add_closing_remarks(self, text: str) -> None:
        """Retain speech that arrives while closing feedback is being planned."""
        self._require(Stage.AWAITING_FEEDBACK)
        if remark := text.strip():
            self._closing_remarks.append(remark)

    def _index_for(self, utterance: Utterance) -> int | None:
        return utterance.question_number - 1 if utterance.question_number is not None else None

    def _prepare(self, kind: UtteranceKind, text: str, index: int | None) -> Utterance:
        utterance = Utterance(
            id=self._next_utterance_id,
            kind=kind,
            text=text,
            question_number=index + 1 if index is not None else None,
        )
        self._next_utterance_id += 1
        self._pending = utterance
        self._stage = Stage.DELIVERING
        return utterance

    def _require(self, stage: Stage) -> None:
        if self._stage is not stage:
            raise InvalidTransition(f"Expected {stage.name}; current stage is {self._stage.name}")
