"""Provider-independent interview policy and speech delivery state."""

from dataclasses import dataclass
from enum import Enum, auto

QUESTIONS = (
    "Tell me about a project you're proud of.",
    "What was the hardest technical problem in it?",
    "How did you know it was working well?",
    "What would you do differently now?",
)
GREETING = "Hey, I'm RehearseAI, your AI interview coach. How's your day going?"
NAME_PROMPT = "What should I call you?"
READINESS_PROMPT = "Ready to jump in?"
READINESS_WAIT = "Just let me know when you're ready."
READINESS_CLARIFICATION = (
    "I didn't catch whether you're ready. Say 'start' when you are, or 'not yet' "
    "if you need a moment."
)
PROJECT_INVITATION = (
    "A coursework project, small script, or another technical task would work too. "
    "Is there one you'd like to use?"
)
GENTLE_PROJECT_INVITATION = (
    "I'm sorry it's feeling that way. We can pause, or use a small script, "
    "coursework project, or another technical task. Would you like to try one?"
)
NO_PROJECT_SCOPE = (
    "This practice needs a technical example to build on. "
    "We can end here, or you can return when one comes to mind."
)

MAX_FOLLOW_UPS_PER_QUESTION = 1
MAX_CLARIFICATIONS_PER_QUESTION = 1
MAX_OPENING_REPEATS = 1


class Stage(Enum):
    NEW = auto()
    DELIVERING = auto()
    AWAITING_GREETING = auto()
    AWAITING_OPENING_ACK = auto()
    AWAITING_NAME = auto()
    AWAITING_OPENING_RETURN = auto()
    AWAITING_DETOUR_RETURN = auto()
    AWAITING_QUESTION_START = auto()
    AWAITING_READINESS = auto()
    AWAITING_ANSWER = auto()
    AWAITING_PROPOSAL = auto()
    AWAITING_NO_PROJECT_EXAMPLE = auto()
    AWAITING_NO_PROJECT_CHOICE = auto()
    AWAITING_FEEDBACK = auto()
    EARLY_CLOSING = auto()
    COMPLETE = auto()
    ENDED_EARLY = auto()


class ReplyKind(Enum):
    FOLLOW_UP = auto()
    NUDGE = auto()
    CLARIFICATION = auto()
    ADVANCE = auto()
    DETOUR = auto()


class Readiness(Enum):
    READY = auto()
    DEFERRED = auto()
    UNCLEAR = auto()


class UtteranceKind(Enum):
    GREETING = auto()
    NAME_PROMPT = auto()
    NAME_ACK = auto()
    OPENING_COACH_REPLY = auto()
    DETOUR_ACK = auto()
    DETOUR_RESUME = auto()
    OPENING_ACK = auto()
    READINESS = auto()
    OPENING_WAIT = auto()
    OPENING_CLARIFICATION = auto()
    QUESTION = auto()
    FOLLOW_UP = auto()
    NUDGE = auto()
    CLARIFICATION = auto()
    REPEAT = auto()
    NO_PROJECT_INVITE = auto()
    NO_PROJECT_SCOPE = auto()
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
        self._preferred_name: str | None = None
        self._name_declined = False
        self._name_detour = False
        self._name_repeats = 0
        self._detour_resume_prompt: Utterance | None = None
        self._detours = [0] * len(QUESTIONS)
        self._opening_return_to_name = False
        self._opening_coach_replies = 0
        self._project_established: bool | None = None
        self._no_project_repeats = 0
        self._opening_repeats = {Stage.AWAITING_GREETING: 0, Stage.AWAITING_READINESS: 0}
        self._readiness_nudges = 0
        self._readiness_clarifications = 0

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
    def preferred_name(self) -> str | None:
        return self._preferred_name

    @property
    def name_declined(self) -> bool:
        return self._name_declined

    @property
    def project_established(self) -> bool | None:
        return self._project_established

    @property
    def remaining_detours(self) -> int:
        return 2 - self._detours[self._question_index] if self._question_index >= 0 else 2

    def remember_name(self, name: str) -> None:
        if not name or len(name) > 40 or not all(part.isalpha() for part in name.split()):
            raise ValueError("Preferred name must be short plain text")
        self._preferred_name = name
        self._name_declined = False

    def decline_name(self) -> None:
        self._name_declined = True

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

    def prepare_readiness(
        self,
        acknowledgement: str,
        greeting_readiness: Readiness = Readiness.UNCLEAR,
        *,
        ask_name: bool = False,
    ) -> Utterance:
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
        if greeting_readiness is Readiness.READY:
            return self._prepare(UtteranceKind.OPENING_ACK, acknowledgement, None)
        if ask_name and not self._preferred_name and not self._name_declined:
            return self._prepare(
                UtteranceKind.NAME_PROMPT, f"{acknowledgement} {NAME_PROMPT}", None
            )
        if greeting_readiness is Readiness.DEFERRED:
            self._readiness_nudges = 1
            return self._prepare(
                UtteranceKind.OPENING_WAIT, f"{acknowledgement} {READINESS_WAIT}", None
            )
        return self._prepare(UtteranceKind.READINESS, f"{acknowledgement} {READINESS_PROMPT}", None)

    def submit_name(self, name: str | None, readiness: Readiness) -> Utterance:
        self._require(Stage.AWAITING_NAME)
        if name:
            self.remember_name(name)
            acknowledgement = f"Nice to meet you, {name}."
        else:
            self._name_declined = True
            acknowledgement = "No problem."
        if self._name_detour:
            return self._prepare(UtteranceKind.DETOUR_ACK, acknowledgement, self._question_index)
        if readiness is Readiness.READY:
            return self._prepare(UtteranceKind.NAME_ACK, acknowledgement, None)
        if readiness is Readiness.DEFERRED:
            self._readiness_nudges = 1
            return self._prepare(
                UtteranceKind.OPENING_WAIT, f"{acknowledgement} {READINESS_WAIT}", None
            )
        return self._prepare(UtteranceKind.READINESS, f"{acknowledgement} {READINESS_PROMPT}", None)

    def answer_opening_question(self, reply: str, name: str | None = None) -> Utterance:
        if self._stage not in {Stage.AWAITING_NAME, Stage.AWAITING_READINESS}:
            raise InvalidTransition("Opening question is no longer pending")
        if not reply.strip() or len(reply) > 120 or "?" in reply or "\n" in reply:
            raise ValueError("Coach reply must be a short statement")
        self._opening_return_to_name = self._stage is Stage.AWAITING_NAME and name is None
        self._opening_coach_replies += 1
        if name:
            self.remember_name(name)
            reply = f"Nice to meet you, {name}. {reply.strip()}"
        return self._prepare(UtteranceKind.OPENING_COACH_REPLY, reply.strip(), None)

    def resume_opening_question(self) -> Utterance:
        self._require(Stage.AWAITING_OPENING_RETURN)
        if self._opening_return_to_name and self._opening_coach_replies < 2:
            return self._prepare(UtteranceKind.NAME_PROMPT, NAME_PROMPT, None)
        return self._prepare(UtteranceKind.READINESS, READINESS_PROMPT, None)

    def start_after_opening_question(self) -> Utterance:
        self._require(Stage.AWAITING_OPENING_RETURN)
        return self._prepare(UtteranceKind.QUESTION, QUESTIONS[0], 0)

    def retry_name_prompt(self) -> Utterance:
        self._require(Stage.AWAITING_NAME)
        self._name_repeats += 1
        if self._name_repeats > 1:
            return self.submit_name(None, Readiness.UNCLEAR)
        return self._prepare(
            UtteranceKind.NAME_PROMPT,
            NAME_PROMPT,
            self._question_index if self._name_detour else None,
        )

    def skip_name_for_start(self) -> Utterance:
        self._require(Stage.AWAITING_NAME)
        self._name_declined = True
        kind = UtteranceKind.DETOUR_ACK if self._name_detour else UtteranceKind.NAME_ACK
        return self._prepare(
            kind, "Okay, let's begin.", self._question_index if self._name_detour else None
        )

    def start_name_detour(self, resume_prompt: Utterance | None = None) -> Utterance:
        if self._stage not in {Stage.AWAITING_ANSWER, Stage.AWAITING_PROPOSAL}:
            raise InvalidTransition(f"Cannot start a detour from {self._stage.name}")
        index = self._question_index
        self._detours[index] = min(self._detours[index] + 1, 2)
        self._name_detour = True
        self._detour_resume_prompt = resume_prompt
        if self._preferred_name:
            return self._prepare(
                UtteranceKind.DETOUR_ACK,
                f"You can call me RehearseAI, and I have you as {self._preferred_name}.",
                index,
            )
        if self._name_declined:
            return self._prepare(
                UtteranceKind.DETOUR_ACK, "No need to share a name. We can carry on.", index
            )
        return self._prepare(
            UtteranceKind.NAME_PROMPT,
            "You're right, I didn't ask. What should I call you?",
            index,
        )

    def start_general_detour(self, text: str, resume_prompt: Utterance | None = None) -> Utterance:
        self._require(Stage.AWAITING_PROPOSAL)
        exhausted = self._detours[self._question_index] >= 2
        self._detours[self._question_index] = min(self._detours[self._question_index] + 1, 2)
        self._detour_resume_prompt = resume_prompt
        if exhausted:
            text = "Let's return to the project question when you're ready."
        return self._prepare(UtteranceKind.DETOUR_ACK, text, self._question_index)

    def restore_answers(self, answers: tuple[str, ...]) -> None:
        self._require(Stage.AWAITING_PROPOSAL)
        self._answers[self._question_index] = list(answers)

    def add_detour_fragment(self, text: str) -> None:
        self._require(Stage.AWAITING_PROPOSAL)
        if fragment := text.strip():
            self._answers[self._question_index].append(fragment)

    def resume_detour(self) -> Utterance | None:
        self._require(Stage.AWAITING_DETOUR_RETURN)
        self._name_detour = False
        prompt = self._detour_resume_prompt
        self._detour_resume_prompt = None
        if prompt and prompt.kind in {
            UtteranceKind.FOLLOW_UP,
            UtteranceKind.NUDGE,
            UtteranceKind.CLARIFICATION,
            UtteranceKind.REPEAT,
        }:
            return self._prepare(UtteranceKind.DETOUR_RESUME, prompt.text, self._question_index)
        if self._answers[self._question_index]:
            self._stage = Stage.AWAITING_PROPOSAL
            return None
        return self._prepare(
            UtteranceKind.DETOUR_RESUME, QUESTIONS[self._question_index], self._question_index
        )

    def begin_question_one(self) -> Utterance:
        self._require(Stage.AWAITING_QUESTION_START)
        return self._prepare(UtteranceKind.QUESTION, QUESTIONS[0], 0)

    def submit_readiness(self, ready: bool) -> Utterance | None:
        self._require(Stage.AWAITING_READINESS)
        if ready:
            return self._prepare(UtteranceKind.QUESTION, QUESTIONS[0], 0)
        if self._readiness_nudges >= 1:
            return None
        self._readiness_nudges += 1
        return self._prepare(UtteranceKind.OPENING_WAIT, READINESS_WAIT, None)

    def clarify_readiness(self) -> Utterance | None:
        self._require(Stage.AWAITING_READINESS)
        if self._readiness_clarifications >= 1:
            return None
        self._readiness_clarifications += 1
        return self._prepare(UtteranceKind.OPENING_CLARIFICATION, READINESS_CLARIFICATION, None)

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

    def request_repeat(self, active_prompt: Utterance | None = None) -> Utterance:
        """Repeat the active prompt, spending only the clarification allowance."""
        if self._stage not in {Stage.AWAITING_ANSWER, Stage.AWAITING_PROPOSAL}:
            raise InvalidTransition("There is no project prompt to repeat")
        index = self._question_index
        if self._clarifications[index] >= MAX_CLARIFICATIONS_PER_QUESTION:
            raise LimitReached("Clarification limit reached for this question")
        self._clarifications[index] += 1
        text = (
            active_prompt.text
            if active_prompt and active_prompt.question_number == index + 1
            else QUESTIONS[index]
        )
        return self._prepare(UtteranceKind.REPEAT, text, index)

    def set_project_status(self, status: str) -> None:
        if status == "technical_example":
            self._project_established = True
        elif status in {"no_example", "nontechnical"} and self._question_index == 0:
            self._project_established = False
        elif (
            status == "unclear" and self._question_index == 0 and self._project_established is None
        ):
            self._project_established = False

    def offer_project_example(self, *, gentle: bool = False) -> Utterance:
        if self._stage not in {Stage.AWAITING_PROPOSAL, Stage.AWAITING_ANSWER}:
            raise InvalidTransition("Project recovery is not pending")
        if self._question_index != 0 or self._project_established is True:
            raise InvalidTransition("Project recovery is not applicable")
        self._project_established = False
        return self._prepare(
            UtteranceKind.NO_PROJECT_INVITE,
            GENTLE_PROJECT_INVITATION if gentle else PROJECT_INVITATION,
            0,
        )

    def explain_no_project_scope(self) -> Utterance:
        self._require(Stage.AWAITING_NO_PROJECT_EXAMPLE)
        return self._prepare(UtteranceKind.NO_PROJECT_SCOPE, NO_PROJECT_SCOPE, 0)

    def repeat_no_project_prompt(self) -> Utterance:
        if self._stage not in {
            Stage.AWAITING_NO_PROJECT_EXAMPLE,
            Stage.AWAITING_NO_PROJECT_CHOICE,
        }:
            raise InvalidTransition("No project recovery prompt is active")
        if self._no_project_repeats >= 1:
            raise LimitReached("No-project repeat limit reached")
        self._no_project_repeats += 1
        kind = (
            UtteranceKind.NO_PROJECT_INVITE
            if self._stage is Stage.AWAITING_NO_PROJECT_EXAMPLE
            else UtteranceKind.NO_PROJECT_SCOPE
        )
        text = PROJECT_INVITATION if kind is UtteranceKind.NO_PROJECT_INVITE else NO_PROJECT_SCOPE
        return self._prepare(kind, text, 0)

    def accept_project_example(self, text: str) -> None:
        if self._stage not in {
            Stage.AWAITING_NO_PROJECT_EXAMPLE,
            Stage.AWAITING_NO_PROJECT_CHOICE,
        }:
            raise InvalidTransition("No project example was requested")
        if not text.strip():
            raise ValueError("Project example cannot be empty")
        self._project_established = True
        self._answers[0].append(text.strip())
        self._stage = Stage.AWAITING_PROPOSAL

    def propose(self, kind: ReplyKind, text: str | None = None) -> Utterance | None:
        """Accept a bounded LLM proposal; Python selects the next question."""
        self._require(Stage.AWAITING_PROPOSAL)
        if not isinstance(kind, ReplyKind):
            raise ValueError("Unknown reply kind")
        index = self._question_index
        if kind is ReplyKind.ADVANCE:
            if text is not None:
                raise ValueError("ADVANCE cannot supply speech text")
            if index == 0 and self._project_established is False:
                raise InvalidTransition("A technical example is needed before dependent questions")
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
        if self._question_index == 0 and self._project_established is not True:
            raise InvalidTransition("Cannot skip to a dependent question without a project")
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
        elif utterance.kind in {UtteranceKind.OPENING_ACK, UtteranceKind.NAME_ACK}:
            self._stage = Stage.AWAITING_QUESTION_START
        elif utterance.kind is UtteranceKind.NAME_PROMPT:
            self._stage = Stage.AWAITING_NAME
        elif utterance.kind is UtteranceKind.OPENING_COACH_REPLY:
            self._stage = Stage.AWAITING_OPENING_RETURN
        elif utterance.kind is UtteranceKind.DETOUR_ACK:
            self._stage = Stage.AWAITING_DETOUR_RETURN
        elif utterance.kind is UtteranceKind.NO_PROJECT_INVITE:
            self._stage = Stage.AWAITING_NO_PROJECT_EXAMPLE
        elif utterance.kind is UtteranceKind.NO_PROJECT_SCOPE:
            self._stage = Stage.AWAITING_NO_PROJECT_CHOICE
        elif utterance.kind in {
            UtteranceKind.READINESS,
            UtteranceKind.OPENING_WAIT,
            UtteranceKind.OPENING_CLARIFICATION,
        }:
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
        elif interrupted.kind is UtteranceKind.OPENING_ACK:
            self._stage = Stage.AWAITING_QUESTION_START
        elif interrupted.kind is UtteranceKind.NAME_PROMPT:
            self._stage = Stage.AWAITING_NAME
        elif interrupted.kind is UtteranceKind.OPENING_COACH_REPLY:
            self._stage = Stage.AWAITING_OPENING_RETURN
        elif interrupted.kind is UtteranceKind.NAME_ACK:
            self._stage = Stage.AWAITING_QUESTION_START
        elif interrupted.kind is UtteranceKind.DETOUR_ACK:
            self._stage = Stage.AWAITING_DETOUR_RETURN
        elif interrupted.kind is UtteranceKind.NO_PROJECT_INVITE:
            self._stage = Stage.AWAITING_NO_PROJECT_EXAMPLE
        elif interrupted.kind is UtteranceKind.NO_PROJECT_SCOPE:
            self._stage = Stage.AWAITING_NO_PROJECT_CHOICE
        elif interrupted.kind in {
            UtteranceKind.READINESS,
            UtteranceKind.OPENING_WAIT,
            UtteranceKind.OPENING_CLARIFICATION,
        }:
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
