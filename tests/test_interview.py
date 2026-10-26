import pytest

from rehearse_ai.interview import (
    QUESTIONS,
    InterviewController,
    InvalidTransition,
    LimitReached,
    ReplyKind,
    Stage,
    Utterance,
    UtteranceKind,
)


def deliver(controller: InterviewController, utterance: Utterance) -> None:
    assert controller.pending == utterance
    assert controller.delivery_finished(utterance.id)


def start_and_answer(
    controller: InterviewController, answer: str = "A real project answer"
) -> None:
    deliver(controller, controller.start())
    assert controller.submit_answer(answer) is None


def reach_feedback(controller: InterviewController) -> None:
    question = controller.start()
    for index, expected in enumerate(QUESTIONS):
        assert question.text == expected
        assert question.question_number == index + 1
        deliver(controller, question)
        controller.submit_answer(f"Answer {index + 1}")
        next_utterance = controller.propose(ReplyKind.ADVANCE)
        if index < len(QUESTIONS) - 1:
            assert next_utterance is not None
            question = next_utterance
        else:
            assert next_utterance is None
    assert controller.stage is Stage.AWAITING_FEEDBACK


def test_four_questions_and_closing_feedback_require_delivery() -> None:
    controller = InterviewController()
    reach_feedback(controller)
    assert controller.answers == (("Answer 1",), ("Answer 2",), ("Answer 3",), ("Answer 4",))

    feedback = controller.prepare_feedback(
        "You explained the design clearly.\nNext time, quantify the result."
    )
    assert feedback.kind is UtteranceKind.FEEDBACK
    assert controller.stage is Stage.DELIVERING
    deliver(controller, feedback)
    assert controller.stage is Stage.COMPLETE


def test_follow_up_and_nudge_share_one_slot() -> None:
    controller = InterviewController()
    start_and_answer(controller, "I built a compiler")
    follow_up = controller.propose(ReplyKind.FOLLOW_UP, "What trade-off mattered most?")
    assert follow_up.kind is UtteranceKind.FOLLOW_UP
    assert controller.current_question_number == 1
    assert controller.answers[0] == ("I built a compiler",)
    deliver(controller, follow_up)
    controller.submit_answer("Compile time mattered most")

    with pytest.raises(LimitReached):
        controller.propose(ReplyKind.NUDGE, "Could you expand?")
    next_question = controller.propose(ReplyKind.ADVANCE)
    assert next_question.text == QUESTIONS[1]
    assert controller.current_question_number == 1


def test_clarification_and_repeat_share_one_slot() -> None:
    controller = InterviewController()
    start_and_answer(controller)
    clarification = controller.propose(ReplyKind.CLARIFICATION, "Which part did you own?")
    deliver(controller, clarification)
    controller.submit_answer("The data model")
    with pytest.raises(LimitReached):
        controller.propose(ReplyKind.CLARIFICATION, "Can you clarify again?")
    assert controller.propose(ReplyKind.ADVANCE).text == QUESTIONS[1]

    another = InterviewController()
    deliver(another, another.start())
    repeated = another.request_repeat()
    assert repeated.text == QUESTIONS[0]
    deliver(another, repeated)
    another.submit_answer("I owned the design")
    with pytest.raises(LimitReached):
        another.propose(ReplyKind.CLARIFICATION, "Which part?")


def test_empty_answer_and_repeat_request_are_bounded() -> None:
    controller = InterviewController()
    deliver(controller, controller.start())
    repeated = controller.submit_answer("  ")
    assert repeated.kind is UtteranceKind.REPEAT
    assert repeated.text == QUESTIONS[0]
    assert controller.answers[0] == ()
    deliver(controller, repeated)
    with pytest.raises(LimitReached):
        controller.request_repeat()
    with pytest.raises(LimitReached):
        controller.submit_answer("")
    assert controller.stage is Stage.AWAITING_ANSWER
    controller.submit_answer("A project I completed")
    assert controller.answers[0] == ("A project I completed",)


def test_interrupted_main_question_needs_new_delivery() -> None:
    controller = InterviewController()
    original = controller.start()
    assert controller.current_question is None
    assert controller.interrupt(original.id)
    assert controller.pending is None
    assert controller.interrupted_utterance == original
    assert not controller.delivery_finished(original.id)
    with pytest.raises(InvalidTransition):
        controller.submit_answer("I started answering")

    retry = controller.retry_interrupted()
    assert retry.id != original.id
    assert retry.text == original.text
    assert not controller.delivery_finished(original.id)
    deliver(controller, retry)
    assert controller.current_question == QUESTIONS[0]


def test_interrupted_follow_up_preserves_slot_and_answer() -> None:
    controller = InterviewController()
    start_and_answer(controller, "I built a compiler")
    original = controller.propose(ReplyKind.NUDGE, "What did you personally build?")
    assert controller.interrupt(original.id)
    retry = controller.retry_interrupted()
    assert retry.kind is UtteranceKind.NUDGE
    deliver(controller, retry)
    controller.submit_answer("The parser")
    assert controller.answers[0] == ("I built a compiler", "The parser")
    with pytest.raises(LimitReached):
        controller.propose(ReplyKind.FOLLOW_UP, "Why that parser?")


def test_interrupted_transition_does_not_skip_next_question() -> None:
    controller = InterviewController()
    start_and_answer(controller)
    next_question = controller.propose(ReplyKind.ADVANCE)
    assert controller.current_question_number == 1
    assert controller.interrupt(next_question.id)
    assert controller.current_question_number == 1
    retry = controller.retry_interrupted()
    assert retry.text == QUESTIONS[1]
    assert not controller.delivery_finished(next_question.id)
    deliver(controller, retry)
    assert controller.current_question_number == 2


def test_interrupted_closing_response_does_not_complete() -> None:
    controller = InterviewController()
    reach_feedback(controller)
    feedback = controller.prepare_feedback(
        "Strong explanation.\nGive a clearer impact metric next time."
    )
    assert controller.interrupt(feedback.id)
    assert controller.stage is Stage.DELIVERING
    assert not controller.delivery_finished(feedback.id)
    retry = controller.retry_interrupted()
    assert retry.kind is UtteranceKind.FEEDBACK
    deliver(controller, retry)
    assert controller.stage is Stage.COMPLETE


def test_substantive_barge_in_resolves_interrupted_main_question() -> None:
    controller = InterviewController()
    first = controller.start()
    assert controller.interrupt(first.id)
    assert controller.accept_barge_in_answer("I built a parser")
    assert controller.current_question_number == 1
    assert controller.answers[0] == ("I built a parser",)
    assert controller.stage is Stage.AWAITING_PROPOSAL
    assert not controller.delivery_finished(first.id)


def test_interrupted_follow_up_accepts_answer_without_spending_another_slot() -> None:
    controller = InterviewController()
    start_and_answer(controller)
    follow_up = controller.propose(ReplyKind.FOLLOW_UP, "What was your role?")
    assert controller.interrupt(follow_up.id)
    assert controller.accept_barge_in_answer("I designed the parser")
    assert controller.answers[0][-1] == "I designed the parser"
    with pytest.raises(LimitReached):
        controller.propose(ReplyKind.NUDGE, "Say more")


def test_interrupted_feedback_can_be_replanned_but_not_marked_complete() -> None:
    controller = InterviewController()
    reach_feedback(controller)
    feedback = controller.prepare_feedback("Clear story.\nAdd a result next time.")
    assert controller.interrupt(feedback.id)
    assert controller.accept_barge_in_answer("Thank you")
    assert controller.closing_remarks == ("Thank you",)
    assert controller.stage is Stage.AWAITING_FEEDBACK
    assert not controller.delivery_finished(feedback.id)


def test_duplicate_and_stale_delivery_and_interruption_events_are_ignored() -> None:
    controller = InterviewController()
    first = controller.start()
    assert not controller.delivery_finished(first.id + 100)
    deliver(controller, first)
    assert not controller.delivery_finished(first.id)
    assert not controller.interrupt(first.id)
    controller.submit_answer("A project")
    second = controller.propose(ReplyKind.ADVANCE)
    assert not controller.delivery_finished(first.id)
    assert controller.pending == second
    assert controller.interrupt(second.id)
    assert not controller.interrupt(second.id)
    assert not controller.delivery_finished(second.id)


def test_completed_interview_rejects_progression() -> None:
    controller = InterviewController()
    reach_feedback(controller)
    deliver(controller, controller.prepare_feedback("Good depth.\nExplain the trade-offs sooner."))
    with pytest.raises(InvalidTransition):
        controller.start()
    with pytest.raises(InvalidTransition):
        controller.submit_answer("More")
    with pytest.raises(InvalidTransition):
        controller.request_repeat()
    with pytest.raises(InvalidTransition):
        controller.propose(ReplyKind.ADVANCE)
    with pytest.raises(InvalidTransition):
        controller.prepare_feedback("More.\nFeedback.")
    with pytest.raises(InvalidTransition):
        controller.retry_interrupted()
    assert not controller.delivery_finished(999)
    assert not controller.interrupt(999)


def test_invalid_proposals_do_not_change_state() -> None:
    controller = InterviewController()
    start_and_answer(controller)
    with pytest.raises(ValueError):
        controller.propose(ReplyKind.FOLLOW_UP, " ")
    with pytest.raises(ValueError):
        controller.propose(ReplyKind.ADVANCE, "Skip to question four")
    with pytest.raises(ValueError):
        controller.propose("ADVANCE")
    assert controller.stage is Stage.AWAITING_PROPOSAL
    assert controller.pending is None
    assert controller.propose(ReplyKind.ADVANCE).text == QUESTIONS[1]


def test_early_end_is_distinct_and_blocks_further_questions() -> None:
    controller = InterviewController()
    first = controller.start()
    assert controller.end_early()
    assert not controller.delivery_finished(first.id)
    closing = controller.prepare_early_feedback("We'll stop here. Thanks for practicing.")
    deliver(controller, closing)
    assert controller.stage is Stage.ENDED_EARLY
    assert not controller.end_early()
    with pytest.raises(InvalidTransition):
        controller.submit_answer("A late answer")
    with pytest.raises(InvalidTransition):
        controller.propose(ReplyKind.ADVANCE)


def test_skip_interrupted_question_is_explicit_and_does_not_record_answer() -> None:
    controller = InterviewController()
    first = controller.start()
    assert controller.interrupt(first.id)
    second = controller.skip_interrupted()
    assert second.text == QUESTIONS[1]
    assert controller.answers == ((), (), (), ())
    assert controller.current_question_number == 1
    deliver(controller, second)
    assert controller.current_question_number == 2


def test_feedback_must_be_two_or_three_actual_lines() -> None:
    controller = InterviewController()
    reach_feedback(controller)
    with pytest.raises(ValueError):
        controller.prepare_feedback("A generic compliment")
    with pytest.raises(ValueError):
        controller.prepare_feedback("First line\n\nThird line")
    assert controller.stage is Stage.AWAITING_FEEDBACK
