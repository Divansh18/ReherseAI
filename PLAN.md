# RehearseAI implementation plan

## Scope and architecture

Build a local browser mock interview using Python 3.11+ and Pipecat. Audio follows a separate speech-to-text (STT) → text LLM → text-to-speech (TTS) cascade. SmallWebRTC connects the browser to the Python process. No realtime audio model, speech-to-speech model, account system, database, or deployment service is in scope.

Keep the implementation in five small areas:

- `config.py`: load and validate local environment settings without exposing keys.
- `src/rehearse_ai/interview.py`: implemented provider-independent state. Python owns the four-question order, one follow-up or nudge and one clarification or repeat per question, and the final feedback transition. Prepared agent text is distinct from delivered speech. Interrupted output requires a new delivery attempt with a new ID.
- `pipeline.py`: assemble Pipecat transport, turn detection, STT, text LLM, TTS, and the adapters that pass confirmed user and agent turns to interview state. Keep provider-specific construction here.
- `session_log.py`: local per-session JSONL events for transcripts, interruptions, and per-turn latency with explicit timestamp boundaries and no credentials.
- `web/`: local browser assets for microphone, audio playback, connection state, and optional visible transcript.

The LLM may word a context-aware follow-up or gentle nudge about the actual answer, but may not choose the next main question or exceed the Python-owned follow-up limit. The required main questions are, in order:

1. Tell me about a project you're proud of.
2. What was the hardest technical problem in it?
3. How did you know it was working well?
4. What would you do differently now?

After question four and any allowed follow-up, deliver 2–3 spoken lines of specific feedback grounded in the candidate's answers. Short mid-answer pauses should not end a turn prematurely. Barge-in should stop agent playback, record the interruption, and reconcile delivery before advancing state.

## Milestones and acceptance checks

| Milestone | Work | Acceptance check |
| --- | --- | --- |
| 1. Foundation | Plan, package, `uv` lock, lint/test configuration, environment template, minimal README, Git baseline. | `uv sync`, package import, Ruff, and pytest scaffold checks are reported accurately; a foundation commit exists if identity is configured. No voice pipeline exists. |
| 2. Interview policy | **Implemented:** pure interview state, fixed question sequence, reply bounds, delivery-aware transitions, and deterministic state tests. | Tests cover all four questions and closing feedback delivery, follow-up/clarification limits, empty answers and repeats, interrupted speech, stale acknowledgments, and completion. Contextual wording awaits the LLM integration. |
| 3. Local voice loop | Assemble current Pipecat SmallWebRTC transport and cascaded STT → text LLM → TTS services; add minimal browser UI and turn handling. | A real local browser session can exchange audio, hear all four questions, pause briefly mid-answer, and interrupt the agent without skipping a question. No speech-to-speech model is used. |
| 4. Observability and prompts | Write `PROMPTS.md`, tune answer-aware follow-up and feedback prompts, and record transcript, interruption, and per-turn latency events. | Logs identify user and agent turns, generated versus delivered speech, interruptions, and defined latency endpoints. Prompt changes and any experiments are documented only when actually run. |
| 5. Real-session verification | Run and review a real voice session, fix observed defects, complete setup/troubleshooting docs, and preserve incremental Git history. | README has reproducible run steps; genuine session evidence demonstrates question order, an answer-aware response, spoken feedback, pauses/interruption behavior, and measured latency. No fabricated artifacts or claims. |

## Decisions still open

- **Service credentials:** Confirm which STT, LLM, and TTS accounts/keys are available before selecting implementations. Initial candidate stack: Deepgram STT, OpenAI text LLM, Cartesia TTS. Each is a separate service, so this remains a cascade. Required keys for that stack: `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, and `CARTESIA_API_KEY`.
- **Voice and models:** Pick exact model and voice IDs from current provider/Pipecat documentation when credentials are known. Avoid hard-coding unverified names now.
- **Turn boundaries:** Compare Pipecat's current turn analyzer and VAD options in a real session. Choose thresholds based on short pauses and barge-in behavior, not a guessed delay.
- **Delivery accounting:** Verify the current Pipecat frames/events that establish actual playback completion and interruption. If server events do not prove browser playback, use a browser acknowledgment before committing a spoken question.
- **Interruption race:** Decide whether an answer captured while agent speech is interrupted should be buffered or treated as a request to repeat. The controller currently rejects candidate answers until the interrupted utterance is retried and acknowledged.
- **Reply proposal shape:** The future LLM adapter must map output to `ReplyKind` and text. The controller enforces count and progression, but relevance of a follow-up and quality of feedback require actual LLM output and session review.
- **Evidence handling:** Decide which real-session artifacts can be shared. Raw audio and local logs remain ignored by Git unless the user intentionally selects a safe, redacted evidence artifact.

## Implemented state contract

`InterviewController` starts in `NEW`. `start()` prepares question one and enters `DELIVERING`; `delivery_finished(id)` moves to `AWAITING_ANSWER` and commits that question as current. A nonempty `submit_answer(text)` stores the answer and enters `AWAITING_PROPOSAL`. `propose(ReplyKind.ADVANCE)` prepares the next fixed question, or enters `AWAITING_FEEDBACK` after question four. A proposed `FOLLOW_UP` or `NUDGE` spends the single substantive slot for that main question. A `CLARIFICATION` spends its separate single clarification slot. `request_repeat()` and an empty answer repeat the exact current main question and spend that same clarification slot. Once the slot is spent, another repeat or empty answer raises `LimitReached` while the controller keeps awaiting an answer; the integration must surface this condition without starting an unbounded clarification loop.

Prepared utterances carry session-scoped IDs. `interrupt(id)` retains the text and intent but invalidates that ID. `retry_interrupted()` issues a new ID. Duplicate or stale delivery/interruption events return `False`. Only delivery of the next main question changes `current_question_number`, and only delivery of prepared 2–3-line feedback enters `COMPLETE`. Transport playback completion is only an approximation of what the human heard; Milestone 3 must select a suitable acknowledgment point.

## API references checked for this plan

- [Pipecat project metadata](https://github.com/pipecat-ai/pipecat/blob/main/pyproject.toml): Python requirement and `webrtc`/`runner` extras.
- [SmallWebRTC server transport](https://docs.pipecat.ai/api-reference/server/services/transport/small-webrtc): audio-only extra and transport role.
- [Choosing a client transport](https://docs.pipecat.ai/client/concepts/choosing-a-transport): matching browser/server transports and local SmallWebRTC use.

Recheck concrete classes, signatures, and events against the installed locked version immediately before Milestone 3 implementation.
