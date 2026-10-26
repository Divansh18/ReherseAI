# RehearseAI implementation plan

## Architecture

RehearseAI runs on Python 3.11+ with Pipecat 1.14.0. `config.py` validates local settings. `interview.py` is the provider-independent state machine. `session.py` maps candidate and agent events to it, cancels obsolete inference, and writes `session_log.py` events. `pipeline.py` assembles SmallWebRTC input → Silero VAD → Deepgram streaming STT → Pipecat user-turn processor → interview adapter → OpenAI TTS → SmallWebRTC output → delivery observer. The OpenAI text LLM is called through Pipecat's `run_inference()` for a structured decision or closing feedback; its JSON is never spoken directly. `server.py` serves the package's `web/` assets and WebRTC offer endpoint. Runtime prompts live in `src/rehearse_ai/prompts/` and are documented in `PROMPTS.md`.

Python asks these main questions exactly and in order:

1. Tell me about a project you're proud of.
2. What was the hardest technical problem in it?
3. How did you know it was working well?
4. What would you do differently now?

The LLM may propose a context-aware follow-up, nudge, clarification, wait, or advance. Python limits each question to one substantive follow-up/nudge and one clarification/repeat, selects each next main question, and blocks progress after closing feedback delivery. A `wait` decision leaves a fragmented answer open, and a rejected proposal never implicitly advances. Candidate text is untrusted data. Closing feedback must use actual recorded answers and contain 2–3 short spoken sentences.

## Five milestones

| Milestone | Status and acceptance check |
| --- | --- |
| 1. Foundation | Complete: package, `uv` lock, configuration template, checks, foundation commit. |
| 2. Interview policy | Complete: four-question delivery-aware state machine and deterministic tests. |
| 3. Local voice loop | Implemented: cascaded Pipecat services, loopback SmallWebRTC server, browser, prompt files, and session logging. A real session revealed premature advancement during a follow-up. The fix is offline-tested; the next real voice check is pending. |
| 4. Observability and prompts | Raw STT, VAD, boundary, and decision events are now logged alongside the exact versioned prompts. Tune turn settings and document prompt experiments only after running them. |
| 5. Real-session verification | Pending: rerun a genuine session on the fix, inspect four-question order, relevance, brief feedback, pause/barge-in handling, and measured latency. Fix any observed faults and finalize troubleshooting docs. |

## Delivery and interruption contract

An agent utterance is first *prepared* with a session-scoped ID. Generation does not count as delivery. The delivery observer calls `delivery_finished(id)` only on Pipecat's `BotStoppedSpeakingFrame` after its output transport; duplicate and stale IDs are ignored. That frame is a server output completion signal, not proof of browser playout or human hearing. The browser transcript therefore distinguishes prepared, interrupted, and server-reported completed speech.

Pipecat `UserTurnProcessor` uses VAD/transcription start strategies and one `SpeechTimeoutUserTurnStopStrategy` for normal turn completion. Its silence policy is now 1.1 seconds, with a 30-second processor watchdog for failure recovery. There is no application end-of-turn timer. On barge-in, Pipecat broadcasts an interruption immediately to cancel speech output; `InterviewSession` cancels an in-flight LLM task and invalidates the pending utterance ID. If the new candidate turn contains a substantive answer, Python accepts it for the interrupted question/follow-up and continues. If it is empty or asks for a repeat, the interrupted text is retried with a fresh ID. Interrupted closing feedback can be replanned from actual answers plus closing remarks, but the interview is not marked complete until a new feedback utterance reaches its delivery event.

## Follow-up turn diagnosis

The inspected real session log shows a follow-up interrupted at event 16, `candidate_final` with only “So the idea” at event 18, `proposal_rejected` at event 20, and question two prepared at event 21. The previous `REHEARSE_TURN_PAUSE_SECS` default was 0.8 seconds. The log lacked VAD and raw STT events, so the exact silence duration and rejected action are unknown. In the installed Pipecat 1.14.0 source, Deepgram final chunks produce `TranscriptionFrame`; `SpeechTimeoutUserTurnStopStrategy` controls turn completion after VAD stop and its configured wait. The controller did not advance merely when the allowance was spent, but the session orchestration converted any rejected proposal into `ADVANCE`. That fallback and the early boundary together caused the visible jump. There is no evidence of duplicate callbacks or a stale LLM reply in the recorded event sequence. The follow-up had no `agent_output_started` event before interruption, despite the user's report that it was heard; check delivery observation in the next live run.

The adapter now retains final segments that arrive before `UserStartedSpeakingFrame`, joins all final segments until one `UserStoppedSpeakingFrame`, and ignores a duplicate stop frame. An incomplete answer can cause a silent `wait`; new speech appends to the same answer. The UI merges continuation turns, while JSONL retains each raw timestamped segment and the inferred boundary. One decision inference is started per completed candidate turn; a resumed candidate turn cancels the previous inference. These are offline-tested changes, not proof that the new 1.1-second setting handles the actual microphone pause.

## Open decisions and verification

- Repeat the actual follow-up interruption and “So the idea…” pause with the new code. Compare VAD stop, STT segment, turn completion, decision, and output timestamps in the new log. A 0.9–1.2 second pause range is an initial tuning experiment, not a guaranteed solution.
- Check browser playout and cleanup in the next real session; server delivery still does not prove browser playout.
- Review whether the revised `wait` prompt makes suitable decisions after incomplete answers. Its new behavior is not yet live-tested, and no latency or quality benchmark has been run.
- Consider browser playout acknowledgment if stronger delivery evidence is needed; current Pipecat server event is deliberately labeled approximate.
- Decide which genuine evidence can be shared. Raw session logs and recordings are ignored by Git.

## API and provider references

- [Pipecat 1.14 SmallWebRTC transport](https://docs.pipecat.ai/api-reference/server/services/transport/small-webrtc) and [transport guidance](https://docs.pipecat.ai/client/concepts/choosing-a-transport).
- [Pipecat source](https://github.com/pipecat-ai/pipecat): locked 1.14.0 service constructors, `run_inference`, turn strategies, `TTSSpeakFrame`, and output events were checked against installed source before implementation.
- [Deepgram Nova-3 model overview](https://developers.deepgram.com/docs/models-languages-overview), [OpenAI GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), and [OpenAI TTS API](https://platform.openai.com/docs/api-reference/audio/createSpeech).
