# RehearseAI implementation plan

## Architecture

RehearseAI runs on Python 3.11+ with Pipecat 1.14.0. `config.py` validates local settings. `interview.py` is the provider-independent state machine. `session.py` maps candidate and agent events to it, cancels obsolete inference, and writes `session_log.py` events. `pipeline.py` assembles SmallWebRTC input → Silero VAD → Deepgram streaming STT → Pipecat user-turn processor → interview adapter → OpenAI TTS → SmallWebRTC output → delivery observer. The OpenAI text LLM is called through Pipecat's `run_inference()` for a structured decision or closing feedback; its JSON is never spoken directly. `server.py` serves the package's `web/` assets and WebRTC offer endpoint. Runtime prompts live in `src/rehearse_ai/prompts/` and are documented in `PROMPTS.md`.

Python asks these main questions exactly and in order:

1. Tell me about a project you're proud of.
2. What was the hardest technical problem in it?
3. How did you know it was working well?
4. What would you do differently now?

The LLM may propose a context-aware follow-up, nudge, clarification, or advance. Python limits each question to one substantive follow-up/nudge and one clarification/repeat, selects each next main question, and blocks progress after closing feedback delivery. Candidate text is untrusted data. Closing feedback must use actual recorded answers and contain 2–3 short spoken sentences.

## Five milestones

| Milestone | Status and acceptance check |
| --- | --- |
| 1. Foundation | Complete: package, `uv` lock, configuration template, checks, foundation commit. |
| 2. Interview policy | Complete: four-question delivery-aware state machine and deterministic tests. |
| 3. Local voice loop | Implemented: cascaded Pipecat services, loopback SmallWebRTC server, browser, prompt files, and session logging. Offline tests and HTTP smoke check pass. Acceptance of **real voice behavior is pending credentials and live audio testing**. |
| 4. Observability and prompts | Foundation implemented: exact versioned prompts and local event log. Review genuine logs, tune turn settings, and document prompt experiments only after running them. |
| 5. Real-session verification | Pending: record a genuine session, inspect four-question order, relevance, brief feedback, pause/barge-in handling, and measured latency. Fix any observed faults and finalize troubleshooting docs. |

## Delivery and interruption contract

An agent utterance is first *prepared* with a session-scoped ID. Generation does not count as delivery. The delivery observer calls `delivery_finished(id)` only on Pipecat's `BotStoppedSpeakingFrame` after its output transport; duplicate and stale IDs are ignored. That frame is a server output completion signal, not proof of browser playout or human hearing. The browser transcript therefore distinguishes prepared, interrupted, and server-reported completed speech.

Pipecat `UserTurnProcessor` uses VAD/transcription start strategies and one `SpeechTimeoutUserTurnStopStrategy`. There is no separate application end-of-turn timer. On barge-in, Pipecat broadcasts an interruption to cancel speech output; `InterviewSession` cancels an in-flight LLM task and invalidates the pending utterance ID. If the new candidate turn contains a substantive answer, Python accepts it for the interrupted question/follow-up and continues. If it is empty or asks for a repeat, the interrupted text is retried with a fresh ID. Interrupted closing feedback can be replanned from actual answers plus closing remarks, but the interview is not marked complete until a new feedback utterance reaches its delivery event.

## Open decisions and verification

- `DEEPGRAM_API_KEY` is not yet configured for this workspace. No genuine Deepgram/OpenAI voice session, quality comparison, prompt experiment, or latency benchmark has been run. Credentials remain in ignored `.env`.
- Confirm the full WebRTC answer, data-channel messages, browser playout, and cleanup with real microphone/audio access. The HTTP page and health endpoint have been smoke-tested only.
- Tune VAD start/stop and the 0.8-second turn pause from a real session, especially for short mid-answer pauses and noisy microphones.
- Consider browser playout acknowledgment if stronger delivery evidence is needed; current Pipecat server event is deliberately labeled approximate.
- Decide which genuine evidence can be shared. Raw session logs and recordings are ignored by Git.

## API and provider references

- [Pipecat 1.14 SmallWebRTC transport](https://docs.pipecat.ai/api-reference/server/services/transport/small-webrtc) and [transport guidance](https://docs.pipecat.ai/client/concepts/choosing-a-transport).
- [Pipecat source](https://github.com/pipecat-ai/pipecat): locked 1.14.0 service constructors, `run_inference`, turn strategies, `TTSSpeakFrame`, and output events were checked against installed source before implementation.
- [Deepgram Nova-3 model overview](https://developers.deepgram.com/docs/models-languages-overview), [OpenAI GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), and [OpenAI TTS API](https://platform.openai.com/docs/api-reference/audio/createSpeech).
