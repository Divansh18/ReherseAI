# RehearseAI

RehearseAI is a local browser voice mock-interview coach. It uses a four-question controller, SmallWebRTC browser transport, Deepgram streaming STT, OpenAI text LLM, OpenAI TTS, and local session logs. A real session exposed lifecycle and output timing faults; the current fix has passed offline tests but still needs a new microphone test.

## Run locally

Use Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/). Create `.env` from `.env.example`, then set the two required server-side credentials:

- `DEEPGRAM_API_KEY` — [Deepgram API key](https://developers.deepgram.com/docs/create-additional-api-keys)
- `OPENAI_API_KEY` — [OpenAI API key](https://platform.openai.com/api-keys), used for both text LLM and TTS

Never place keys in browser code. `.env` and `sessions/` are ignored by Git. Start with one command:

```sh
uv run python -m rehearse_ai
```

Open `http://127.0.0.1:7860/`, choose **Start interview**, allow microphone access, and use headphones if possible. The browser shows connection and microphone state, the current question, and the transcript. **End session** closes WebRTC and stops local microphone tracks. The server binds only to a loopback address. A browser with WebRTC and microphone support is required; localhost is a secure context for microphone permission.

`.env.example` lists configurable port, models, voice, VAD thresholds, turn pause, and session directory. Defaults are Deepgram `nova-3-general`, OpenAI text `gpt-4.1-mini`, and OpenAI TTS `gpt-4o-mini-tts` with voice `coral`. These are separate STT → text LLM → TTS calls. OpenAI TTS avoids a third account. The LLM's structured `run_inference()` call returns a complete JSON decision before validation, so no unvalidated control text can enter TTS. Pipecat's OpenAI TTS service streams generated audio chunks. Provider and transport delay contributions have not yet been measured separately in a real session. No realtime or speech-to-speech model is used.

## Behavior and evidence

Python owns the four exact main questions, their order, one substantive follow-up or nudge per question, and one clarification or repeat per question. The LLM proposes a short contextual action from actual answers and receives the exact current question, its answer, the last delivered prompt, allowance counts, and interruption context. Older answers are background only. Explicit spoken stop, repeat, and skip requests are handled before inference. Normal closing feedback is generated from session answers as 2–3 short sentences; an early stop enters a separate state and offers a 1–2 sentence closing based only on collected answers. With no collected answer, it gives a short generic closing. No keyword grading is used. The exact runtime prompts are linked in [PROMPTS.md](PROMPTS.md).

Pipecat VAD and one `SpeechTimeoutUserTurnStopStrategy` detect speech and decide when a candidate turn ends. Its silence policy now defaults to **1.1 seconds** (`REHEARSE_TURN_PAUSE_SECS`), a starting point for a live tuning experiment. The Pipecat turn processor's 30-second watchdog remains a failure fallback. Pipecat's interruption frame cancels in-flight TTS/output promptly on speech start, independently of the longer completion wait. The session cancels obsolete LLM inference. Final STT segments are accumulated until Pipecat reports the completed turn; the browser merges continuation turns into a readable answer while the JSONL log retains each raw segment. An incomplete answer may yield a silent `wait` decision. A rejected extra prompt never forces advancement. Stale delivery IDs cannot advance the controller.

Each attempt writes `sessions/<uuid>.jsonl` with ordered UTC timestamps and a common server monotonic `session_elapsed_ms` clock, utterance IDs, candidate transcription, and nullable latency. Events include estimated VAD speech end, Pipecat turn completion, final STT receipt, LLM request/validated decision, TTS request/first generated audio, first server output, cancellation, and delivery. Pipecat's structured inference does not expose a first-token time here, and WebRTC does not expose reliable per-utterance browser playback onset; both remain null. Generated audio is logged as generated, not played. The recorded latency is **candidate turn stop → server output start**, an approximation; it is not browser playback latency. Delivery acknowledgment uses Pipecat's `BotStoppedSpeakingFrame` after the output transport only when the matching utterance had audio output and TTS completion and was not cancelled. Pipecat may also emit that frame on interruption, so cancelled output stops are ignored. This remains server-side evidence, not proof every word played in the browser or was heard. The browser shows “Preparing response” while TTS is pending and adds coach text to the transcript only when server output starts, labeling its status explicitly.

### Latest spoken-stop regression

The newest local log, `sessions/50ebe9c1-0f7f-4bc9-8dce-4293aeae981a.jsonl`, records “Can I end this interview?” as two final STT segments at events 38 and 40, combined into `candidate_final` at event 41. Event 44 then chose `advance`, and the server output question two. “Want you to end the interview. Right now.” arrived as two segments at events 51 and 54, combined at event 55; event 58 chose a nudge and the server output another question. Neither turn produced `early_end_requested`. The log also lacks `session_elapsed_ms`, `tts_requested`, and `candidate_turn_completed`, all unconditional in commit `7c6647d`; a server was still listening on port 7860 from this checkout. This session therefore ran in a process with an older code load. Restart the server before retesting. The `7c6647d` matcher itself also missed “Can I…” and the subject-dropped “Want you to…” phrasing, so a restart alone would not have covered these examples.

Stop intent now matches a complete affirmative request clause rather than a substring. Final STT segments are joined, then checked as each segment arrives; a match cancels pending inference and question progression before ordinary answer submission. Pipecat's speech-onset interruption remains responsible for immediate audio cancellation, and the TTS adapter checks the pending utterance ID again before sending speech. The closing response waits until the candidate turn completes, so it does not talk over the request. The tests cover the exact transcriptions, a queued question, a cancellation between its marker and TTS request, and negative statements such as “I ended the interview yesterday” and “Don't end the interview.” These are mocked routing checks, not proof of live microphone behavior.

In the same log, the candidate's three STT segments at events 21, 23, and 26 were combined into a complete answer: the project posts a structured risk report as a PR comment in under sixty seconds. Event 30 nevertheless asked the candidate to clarify what the project does and what “posts a structured risk report” means. The log does not contain the LLM request payload, so its exact prompt context in that older process cannot be reconstructed. The current request now includes a single `current_question_answer` field joining every collected answer for that question; the runtime prompt directs the coach to ask for one missing detail or advance rather than ask the candidate to restate details already given. Live relevance remains unverified.

### Earlier lifecycle diagnosis and next test

An earlier inspected real log is `sessions/d291e05e-4a8b-4e42-8cd7-a54a4d03b24d.jsonl` (local, ignored by Git). Question two was prepared as utterance 4 at 69.94 seconds and interrupted at 70.04 seconds, with **no** output-start event. The following candidate turn was treated as an answer to question two, and utterance 5 asked about question one's risk report instead of the technical challenge. “Can we end the interview?” was transcribed at 156.48 and 171.72 seconds; both requests went through the ordinary answer/LLM path. Another stop request at 208.80 seconds produced a clarification about older wording. Agent text was shown on `agent_prepared`, before output: for the first question the gap to `agent_output_started` was about 4.03 seconds; one later response took about 9.05 seconds. The old log lacks TTS request and first-audio timestamps, so it cannot assign that delay to synthesis, output buffering, or browser playback.

For the next microphone test, restart the server with the new code, open the local page, answer question one, and pause briefly while answering a follow-up (“So the idea…”), then continue. On question two, interrupt immediately after its text is prepared but before speech if possible; it must be asked again before an answer is assigned to it. Ask “Can we end the interview?” during a later answer or while the coach is preparing a response; expect a short early closing, no next question, and then use **End session** to release the microphone. Inspect the new `sessions/<uuid>.jsonl` for `agent_prepared`, `tts_requested`, `tts_first_audio_generated`, `agent_output_started`, `agent_interrupted`, `candidate_turn_completed`, `early_end_requested`, and `agent_delivery_reported` by utterance ID and `session_elapsed_ms`. If the pause still splits the answer, compare `vad_stopped` and `candidate_turn_completed` before adjusting `REHEARSE_TURN_PAUSE_SECS` within roughly 0.9–1.2 seconds. The new lifecycle and pause behavior have not yet been verified with a microphone.

## Offline checks

```sh
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv lock --check
uv build
```

See [PLAN.md](PLAN.md) for open integration questions and the remaining real-session verification work.
