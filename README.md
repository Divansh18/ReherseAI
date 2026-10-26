# RehearseAI

RehearseAI is a local browser voice mock-interview coach. It uses a four-question controller, SmallWebRTC browser transport, Deepgram streaming STT, OpenAI text LLM, OpenAI TTS, and local session logs. A real session exposed an early turn boundary during a follow-up answer; the turn-taking fix below has passed offline tests but still needs a new microphone test.

## Run locally

Use Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/). Create `.env` from `.env.example`, then set the two required server-side credentials:

- `DEEPGRAM_API_KEY` — [Deepgram API key](https://developers.deepgram.com/docs/create-additional-api-keys)
- `OPENAI_API_KEY` — [OpenAI API key](https://platform.openai.com/api-keys), used for both text LLM and TTS

Never place keys in browser code. `.env` and `sessions/` are ignored by Git. Start with one command:

```sh
uv run python -m rehearse_ai
```

Open `http://127.0.0.1:7860/`, choose **Start interview**, allow microphone access, and use headphones if possible. The browser shows connection and microphone state, the current question, and the transcript. **End session** closes WebRTC and stops local microphone tracks. The server binds only to a loopback address. A browser with WebRTC and microphone support is required; localhost is a secure context for microphone permission.

`.env.example` lists configurable port, models, voice, VAD thresholds, turn pause, and session directory. Defaults are Deepgram `nova-3-general`, OpenAI text `gpt-4.1-mini`, and OpenAI TTS `gpt-4o-mini-tts` with voice `coral`. These are separate STT → text LLM → TTS calls. OpenAI TTS avoids a third account but its HTTP synthesis may add delay; latency and quality have not been benchmarked. The LLM uses a structured one-shot text call rather than a streaming spoken response, which prioritizes bounded decisions over first-word speed. No realtime or speech-to-speech model is used.

## Behavior and evidence

Python owns the four exact main questions, their order, one substantive follow-up or nudge per question, and one clarification or repeat per question. The LLM proposes a short contextual action from actual answers. Candidate transcript text is treated as untrusted data. Closing feedback is generated from the session answers as 2–3 short sentences; no keyword grading or canned feedback is used. The exact runtime prompts are linked in [PROMPTS.md](PROMPTS.md).

Pipecat VAD and one `SpeechTimeoutUserTurnStopStrategy` detect speech and decide when a candidate turn ends. Its silence policy now defaults to **1.1 seconds** (`REHEARSE_TURN_PAUSE_SECS`), a starting point for a live tuning experiment. The Pipecat turn processor's 30-second watchdog remains a failure fallback. Pipecat's interruption frame cancels in-flight TTS/output promptly on speech start, independently of the longer completion wait. The session cancels obsolete LLM inference. Final STT segments are accumulated until Pipecat reports the completed turn; the browser merges continuation turns into a readable answer while the JSONL log retains each raw segment. An incomplete answer may yield a silent `wait` decision. A rejected extra prompt never forces advancement. Stale delivery IDs cannot advance the controller.

Each attempt writes `sessions/<uuid>.jsonl` with ordered timestamps, utterance IDs, candidate transcription, prepared/delivered/interrupted agent text, and nullable latency. New diagnostic events include VAD start/stop, individual final STT segments and Pipecat frame timestamps, turn duration and configured pause, inference start/completion/cancellation, and the proposed action. The recorded latency is **candidate turn stop → server output start**, an approximation; it is not browser playback latency. Delivery acknowledgment uses Pipecat's `BotStoppedSpeakingFrame` after the output transport. This reports server output completion and does **not** prove every word played in the browser or was heard.

### Real-session diagnosis and next test

In the inspected session log, the follow-up was interrupted at event 16, `candidate_final` recorded only “So the idea” at event 18, and event 20 rejected the next proposal. The old code unconditionally advanced after any rejected proposal, preparing question two at event 21. The log did not include raw STT or VAD timing, so it cannot establish the exact silence length or the rejected proposal's action. The previous configured pause was 0.8 seconds. Deepgram final transcript chunks did **not** directly advance the interview; Pipecat's stop strategy defined the turn boundary. The adapter also cleared a segment if it arrived before Pipecat's turn-start frame, which could lose words.

For the next microphone test, restart the server with the new code, open the local page, answer question one, then pause around one second mid-answer to its follow-up (“So the idea…”) and continue. Confirm the coach waits for the continuation rather than asking question two. Interrupt the coach once, then end the session and inspect the new `sessions/<uuid>.jsonl` for `stt_final_segment`, `vad_stopped`, `candidate_final`, `decision_proposed`, and `agent_prepared`. If the pause still splits the answer, compare those timestamps before adjusting `REHEARSE_TURN_PAUSE_SECS` within roughly 0.9–1.2 seconds. The new behavior has not yet been verified with a microphone.

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
