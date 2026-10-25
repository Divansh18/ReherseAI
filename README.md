# RehearseAI

RehearseAI is a local browser voice mock-interview coach. Milestone 3 implements the four-question controller, SmallWebRTC browser transport, Deepgram streaming STT, OpenAI text LLM, OpenAI TTS, and local session logs. The HTTP server and browser assets have been smoke-tested, and provider behavior has been tested with mocked inference and speech. **No real spoken session has been verified yet.**

## Run locally

Use Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/). Create `.env` from `.env.example`, then set the two required server-side credentials:

- `DEEPGRAM_API_KEY` — [Deepgram API key](https://developers.deepgram.com/docs/create-additional-api-keys)
- `OPENAI_API_KEY` — [OpenAI API key](https://platform.openai.com/api-keys), used for both text LLM and TTS

Never place keys in browser code. `.env` and `sessions/` are ignored by Git. Start with one command:

```sh
uv run python -m rehearse_ai
```

Open `http://127.0.0.1:7860/`, choose **Start interview**, allow microphone access, and use headphones if possible. The browser shows connection and microphone state, the current question, and the transcript. **End session** closes WebRTC and stops local microphone tracks. The server binds only to a loopback address. A browser with WebRTC and microphone support is required; localhost is a secure context for microphone permission.

`.env.example` lists configurable port, models, voice, VAD thresholds, turn pause, and session directory. Defaults are Deepgram `nova-3-general`, OpenAI text `gpt-4.1-mini`, and OpenAI TTS `gpt-4o-mini-tts` with voice `coral`. These are separate STT → text LLM → TTS calls. OpenAI TTS avoids a third account but its HTTP synthesis may add delay; latency and quality have not been measured in a real session. The LLM uses a structured one-shot text call rather than a streaming spoken response, which prioritizes bounded decisions over first-word speed. No realtime or speech-to-speech model is used.

## Behavior and evidence

Python owns the four exact main questions, their order, one substantive follow-up or nudge per question, and one clarification or repeat per question. The LLM proposes a short contextual action from actual answers. Candidate transcript text is treated as untrusted data. Closing feedback is generated from the session answers as 2–3 short sentences; no keyword grading or canned feedback is used. The exact runtime prompts are linked in [PROMPTS.md](PROMPTS.md).

Pipecat VAD and one user-turn stop strategy detect speech, with a configurable 0.8-second pause. Pipecat's interruption frame cancels in-flight TTS/output, and the session cancels obsolete LLM inference. If the candidate interrupts an utterance with a substantive answer, the controller records the answer and moves on without replay. An empty barge-in retries the interrupted utterance with a new ID. Stale delivery IDs cannot advance the controller. The visible transcript marks prepared and interrupted agent text separately from server-reported delivery.

Each attempt writes `sessions/<uuid>.jsonl` with ordered timestamps, utterance IDs, candidate transcription, prepared/delivered/interrupted agent text, and nullable latency. The recorded latency is **candidate turn stop → server output start**, an approximation; it is not browser playback latency. Delivery acknowledgment uses Pipecat's `BotStoppedSpeakingFrame` after the output transport. This reports server output completion and does **not** prove every word played in the browser or was heard. Real-session audio, latency, and prompt experiments remain unverified.

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
