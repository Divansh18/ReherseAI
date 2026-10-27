# RehearseAI

RehearseAI is a local browser voice coach for a four-question project interview. It listens to a spoken answer, may ask one relevant follow-up or nudge per question, and finishes a normal session with brief answer-grounded feedback and a sign-off. The opening, optional name, readiness, stop requests, repeats, and interruptions are handled separately from the four questions. The required 3–5 minute submission recording and its matching latency/transcript export are **pending**; see [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md).

## Run locally

Python **3.11+** is required (the repository's `.python-version` selects 3.12). Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --frozen --group dev
cp .env.example .env
```

Create server-side keys at [Deepgram](https://developers.deepgram.com/docs/create-additional-api-keys), [OpenAI](https://platform.openai.com/api-keys), and [Cartesia](https://play.cartesia.ai/keys). Put `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, and `CARTESIA_API_KEY` in `.env`. With `REHEARSE_TTS_PROVIDER=openai`, the Cartesia key is not required. `.env.example` lists every setting and its current default; no key belongs in browser assets. Start with one command:

```sh
uv run python -m rehearse_ai
```

Open **http://127.0.0.1:7860/**, select **Start interview**, and grant microphone access. Headphones reduce feedback into the microphone. The server binds only to loopback. `/healthz` and the browser's collapsed Diagnostics area show the running build ID; refresh after a restart to compare them. **End session** releases the browser microphone and peer connection immediately. Normal and spoken early closing use a short buffered-audio drain before automatic cleanup, which is an approximation rather than a browser playout acknowledgment.

## Architecture and provider choices

`SmallWebRTC` browser audio → Pipecat `VADProcessor` (Silero) → Deepgram streaming STT (`nova-3-general`) → Pipecat `UserTurnProcessor` → `InterviewInput` → OpenAI text LLM (`gpt-4.1-mini`, structured `run_inference`) → Cartesia streaming TTS (`sonic-3.6`, Skylar `db6b0ed5-d5d3-463d-ae85-518a07d3c2b4`) → `AudioObserver` → SmallWebRTC output → `DeliveryObserver`.

The STT, text LLM, and TTS are separate services; there is no realtime speech-to-speech model. The text LLM runs as a structured Pipecat service call outside the audio frame chain, and its validated JSON is never spoken directly. `interview.py` owns exact question order, state, and limits without provider imports. `session.py` routes completed turns, validates permitted decisions, cancels stale generations, and correlates utterance IDs. In `pipeline.py`, `InterviewInput` merges final STT chunks until turn completion; `AudioObserver` correlates TTS contexts and drops stale audio; `DeliveryObserver` observes transport speaking frames. `tts.py` chooses a provider and observes provider chunks independently of transport output. `session_log.py` writes raw ordered events. `config.py` validates only credentials for the selected providers. Versioned prompt files in `src/rehearse_ai/prompts/` are the runtime authority; [PROMPTS.md](PROMPTS.md) includes their exact text and iteration record.

Deepgram offers streaming recognition but can mishear project names. The OpenAI text model supplies contextual proposals while Python enforces transitions; structured inference is fully buffered before TTS and can add delay. Cartesia uses Pipecat's streaming WebSocket service and was selected after a **six-request direct API diagnostic**: median request-to-first-PCM was **166.3 ms** for Cartesia and **1,277.8 ms** for OpenAI `gpt-4o-mini-tts`/`shimmer`; all six succeeded. These are three short utterances per provider, not reliability, voice-quality, browser, or end-to-end latency estimates. See [TTS_COMPARISON.md](TTS_COMPARISON.md) and its [tracked measurements](samples/tts-comparison-2026-10-26/measurements.json). Audio samples are local and ignored by Git. OpenAI TTS remains manually selectable through `REHEARSE_TTS_PROVIDER=openai`, with no automatic mid-response switch; its separate configuration and context-timeout diagnosis are in [TTS_DIAGNOSIS.md](TTS_DIAGNOSIS.md). Provider pricing, accents, and perceived voice quality were not benchmarked.

## Turn-taking, interruptions, and evidence

Defaults are VAD confidence **0.7**, speech start **0.2 s**, VAD stop **0.35 s**, and one Pipecat `SpeechTimeoutUserTurnStopStrategy` with **1.1 s** speech timeout. The turn processor also has a **30 s failure watchdog**. The timeout was raised from **0.8 s** after a real follow-up answer split at “So the idea”; a separate orchestration fallback also advanced after a rejected extra probe. The 1.1 s value is a tuning starting point, not proven to cover every mid-answer pause. Final STT segments accumulate until Pipecat completes one user turn. Speech onset triggers Pipecat interruption and cancels pending LLM/TTS immediately, independently of the longer completion wait. Python keeps one substantive probe per main question, separates explicit repeat/skip/stop, and never treats an unheard generated question as delivered.

Agent text is prepared with an utterance ID before TTS, but enters the visible transcript only on server output start. A matching `BotStoppedSpeakingFrame` after TTS completion and without cancellation is the delivery acknowledgment; Pipecat can also emit stop on interruption, so stale and cancelled stops are ignored. A cancelled TTS context's late stop is filtered before transport output, and a fresh output start retires unmatched cancellation-stop accounting. This is **server output evidence**, not proof of browser playback or human hearing. Raw `sessions/<id>.jsonl` events use a common server monotonic session clock and retain final STT chunks, VAD stops, inference, provider audio, output, cancellation, and delivery. Final STT segments arriving before Pipecat's turn-start frame are tagged with the upcoming candidate turn at ingestion. The structured LLM call exposes no first-token timestamp and the browser supplies no reliable per-utterance playback onset. These remain null in the export; `tts_first_audio_ms` is explicitly labeled as a server-output proxy. Provider first-audio is a separate observation. Historical development sessions are not the future recording's evidence.

For a recorded session, use [RECORDING.md](RECORDING.md) to export `transcript.txt`, `latency.jsonl`, and `latency_summary.md` from that same session. The exporter can correlate older pre-start STT events by event order only when their complete accumulated text and segment count match the final turn; `stt_final_basis` records the method, and ambiguous or absent STT timing stays null. It estimates speech end from VAD stop minus its configured silence threshold, excludes negative/overlapping stage differences, and reports nearest-rank p50/p95 on noninterrupted turns with a correlated output event. **Recording-session p50, p95, and stage averages are pending; no matching recording has been supplied.** The direct-provider medians above must not be read as those end-to-end figures.

## Limits and next day

A local microphone run still needs to establish a complete four-question path with natural pause and barge-in behavior on the final build. STT wording errors can still affect follow-ups; structured decisions may choose a poor detour or wait action despite Python guards. The single silence threshold trades fewer premature boundaries for slower turn endings. Browser playback onset, true human audibility, and LLM first token are unmeasured. With another day, I would add correlated browser playout telemetry, conduct the required recorded pause/interruption session, tune turn timing from that evidence, and review more real answers for follow-up relevance before changing prompts.

**AI tools used:** Codex assisted with code, debugging, tests, and documentation. The author should add any other tools used. **Approximate active time spent:** author to fill in; Git timestamps do not measure work time. No 3–4 hour claim is made.

## Checks

```sh
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
node --check src/rehearse_ai/web/app.js
node --test tests/browser_ui.test.mjs
uv lock --check
uv build
```
