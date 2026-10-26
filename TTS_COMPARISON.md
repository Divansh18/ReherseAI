# Small direct TTS comparison — 2026-10-26

This is a six-request diagnostic, not an application or browser latency benchmark. The app default was subsequently switched to Cartesia at the user's request. Both keys were read from local `.env`; neither key is included in this document or the measurements file.

## Configuration and method

- OpenAI: `gpt-4o-mini-tts`, `shimmer`, speed `1.0`, and the app's current `src/rehearse_ai/prompts/tts_style.txt` instructions, via streaming `/v1/audio/speech`.
- Cartesia: `sonic-3.6`, Skylar (`db6b0ed5-d5d3-463d-ae85-518a07d3c2b4`), English, default generation settings, API version `2026-08-14`, via streaming `/tts/bytes`.
- Each request used raw mono 24 kHz, 16-bit PCM. The received bytes were wrapped in WAV files without changing the samples. The same three short interview lines were sent to each provider in alternating order: OpenAI, Cartesia, OpenAI, Cartesia, OpenAI, Cartesia. An HTTP client was reused across requests.
- Request-to-first-audio starts immediately before the HTTPS request and stops at the first nonempty PCM body chunk. Total synthesis time stops when the response stream ends. These times include client/network and provider response time. They exclude Pipecat processing, transport output, browser buffering, and playback onset. A direct HTTPS request is also not identical to Cartesia's recommended persistent WebSocket service in Pipecat.
- All six requests succeeded. The initial restricted-network run failed before reaching either service and produced no audio; it is excluded from these measurements.

| Order | Line | Provider | First audio | Total | PCM bytes | Sample |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | Project | OpenAI | 1,277.8 ms | 2,134.0 ms | 136,800 | [WAV](samples/tts-comparison-2026-10-26/01_openai.wav) |
| 2 | Project | Cartesia | 179.9 ms | 474.1 ms | 99,840 | [WAV](samples/tts-comparison-2026-10-26/01_cartesia.wav) |
| 3 | Technical problem | OpenAI | 11,717.3 ms | 12,539.0 ms | 148,800 | [WAV](samples/tts-comparison-2026-10-26/02_openai.wav) |
| 4 | Technical problem | Cartesia | 166.3 ms | 556.3 ms | 111,360 | [WAV](samples/tts-comparison-2026-10-26/02_cartesia.wav) |
| 5 | Differently now | OpenAI | 1,080.9 ms | 1,872.2 ms | 117,600 | [WAV](samples/tts-comparison-2026-10-26/03_openai.wav) |
| 6 | Differently now | Cartesia | 141.0 ms | 349.5 ms | 84,480 | [WAV](samples/tts-comparison-2026-10-26/03_cartesia.wav) |

**Medians (three successful requests each):** OpenAI first audio **1,277.8 ms**, total **2,134.0 ms**; Cartesia first audio **166.3 ms**, total **474.1 ms**. The OpenAI technical-problem request was a conspicuous slow outlier; six requests cannot establish p95 latency or reliability. Each WAV was opened and checked for its expected 24 kHz mono PCM format and nonzero signal; this does not establish perceived voice quality.

The exact lines were:

1. “Tell me about a project you're proud of.”
2. “What was the hardest technical problem in it?”
3. “What would you do differently now?”

Machine-readable results are in [measurements.json](samples/tts-comparison-2026-10-26/measurements.json). The WAV files are local generated samples and are ignored by Git.

## Decision

Cartesia was faster in these direct API trials, and all six calls succeeded. At the user's request, `sonic-3.6` with this exact Skylar voice ID is now the application default through Pipecat's streaming WebSocket service. The app retains an explicit OpenAI fallback setting; it does not switch providers mid-response. Listen to each matched pair to judge voice preference. A separate bounded Pipecat synthesis check is in [pipecat_stream_check.json](samples/tts-comparison-2026-10-26/pipecat_stream_check.json): normal and fresh contexts emitted audio, an interrupted context was cancelled, and no old audio reached its test sink after cancellation. That check excludes the browser, microphone, Deepgram, and LLM. Do not infer browser playback latency or production reliability from either check.

## Official references

- [Cartesia Sonic 3.6 model and Skylar voice](https://docs.cartesia.ai/build-with-cartesia/tts-models/latest)
- [Cartesia streaming endpoint trade-offs](https://docs.cartesia.ai/use-the-api/compare-tts-endpoints)
- [Cartesia output formats](https://docs.cartesia.ai/build-with-cartesia/capability-guides/tts-output-audio-format)
- [Pipecat Cartesia TTS integration](https://docs.pipecat.ai/api-reference/server/services/tts/cartesia)
- [OpenAI text-to-speech streaming and PCM format](https://developers.openai.com/api/docs/guides/text-to-speech)
