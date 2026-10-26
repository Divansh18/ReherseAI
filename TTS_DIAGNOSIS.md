# TTS context timeout investigation

This investigation concerns real session `3725ce4f-17fd-4262-9d38-bca06677f30b` on build `0.1.0+49e1ee1a8659`. It does not use or store API key values or audio recordings.

## Affected utterance

The candidate's complete turn (“Won't you ask about what I have done till now about my experience?”) ended at session time 86,083 ms. The structured LLM decision completed at 87,924 ms with action `detour` and the nonempty text “I can answer questions about your experience if you'd like. What would you like to share about it?” Python prepared utterance 10 and requested TTS at 87,924 ms. Pipecat started TTS context `79b88159-e0df-436b-9a75-be034e8c8d6e` at 87,929 ms. At 90,928 ms it emitted `TTSStoppedFrame`; at 90,931 ms its `OpenAITTSService` emitted `ErrorFrame` with `TTS context ... completed with no audio`. The session recorded no first audio frame, server output start, or delivery for utterance 10. Candidate speech cancelled it only at 108,665 ms, more than 20 seconds after TTS started. The preceding utterance 9 was delivered before utterance 10 began; no shared context ID or overlapping request appears in this sequence.

The same session contains no-audio errors for utterances 3, 13, and 14. Each context ended **2,999–3,001 ms** after `tts_started`. Successful contexts first emitted audio in about 1.4–2.2 seconds. These are server event times, not browser playback times.

## Layer and bounded synthesis checks

Installed Pipecat 1.14.0 `TTSService` defaults `stop_frame_timeout_s` to 3.0. Its audio-context playback task waits for a context item with that timeout. If none arrives, it closes the context and `_record_context_audio_outcome` emits the no-audio `ErrorFrame`. The message is emitted by Pipecat's context layer; it does **not** prove OpenAI sent an empty response. The local source even notes that an HTTP TTS service may take more than three seconds before sending audio. The original session did not log raw provider chunks or an HTTP error/status, so whether that particular provider request eventually produced audio is unknown. The first observed failure is the context's three-second closure before any downstream audio frame.

Using the configured `gpt-4o-mini-tts`, `shimmer`, speed `1.0`, and versioned style instructions, with the exact nonempty utterance-10 text:

| Check | First audio | Audio | Outcome |
| --- | ---: | ---: | --- |
| Direct OpenAI streaming API, before change | 3,096 ms | 39 chunks, 314,400 PCM bytes | HTTP 200; first audio arrived after Pipecat's old deadline. |
| Minimal Pipecat pipeline, old 3-second context timeout | 2,179 ms | 12 chunks, 285,600 bytes | Completed; this request happened to beat the deadline. |
| Direct OpenAI streaming API, after change | 2,380 ms | 32 chunks, 256,800 bytes | HTTP 200. |
| Minimal Pipecat pipeline, 8-second context timeout | 2,244 ms | 11 chunks, 244,800 bytes | Completed without pipeline error. |

A separate mocked-provider Pipecat regression deliberately delays its first audio beyond a short context deadline: it reproduces the premature no-audio error, while a longer deadline delivers 4,800 bytes without that error. It makes no claim about OpenAI latency.

A temporary full SmallWebRTC server on port 7861 used an eight-second context timeout and a synthetic, silent client microphone. Its uninterrupted greeting delivered 220,800 TTS bytes (10 downstream chunks) with no error. In two further real pipeline checks, a **locally synthesized** “Stop the interview” phrase was sent during greeting generation and after greeting output began. The sessions recognized that stop request, cancelled utterance 1, and delivered only early-closing utterance 2 (153,600 and 151,200 TTS bytes respectively). No cancelled utterance was marked delivered. These were automated WebRTC checks, not the user's microphone or a listening-quality test. Their ignored logs are `sessions/45603d9d-44ca-462e-a123-359496f86d95.jsonl`, `sessions/d457da22-4adb-4d9c-8c0f-6647bdebcecc.jsonl`, and `sessions/4301f704-d6eb-4864-bc3f-821e09572d31.jsonl` (temporary build `0.1.0+bb3bf5f7ec33`).

Eight seconds was insufficient in a later full WebRTC attempt: `sessions/5079e254-9f97-47fc-a3ad-c2309322c988.jsonl` records context `9b514e11-e559-4bdc-90ef-0f2a69815db1` closing at 8,003 ms with zero downstream bytes, followed by the **first provider-yielded 24,000-byte audio chunk at 9,673 ms** for the same context. That chunk never reached the downstream audio observer. This establishes the timeout race for an instrumented real request. The original `3725...` session lacked raw-provider instrumentation, so the exact fate of its HTTP stream remains unknown.

With the final 20-second timeout and provider-frame instrumentation, two uninterrupted synthetic WebRTC sessions delivered their greetings without a no-audio error: `sessions/4d4c3e61-66bf-4e0f-8fe3-ca72172ff70f.jsonl` recorded first provider audio at 3,367 ms and 290,400 provider/downstream PCM bytes; `sessions/18bdecbf-332e-45e8-821b-7118f266c1cf.jsonl` recorded first provider audio at 1,183 ms and 242,400 bytes. A playback interruption in `sessions/48f21570-0a9b-47fd-8086-f5c8e0e4e5e0.jsonl` cancelled utterance 1 after two provider chunks, recognized the locally synthesized stop phrase, and delivered only early closing utterance 2 (153,600 bytes). A generation interruption in `sessions/f3b046ec-4130-44e2-86fa-540be8076928.jsonl` cancelled utterance 1 before provider audio and delivered a fresh utterance 2 (237,600 bytes), with no late delivery of utterance 1; that synthetic stop phrase transcribed as empty in this attempt, so it retried the greeting rather than ending. These four sessions used temporary build `0.1.0+a59a818d5a5c`, not the user's microphone.

## Fix and limits

The OpenAI Pipecat TTS context now waits up to **20 seconds** for an idle context item, above the longest first-audio delay actually observed (9.673 seconds). The service still streams audio as it arrives; normal completion does not wait for the idle timeout. A genuinely silent or stalled context can now take up to 17 seconds longer than before to report failure. Provider-yielded and downstream audio chunk and byte totals are logged separately by context, without recording audio or secrets. A late audio frame from a cancelled or stale context is discarded before transport and cannot mark delivery. No automatic retry or provider switch was added.

This evidence strongly identifies the too-short context timeout as the recurring *false no-audio* mechanism. It cannot prove that every affected historical OpenAI request would have produced audio if given longer. A real microphone session on the final build must confirm whether the warning recurs. If it does, inspect the new context elapsed time and byte totals alongside any provider error/status before changing provider or retry policy.
