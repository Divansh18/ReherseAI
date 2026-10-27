# Latency summary

Session: `d564949d-6780-414f-8192-22bae9e9345d`. All timestamps use server monotonic milliseconds from this session's log origin; UTC wall timestamps are not used for subtraction.

`user_stopped_ms` estimates the last speech as the final Silero VAD stop event in the candidate turn minus that event's `stop_secs` silence threshold. It precedes Pipecat's configured turn-completion wait and remains an estimate, not a measured last audible phoneme. Missing VAD thresholds remain null.
`tts_first_audio_ms` is a correlated **server output start proxy**, not measured browser playback. Provider first-audio is separately recorded in the extra column. The current structured LLM call exposes no first-token event, so that field is null.
STT finals use their logged turn ID and accumulated text. Older pre-start segments are linked by event order only when the complete text and segment count match; ambiguous or absent STT timestamps remain null. Ambiguous STT correlation: 0 turns.

## User speech end to first server output

- Candidate turns: 14; included: 12; excluded (interrupted, cancelled, missing, or reversed): 2.
- p50: 2798.9 ms; p95: 5007.3 ms.
- Percentiles use nearest rank: sorted value at ceil(p × n), with no interpolation.

## Stage observations

- speech end → last STT final: 443.2 ms (n=14; 0 missing/overlapping)
- last STT final → turn completion: 1008.6 ms (n=14; 0 missing/overlapping)
- LLM request → first token: unavailable (no first-token event; request timestamps are retained separately).
- provider TTS request → provider first audio: 102.9 ms (n=12; 2 missing/overlapping)
- speech end → server output start: 3004.9 ms (n=12; 2 missing/overlapping)
Stage averages use all turns with observed, nonnegative endpoints, including interrupted output when it began. Negative differences from overlapping stages are excluded, not reported as processing durations. Interrupted turns remain in JSONL and are excluded from the primary response percentile. No browser onset is available.
