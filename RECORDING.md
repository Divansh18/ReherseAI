# Recorded session evidence

A [Loom recording](https://www.loom.com/share/928bba7d68ec4174856488a0ed2e1bb7) has been supplied, and the author confirmed its matching local session ID below. The [transcript](evidence/recorded-session/transcript.txt), [latency.jsonl](evidence/recorded-session/latency.jsonl), and [latency summary](evidence/recorded-session/latency_summary.md) are tracked copies of that session's export. The raw log and other development sessions remain ignored. Direct TTS samples are not recording-session evidence.

The export was generated from the confirmed raw log with the existing exporter. To reproduce it, use a **fresh** output folder; the exporter refuses to overwrite a nonempty folder:

```sh
uv run python -m rehearse_ai.export_session --session sessions/d564949d-6780-414f-8192-22bae9e9345d.jsonl --output recordings/d564949d-6780-414f-8192-22bae9e9345d-recheck
```

The raw local log is `sessions/d564949d-6780-414f-8192-22bae9e9345d.jsonl` and was not rewritten. The tracked evidence files match the original local export byte for byte. They preserve STT wording, including recognition errors. The video itself remains on Loom; no recording file is in the repository.

`user_stopped_ms` estimates the last speech time by subtracting the logged Silero VAD silence threshold from the VAD stop event. The assignment's `tts_first_audio_ms` field is labeled in each row as a **server output start proxy**; browser playback onset is unavailable. `llm_first_token_ms` remains null because Pipecat's structured inference call does not expose it. A response without LLM inference also has a null LLM request timestamp. `stt_final_basis` records whether STT was correlated by its logged turn or reconstructed from an older log's event order and complete accumulated text. Ambiguous or absent STT events leave `stt_final_ms` null; the summary counts ambiguous turns. Raw event order is never changed. The summary excludes interrupted/missing/reversed samples from the primary percentile and uses nearest rank; overlapping stage timestamps are never reported as negative processing durations. Review these approximations before quoting any result.

## Confirmed recording session

- Session ID: `d564949d-6780-414f-8192-22bae9e9345d`
- Build ID: `0.1.0+1b7eb61e63ad`
- Log time: 27 October 2026, 22:17:43–22:21:41 IST (3:57.9 of server events). Loom reports 4:32 for the video; these measure different spans.
- Recording link: [Rehearse AI Interview Coaching Session](https://www.loom.com/share/928bba7d68ec4174856488a0ed2e1bb7)
- Raw log: `sessions/d564949d-6780-414f-8192-22bae9e9345d.jsonl` (preserved unchanged).
- Tracked export: `evidence/recorded-session/` contains `transcript.txt`, `latency.jsonl`, and `latency_summary.md`.
- Normal answer: about **0:46**, verified in Loom's visible transcript as the first project answer.
- Mid-answer pause and continuation: **unverified**; no video timestamp recorded.
- Interruption while the agent is talking: **unverified**; no video timestamp recorded. The export marks no candidate turn as interrupted, and the raw log's two inference cancellations occurred before agent output, so this scenario is not established by the log.
- Four-question path: the log reaches all four main questions, gives provisional feedback on request, then records a spoken request to finish and an early sign-off. It does not demonstrate automatic normal completion or final feedback.
