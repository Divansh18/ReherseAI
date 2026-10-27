# Capture the submission evidence

A [Loom recording](https://www.loom.com/share/928bba7d68ec4174856488a0ed2e1bb7) has been supplied, and the author confirmed its matching local session ID below. The export exists locally under the Git-ignored `recordings/` directory. The normal-answer, pause, and interruption scenarios still need verification against the video; direct TTS samples are not recording-session evidence.

1. Run `uv sync --frozen --group dev`, configure `.env` as described in `README.md`, then start `uv run python -m rehearse_ai`.
2. Record a **single 3–5 minute screen session with audio**. Show a normal project answer, a mid-answer pause (continue the same answer), and an interruption while RehearseAI speaks. Finish all four questions and feedback if the session behaves well. Keep the build ID visible in browser Diagnostics or note it separately.
3. Save the session UUID shown in Diagnostics. The raw local log is `sessions/<SESSION_ID>.jsonl`; do not edit it. Note the recording link and scenario timestamps below only after the real recording exists.
4. Export the **same** session into a fresh, explicit local folder (replace the UUID in both paths):

   ```sh
   uv run python -m rehearse_ai.export_session --session sessions/<SESSION_ID>.jsonl --output recordings/<SESSION_ID>
   ```

5. Review `recordings/<SESSION_ID>/transcript.txt`, `latency.jsonl`, and `latency_summary.md` against the video. The exporter refuses a nonempty output folder and never rewrites the raw log. `recordings/` is Git-ignored because transcript/recording content is private; deliberately add only the approved evidence when submitting.

`user_stopped_ms` estimates the last speech time by subtracting the logged Silero VAD silence threshold from the VAD stop event. The assignment's `tts_first_audio_ms` field is labeled in each row as a **server output start proxy**; browser playback onset is unavailable. `llm_first_token_ms` remains null because Pipecat's structured inference call does not expose it. A response without LLM inference also has a null LLM request timestamp. `stt_final_basis` records whether STT was correlated by its logged turn or reconstructed from an older log's event order and complete accumulated text. Ambiguous or absent STT events leave `stt_final_ms` null; the summary counts ambiguous turns. Raw event order is never changed. The summary excludes interrupted/missing/reversed samples from the primary percentile and uses nearest rank; overlapping stage timestamps are never reported as negative processing durations. Review these approximations before quoting any result.

## Confirmed recording session

- Session ID: `d564949d-6780-414f-8192-22bae9e9345d`
- Build ID: `0.1.0+1b7eb61e63ad`
- Log time: 27 October 2026, 22:17:43–22:21:41 IST (3:57.9 of server events). Loom reports 4:32 for the video; these measure different spans.
- Recording link: [Rehearse AI Interview Coaching Session](https://www.loom.com/share/928bba7d68ec4174856488a0ed2e1bb7)
- Raw log: `sessions/d564949d-6780-414f-8192-22bae9e9345d.jsonl` (preserved unchanged).
- Local export: `recordings/d564949d-6780-414f-8192-22bae9e9345d/` contains `transcript.txt`, `latency.jsonl`, and `latency_summary.md`. These private files are ignored by Git and have not been submitted.
- Exact export command: `uv run python -m rehearse_ai.export_session --session sessions/d564949d-6780-414f-8192-22bae9e9345d.jsonl --output recordings/d564949d-6780-414f-8192-22bae9e9345d`
- Normal answer timestamp: pending
- Mid-answer pause timestamp: pending
- Interruption timestamp: pending
- Four-question path: the log reaches all four main questions, gives provisional feedback on request, then records a spoken request to finish and an early sign-off. Normal final feedback is not established by this log; verify the video before describing its scenarios.
