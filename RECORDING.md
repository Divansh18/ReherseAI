# Capture the submission evidence

The recording and its matching export are **pending**. Development logs and direct TTS samples are not submission-session evidence.

1. Run `uv sync --frozen --group dev`, configure `.env` as described in `README.md`, then start `uv run python -m rehearse_ai`.
2. Record a **single 3–5 minute screen session with audio**. Show a normal project answer, a mid-answer pause (continue the same answer), and an interruption while RehearseAI speaks. Finish all four questions and feedback if the session behaves well. Keep the build ID visible in browser Diagnostics or note it separately.
3. Save the session UUID shown in Diagnostics. The raw local log is `sessions/<SESSION_ID>.jsonl`; do not edit it. Note the recording link and scenario timestamps below only after the real recording exists.
4. Export the **same** session into a fresh, explicit local folder (replace the UUID in both paths):

   ```sh
   uv run python -m rehearse_ai.export_session --session sessions/<SESSION_ID>.jsonl --output recordings/<SESSION_ID>
   ```

5. Review `recordings/<SESSION_ID>/transcript.txt`, `latency.jsonl`, and `latency_summary.md` against the video. The exporter refuses a nonempty output folder and never rewrites the raw log. `recordings/` is Git-ignored because transcript/recording content is private; deliberately add only the approved evidence when submitting.

`user_stopped_ms` estimates the last speech time by subtracting the logged Silero VAD silence threshold from the VAD stop event. The assignment's `tts_first_audio_ms` field is labeled in each row as a **server output start proxy**; browser playback onset is unavailable. `llm_first_token_ms` remains null because Pipecat's structured inference call does not expose it. A response without LLM inference also has a null LLM request timestamp. `stt_final_basis` records whether STT was correlated by its logged turn or reconstructed from an older log's event order and complete accumulated text. Ambiguous or absent STT events leave `stt_final_ms` null; the summary counts ambiguous turns. Raw event order is never changed. The summary excludes interrupted/missing/reversed samples from the primary percentile and uses nearest rank; overlapping stage timestamps are never reported as negative processing durations. Review these approximations before quoting any result.

## Fill in after capture

- Session ID: pending
- Build ID: pending
- Recording link: pending
- Normal answer timestamp: pending
- Mid-answer pause timestamp: pending
- Interruption timestamp: pending
- Four-question completion and feedback observed: pending
