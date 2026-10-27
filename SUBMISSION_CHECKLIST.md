# Submission checklist

The assignment brief is in the local `takehome-voice-agent.zip`. This checklist records repository state; it does not mark the take-home submission complete.

## Ready

- Pipecat 1.14.0 Python cascade with separate Deepgram STT, OpenAI text LLM, and Cartesia TTS: `src/rehearse_ai/pipeline.py`, `src/rehearse_ai/tts.py`, `src/rehearse_ai/session.py`.
- Four ordered questions and bounded delivery-aware interview controller: `src/rehearse_ai/interview.py`; tests in `tests/`.
- Incremental Git history: 23 feature/fix commits preceded this preparation; inspect with `git log --oneline`.
- Reproducible dependency definition and lock: `pyproject.toml`, `uv.lock`, `.python-version`.
- Configuration template without keys: `.env.example`; single start command and architecture: `README.md`.
- Exact runtime prompt files: `src/rehearse_ai/prompts/*.txt`; explanations and genuine prompt iterations: `PROMPTS.md`.
- Raw per-session timing and transcript instrumentation: `src/rehearse_ai/session_log.py`, `src/rehearse_ai/session.py`, `src/rehearse_ai/pipeline.py`.
- Evidence export command and synthetic checks: `src/rehearse_ai/export_session.py`, `tests/test_export_session.py`, `RECORDING.md`.
- Direct-provider diagnostic data, explicitly separate from end-to-end latency: `TTS_COMPARISON.md`, `samples/tts-comparison-2026-10-26/measurements.json`.

## Pending before submission

- **Real 3–5 minute screen recording with audio** of one session containing a normal answer, mid-answer pause, and interruption. Add a real link and scenario timestamps to `RECORDING.md` after capture.
- **`transcript.txt` and `latency.jsonl` from that same recorded session**, exported with the command in `RECORDING.md`, reviewed, and deliberately added to the submission. Raw development sessions are not substitutes.
- **Recording-session latency p50/p95 and stage averages**, copied from its `latency_summary.md` with the documented server-output proxy and missing first-token/browser-playback fields. No recording-session result is claimed yet.
- Author's **honest approximate active time spent** in `README.md`; Git history cannot establish it. Add any AI tools used beyond the known Codex assistance.
- Final human microphone/listening check of four-question completion, short pause, interruption, spoken stop, and closing feedback on the recorded build.
- Choose GitHub repository/access or a ZIP retaining `.git`, then submit it separately. No remote is configured and nothing was pushed by this preparation.
- The brief lists **25 October 2026** as the deadline; the author must confirm submission timing or any extension. No message has been sent.
