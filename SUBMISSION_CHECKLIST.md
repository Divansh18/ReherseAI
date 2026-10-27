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
- Author-confirmed [Loom recording](https://www.loom.com/share/928bba7d68ec4174856488a0ed2e1bb7) matched to local session `d564949d-6780-414f-8192-22bae9e9345d`; exported `transcript.txt`, `latency.jsonl`, and `latency_summary.md` are in the Git-ignored `recordings/d564949d-6780-414f-8192-22bae9e9345d/` folder. `RECORDING.md` gives the exact command and timing limitations.

## Pending before submission

- Verify the normal answer, mid-answer pause, and interruption in the Loom video; scenario timestamps are pending. The local log reaches all four questions but ends early after requested provisional feedback, so do not claim normal final feedback from this recording.
- Review and deliberately include the private, Git-ignored recording-session export in the final submission. It has not been committed or uploaded.
- Decide whether to cite the recording-session latency summary after reviewing the video and measurement limits. Its 12 eligible turns have server-output p50 **2798.9 ms** and p95 **5007.3 ms** (nearest rank); these are not browser playback times. First-token and browser-onset timestamps are unavailable.
- Author's **honest approximate active time spent** in `README.md`; Git history cannot establish it. Add any AI tools used beyond the known Codex assistance.
- Final human microphone/listening check of four-question completion, short pause, interruption, spoken stop, and closing feedback on the recorded build.
- Choose GitHub repository/access or a ZIP retaining `.git`, then submit it separately. No remote is configured and nothing was pushed by this preparation.
- The brief lists **25 October 2026** as the deadline; the author must confirm submission timing or any extension. No message has been sent.
