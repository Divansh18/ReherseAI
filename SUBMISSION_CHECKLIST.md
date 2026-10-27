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
- Author-confirmed [Loom recording](https://www.loom.com/share/928bba7d68ec4174856488a0ed2e1bb7) matched to session `d564949d-6780-414f-8192-22bae9e9345d`; tracked [transcript](evidence/recorded-session/transcript.txt), [latency.jsonl](evidence/recorded-session/latency.jsonl), and [summary](evidence/recorded-session/latency_summary.md) preserve the export. `RECORDING.md` gives the scenario status and timing limitations.
- Recording-session latency is reported in `README.md`: 12 eligible turns, server-output p50 **2798.9 ms**, p95 **5007.3 ms**, with available stage averages and missing-field explanations. This is not measured browser playback.

## Pending before submission

- **Recording scenarios B and C are not established:** normal answering appears around 0:46 in Loom's visible transcript, but the mid-answer pause and agent-speech interruption have no verified video timestamps. The export marks no interrupted turn; two inference cancellations occurred before agent output. The brief requires all three scenarios in one 3–5 minute recording.
- **Normal automatic completion is not shown:** this recording reaches question four and provisional feedback, then ends through a spoken request. The brief asks for a real four-question conversation plus final feedback.
- Final human microphone/listening check of four-question completion, short pause, interruption, spoken stop, and closing feedback on the recorded build.
- Connect and push the requested public GitHub repository after verifying authentication and ownership; do not submit by email here.
- The brief lists **25 October 2026** as the deadline; the author must confirm submission timing or any extension. No message has been sent.
