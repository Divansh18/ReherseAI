# Runtime prompts

The exact text sent as the system instruction is loaded at runtime from the versioned files [decision.txt](src/rehearse_ai/prompts/decision.txt) and [feedback.txt](src/rehearse_ai/prompts/feedback.txt). This document does not duplicate their text.

The decision call receives the current fixed question, actual candidate answers, and Python-owned reply limits as JSON data. Its structured response proposes `follow_up`, `nudge`, `clarification`, or `advance` plus text. Python validates the action and chooses the next main question. The feedback call receives the recorded answers and closing remarks; its structured response supplies 2–3 lines. Neither JSON response is sent directly to TTS. Candidate transcript content is treated as data, not instructions.

Prompt experiments: **pending**. No alternative prompt or live result has been tested yet.
