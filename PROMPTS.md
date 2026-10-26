# Runtime prompts

The exact text sent as the system instruction is loaded at runtime from the versioned files [decision.txt](src/rehearse_ai/prompts/decision.txt) and [feedback.txt](src/rehearse_ai/prompts/feedback.txt). This document does not duplicate their text.

The decision call receives the current fixed question, actual candidate answers, and the remaining Python-owned reply allowances as JSON data. Its structured response proposes `follow_up`, `nudge`, `clarification`, `wait`, or `advance` plus text. `wait` keeps an incomplete answer open without spoken output; an exhausted allowance never authorizes automatic advancement. Python validates the action and chooses the next main question. The feedback call receives the recorded answers and closing remarks; its structured response supplies 2–3 lines. Neither JSON response is sent directly to TTS. Candidate transcript content is treated as data, not instructions.

The initial decision prompt was used in the inspected real session. Its revision adds explicit remaining allowances and an incomplete-answer `wait` option. The revised prompt has only been exercised with mocked decisions; a comparative live prompt experiment is pending.
