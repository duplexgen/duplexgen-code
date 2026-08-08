# Stage 2: Slot identification

Detects candidate intra-utterance action points via heuristic + LLM boundary
detection.

Slot detection shares its implementation with Stage 4. The heuristic + LLM
boundary detector (`detect_turn_boundaries`) lives in `src/synthesis/core.py`
and runs as part of the [Stage 4](stage4-generation.md) invocation — there is no
separate command for this stage.

Next: [Stage 3](stage3-prediction.md).
