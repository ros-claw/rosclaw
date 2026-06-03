---
pattern_id: pattern_advisor
applicable_symptoms: [advisor]
domain: Planning_Decision
---

# Fixed replanning thresholds or timeouts cause either excessive replanning overhead or failure to adapt to unexpected changes in long-horizon tasks.

**Domain**: `Planning_Decision`

## Fix

Use an LLM-based Advisor module that continuously monitors system state and task progress, evaluating contextual cues (e.g., unexpected sensor readings, partial failures) to issue a replanning request only when necessary.

## Anti-pattern

Using fixed thresholds or timeouts to trigger replanning.

## Cross-domain analogies

- **Perception_Vision** → Use learned adaptive thresholds from simulated data to trigger replanning only when necessary.
  - related fix: Use deep learning models (e.g., CNNs) trained on simulated galaxy merger images to automatically classify merger stages.
- **Learning_Training** → Use adaptive threshold scheduling based on synthetic replanning cost-benefit data.
  - related fix: Use Marky to programmatically generate 4.2 million synthetic instruction–trajectory pairs from structured environment representations and action sequences.
- **Control_Locomotion** → Replace fixed thresholds with a learned policy that adapts replanning frequency based on sensed task state.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- advisor.before.py
+++ advisor.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Fixed replanning thresholds or timeouts cause either excessive replanning overhead or failure to adapt to unexpected changes in long-horizon tasks.

+# Fix    : Use an LLM-based Advisor module that continuously monitors system state and task progress, evaluating contextual cues (e.g., unexpected sensor readings, partial failures) to issue a replanning request only when necessary.

+# Avoid  : Using fixed thresholds or timeouts to trigger replanning.

```
