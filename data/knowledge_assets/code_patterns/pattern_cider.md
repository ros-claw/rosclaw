---
pattern_id: pattern_cider
applicable_symptoms: [cider]
domain: Planning_Decision
---

# CIDEr metric fails to evaluate grounded navigation instructions because it ignores spatial reasoning and action grounding.

**Domain**: `Planning_Decision`

## Fix

Use ET (Embodied navigation Instruction Evaluation) metric that accounts for spatial alignment and action grounding.

## Anti-pattern

Using CIDEr for grounded navigation instruction evaluation.

## Cross-domain analogies

- **Perception_Vision** → Augment training data with synthetic spatial-reasoning and action-grounded variants to match real evaluation distribution.
  - related fix: Generate synthetic depth images with simulated sensor noise, self-occlusion, and lighting effects to match real sensor distribution.
- **Learning_Training** → Closed-loop verification enforces spatial grounding by reconstructing paths from instructions.
  - related fix: Train a Transformer-based Speaker jointly with a Listener in a Double Back-Translation loop, where the Speaker generates instructions from paths and the Listener reconstructs paths from instructions, enforcing instruction-path consistency through iterative refinement.
- **Control_Locomotion** → Use reinforcement learning to map visual observations directly to navigation actions, bypassing metric limitations.
  - related fix: Use reinforcement learning to learn a control policy that directly maps sensor observations to actuator commands for plasma shape and position control.

## Patch

```diff
--- cider.before.py
+++ cider.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: CIDEr metric fails to evaluate grounded navigation instructions because it ignores spatial reasoning and action grounding.

+# Fix    : Use ET (Embodied navigation Instruction Evaluation) metric that accounts for spatial alignment and action grounding.

+# Avoid  : Using CIDEr for grounded navigation instruction evaluation.

```
