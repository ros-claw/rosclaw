---
pattern_id: pattern_closed_loop_reasoning
applicable_symptoms: [closed_loop_reasoning]
domain: Planning_Decision
---

# Open-loop exploration suffers from drift and map inconsistency due to perceptual aliasing and motion errors.

**Domain**: `Planning_Decision`

## Fix

Closed-loop reasoning: iteratively update belief state from real-time sensor feedback, evaluate actions, execute, observe, and update until task goal is achieved.

## Anti-pattern

Open-loop methods that execute a precomputed plan without adjustment.

## Cross-domain analogies

- **Perception_Vision** → Design exploration around perceptually distinctive landmarks to reduce drift and aliasing.
  - related fix: Prefer instruction design using landmarks that are visually salient and detectable by the agent's sensor suite (e.g., large objects, distinct colors).
- **Learning_Training** → Pre-train a world model on diverse trajectories, then fine-tune with closed-loop correction for drift.
  - related fix: Pre-train on large-scale image-text-action triplets via self-supervised learning, then fine-tune on downstream VLN tasks
- **Control_Locomotion** → Train an end-to-end policy with domain randomization to directly map noisy observations to actions, closing the loop.
  - related fix: Train a single end-to-end neural network policy via large-scale RL in simulation with domain randomization, mapping depth image directly to motor commands

## Patch

```diff
--- closed_loop_reasoning.before.py
+++ closed_loop_reasoning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Open-loop exploration suffers from drift and map inconsistency due to perceptual aliasing and motion errors.

+# Fix    : Closed-loop reasoning: iteratively update belief state from real-time sensor feedback, evaluate actions, execute, observe, and update until task goal is achieved.

+# Avoid  : Open-loop methods that execute a precomputed plan without adjustment.

```
