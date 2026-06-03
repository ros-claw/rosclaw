---
pattern_id: pattern_neuro_symbolic_reasoning
applicable_symptoms: [neuro_symbolic_reasoning]
domain: Planning_Decision
---

# VLN agent ignores landmark cues in long instructions

**Domain**: `Planning_Decision`

## Fix

Neuro-symbolic reasoning: decompose high-level instructions into executable subgoals using symbolic reasoning, then guide exploration with neural networks

## Anti-pattern

End-to-end neural planning without explicit subgoal decomposition

## Cross-domain analogies

- **Perception_Vision** → Use closed-loop verification to confirm landmark attention before proceeding.
  - related fix: Prefer instruction design using landmarks that are visually salient and detectable by the agent's sensor suite (e.g., large objects, distinct colors).
- **Learning_Training** → Use closed-loop verification with full-kinematics simulation to enforce landmark grounding during instruction following.
  - related fix: Use full-kinematics agents with a robust physics engine to enable realistic motion dynamics and high-fidelity simulation, reducing sim-to-real gap.
- **Control_Locomotion** → Incorporate landmark cues as perceptual input to dynamically adjust navigation policy and waypoint selection.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- neuro_symbolic_reasoning.before.py
+++ neuro_symbolic_reasoning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent ignores landmark cues in long instructions

+# Fix    : Neuro-symbolic reasoning: decompose high-level instructions into executable subgoals using symbolic reasoning, then guide exploration with neural networks

+# Avoid  : End-to-end neural planning without explicit subgoal decomposition

```
