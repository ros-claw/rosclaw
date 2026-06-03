---
pattern_id: pattern_neurosymbolic_planning
applicable_symptoms: [neurosymbolic_planning]
domain: Planning_Decision
---

# VLN agents fail to generalize to unseen environments without task-specific training data

**Domain**: `Planning_Decision`

## Fix

Neurosymbolic planner: neural perception + symbolic reasoning over scene graphs to generate executable navigation plans from natural language instructions

## Anti-pattern

End-to-end neural methods that require environment-specific fine-tuning

## Cross-domain analogies

- **Perception_Vision** → Use multi-sensor fusion to combine visual and non-visual cues for robust navigation without task-specific data.
  - related fix: Active mitigation strategies such as multi-sensor fusion or predictive reacquisition to handle blind zones
- **Learning_Training** → Use concatenated diverse sub-trajectories to create synthetic training data for unseen environments.
  - related fix: Use R4R dataset (concatenated R2R paths) to create longer, circuitous trajectories that better differentiate instruction-following agents from goal-seeking ones.
- **Control_Locomotion** → Train an end-to-end policy with domain-randomized visual inputs to generalize without task-specific data.
  - related fix: Train a single end-to-end neural network policy via large-scale RL in simulation with domain randomization, mapping depth image directly to motor commands

## Patch

```diff
--- neurosymbolic_planning.before.py
+++ neurosymbolic_planning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to generalize to unseen environments without task-specific training data

+# Fix    : Neurosymbolic planner: neural perception + symbolic reasoning over scene graphs to generate executable navigation plans from natural language instructions

+# Avoid  : End-to-end neural methods that require environment-specific fine-tuning

```
