---
pattern_id: pattern_vln_zero
applicable_symptoms: [vln_zero]
domain: Planning_Decision
---

# Zero-shot VLN agents suffer from high VLM call costs and slow navigation due to repeated reasoning in unfamiliar environments.

**Domain**: `Planning_Decision`

## Fix

Two-phase framework: rapid exploration builds symbolic scene graphs, then neurosymbolic planner reuses cached task-location trajectories for efficient deployment.

## Anti-pattern

Existing zero-shot models rely on frequent VLM calls for each navigation step, leading to high latency and computational cost.

## Cross-domain analogies

- **Perception_Vision** → Use simulated environment perturbations to pre-train efficient reasoning shortcuts, reducing costly online VLM calls.
  - related fix: Generate synthetic depth images with simulated sensor noise, self-occlusion, and lighting effects to match real sensor distribution.
- **Learning_Training** → Use self-supervised pseudo-label generation to pre-train navigation policies, reducing costly online VLM calls.
  - related fix: Use unsupervised adversarial training with self-supervised learning (e.g., rotation prediction) to generate pseudo-labels for robust training against adversarial perturbations.
- **Control_Locomotion** → Train a single end-to-end policy mapping visual inputs to actions, reducing costly VLM calls via sim-to-real transfer.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- vln_zero.before.py
+++ vln_zero.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Zero-shot VLN agents suffer from high VLM call costs and slow navigation due to repeated reasoning in unfamiliar environments.

+# Fix    : Two-phase framework: rapid exploration builds symbolic scene graphs, then neurosymbolic planner reuses cached task-location trajectories for efficient deployment.

+# Avoid  : Existing zero-shot models rely on frequent VLM calls for each navigation step, leading to high latency and computational cost.

```
