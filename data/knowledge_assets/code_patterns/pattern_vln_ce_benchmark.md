---
pattern_id: pattern_vln_ce_benchmark
applicable_symptoms: [vln_ce_benchmark]
domain: Planning_Decision
---

# VLN agents trained on discrete grids fail in continuous 3D spaces due to unrealistic action spaces and lack of continuous trajectory planning.

**Domain**: `Planning_Decision`

## Fix

Use VLN-CE benchmark with continuous action spaces, realistic 3D environments, and metrics like success rate, navigation error, and path length to evaluate and compare continuous VLN agents.

## Anti-pattern

Evaluating VLN agents only in discrete grid-based environments.

## Cross-domain analogies

- **Perception_Vision** → Active perception with semantic mapping inspires continuous viewpoint sampling to resolve action-space ambiguity.
  - related fix: Integrate active perception with semantic mapping: agent selects viewpoints to reduce ambiguity while building a task-driven semantic map from RGB-D or lidar data.
- **Learning_Training** → Augment continuous action spaces with synthetic trajectory data from 3D scene graphs and LLMs.
  - related fix: ScaleVLN: large-scale synthetic data generation by combining 3D scene graphs with LLM-generated instructions and augmenting with panoramic views and object-level grounding.
- **Control_Locomotion** → Train an end-to-end policy with domain randomization to map noisy 3D observations directly to continuous trajectories.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- vln_ce_benchmark.before.py
+++ vln_ce_benchmark.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents trained on discrete grids fail in continuous 3D spaces due to unrealistic action spaces and lack of continuous trajectory planning.

+# Fix    : Use VLN-CE benchmark with continuous action spaces, realistic 3D environments, and metrics like success rate, navigation error, and path length to evaluate and compare continuous VLN agents.

+# Avoid  : Evaluating VLN agents only in discrete grid-based environments.

```
