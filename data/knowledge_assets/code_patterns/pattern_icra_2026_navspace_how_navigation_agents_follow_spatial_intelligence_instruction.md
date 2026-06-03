---
pattern_id: pattern_icra_2026_navspace_how_navigation_agents_follow_spatial_intelligence_instruction
applicable_symptoms: [icra_2026_navspace_how_navigation_agents_follow_spatial_intelligence_instructions]
domain: Planning_Decision
---

# VLN agents fail to follow spatial instructions that require reasoning about object relationships and spatial layouts in long-horizon navigation tasks.

**Domain**: `Planning_Decision`

## Fix

NavSpace benchmark with spatial intelligence instructions and evaluation metrics that test object-relationship and layout reasoning.

## Anti-pattern

Standard VLN benchmarks like R2R that focus on path following without explicit spatial reasoning.

## Cross-domain analogies

- **Perception_Vision** → Hierarchical decomposition with proximal-distal attention and temporal modeling for long-horizon spatial reasoning.
  - related fix: Hierarchical network with proximal sub-network for near-field features, distal sub-network for far-field context, recurrent temporal modeling, and attention-based fusion to output continuous risk scores.
- **Learning_Training** → Use synthetic trajectory generation to create training data for spatial reasoning tasks.
  - related fix: Train a speaker model to generate synthetic instruction–trajectory pairs from unannotated visual paths, then augment the original training set with these synthetic pairs.
- **Control_Locomotion** → Use lightweight hierarchical decomposition to compress spatial reasoning into low-latency local policies.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- icra_2026_navspace_how_navigation_agents_follow_spatial_intelligence_instructions.before.py
+++ icra_2026_navspace_how_navigation_agents_follow_spatial_intelligence_instructions.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to follow spatial instructions that require reasoning about object relationships and spatial layouts in long-horizon navigation tasks.

+# Fix    : NavSpace benchmark with spatial intelligence instructions and evaluation metrics that test object-relationship and layout reasoning.

+# Avoid  : Standard VLN benchmarks like R2R that focus on path following without explicit spatial reasoning.

```
