---
pattern_id: pattern_mid_term_waypoint_goals
applicable_symptoms: [mid_term_waypoint_goals]
domain: Planning_Decision
---

# Long-horizon navigation tasks require full path precomputation, which is computationally expensive and brittle to environmental changes.

**Domain**: `Planning_Decision`

## Fix

Use a VLM global planner to predict sparse mid-term waypoint goals (x, y, θ) that decompose the task into manageable sub-goals, refined by a local planner.

## Anti-pattern

Precomputing a dense global path that must be replanned entirely when the environment changes.

## Cross-domain analogies

- **Perception_Vision** → Use shared planning layers with task-specific encoders for joint path embedding, avoiding full precomputation.
  - related fix: Use a multimodal versatile network (MMV) with shared transformer layers and modality-specific encoders to learn joint embeddings across modalities.
- **Learning_Training** → Pretrain a hierarchical planner on offline data, then fine-tune online for adaptive long-horizon navigation.
  - related fix: Pretrain a vision-language model on large-scale web data to learn cross-modal alignment, then fine-tune on navigation tasks
- **Control_Locomotion** → Distill offline planning into closed-loop reactive policies with online depth-based adaptation.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- mid_term_waypoint_goals.before.py
+++ mid_term_waypoint_goals.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Long-horizon navigation tasks require full path precomputation, which is computationally expensive and brittle to environmental changes.

+# Fix    : Use a VLM global planner to predict sparse mid-term waypoint goals (x, y, θ) that decompose the task into manageable sub-goals, refined by a local planner.

+# Avoid  : Precomputing a dense global path that must be replanned entirely when the environment changes.

```
