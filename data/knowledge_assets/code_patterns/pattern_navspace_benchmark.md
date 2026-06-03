---
pattern_id: pattern_navspace_benchmark
applicable_symptoms: [navspace_benchmark]
domain: Planning_Decision
---

# Navigation agents fail to systematically evaluate spatial reasoning abilities like understanding environment state, space structure, precise movement, viewpoint shifting, vertical perception, and spatial relationships from instructions.

**Domain**: `Planning_Decision`

## Fix

Use NavSpace benchmark with 6 task categories (1,228 trajectory-instruction pairs) to isolate and measure spatial intelligence in instruction-following navigation agents.

## Anti-pattern

Existing VLN benchmarks do not systematically probe distinct spatial reasoning facets.

## Cross-domain analogies

- **Perception_Vision** → Dual-view prompt forces systematic spatial reasoning by integrating complementary perspectives into a single inference step.
  - related fix: Dual-view visual prompt: combine two complementary spatial views into a single prompt at inference time, applied on top of a VLA model.
- **Learning_Training** → Pretrain on diverse spatial reasoning tasks to learn systematic environment understanding, then fine-tune on navigation.
  - related fix: Pretrain a vision-language model on large-scale web data to learn cross-modal alignment, then fine-tune on navigation tasks
- **Control_Locomotion** → Multi-expert distillation with depth-based exteroception could enable systematic spatial reasoning by fusing multi-view perception with instruction-guided fine-tuning.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- navspace_benchmark.before.py
+++ navspace_benchmark.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation agents fail to systematically evaluate spatial reasoning abilities like understanding environment state, space structure, precise movement, viewpoint shifting, vertical perception, and spatial relationships from instructions.

+# Fix    : Use NavSpace benchmark with 6 task categories (1,228 trajectory-instruction pairs) to isolate and measure spatial intelligence in instruction-following navigation agents.

+# Avoid  : Existing VLN benchmarks do not systematically probe distinct spatial reasoning facets.

```
