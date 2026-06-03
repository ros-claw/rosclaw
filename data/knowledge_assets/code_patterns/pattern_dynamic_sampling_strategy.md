---
pattern_id: pattern_dynamic_sampling_strategy
applicable_symptoms: [dynamic_sampling_strategy]
domain: Systems_Compute
---

# Inference-time token budget is exceeded when processing observations, causing computational overload or violating deployment constraints.

**Domain**: `Systems_Compute`

## Fix

Dynamically adjust the number of observation tokens at runtime under a token length budget, discarding or compressing less informative tokens to stay within budget.

## Anti-pattern

Using a fixed number of tokens per observation regardless of budget constraints.

## Cross-domain analogies

- **Perception_Vision** → Use synthetic token pruning guided by a text-to-image prior to compress observations.
  - related fix: Generate synthetic images from landmark text descriptions via a text-to-image diffusion model, and train the agent with an auxiliary grounding loss that aligns instruction representations with imagination embeddings
- **Planning_Decision** → Train agents on constrained token budgets to remove reliance on unlimited compute.
  - related fix: Train and evaluate agents in continuous environments with raw egocentric observations, uncertain localization, and fine-grained motor control, removing the graph assumption.
- **Learning_Training** → Calibrate observation token budget via occlusion-aware selective sampling mimicking depth noise modeling.
  - related fix: Synthetic depth image generation with self-occlusion-aware ray casting and noise-aware modeling calibrated from real depth sensor characteristics.

## Patch

```diff
--- dynamic_sampling_strategy.before.py
+++ dynamic_sampling_strategy.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Inference-time token budget is exceeded when processing observations, causing computational overload or violating deployment constraints.

+# Fix    : Dynamically adjust the number of observation tokens at runtime under a token length budget, discarding or compressing less informative tokens to stay within budget.

+# Avoid  : Using a fixed number of tokens per observation regardless of budget constraints.

```
