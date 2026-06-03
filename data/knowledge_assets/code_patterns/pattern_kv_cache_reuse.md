---
pattern_id: pattern_kv_cache_reuse
applicable_symptoms: [kv_cache_reuse]
domain: Memory_Reasoning
---

# Multi-turn dialogue in long video streams causes unbounded context size and high inference cost due to full recomputation of long sequences.

**Domain**: `Memory_Reasoning`

## Fix

Reuse key-value (KV) caches from previous turns to avoid full recomputation, maintaining bounded context size and controlled inference cost.

## Anti-pattern

Full recomputation of long sequences for each turn.

## Cross-domain analogies

- **Perception_Vision** → Use a generative model with a compressed context loss to directly predict responses from truncated memory in one pass.
  - related fix: Use a GAN with a compressed sensing loss to directly reconstruct images from undersampled k-space data in a single feedforward pass.
- **Planning_Decision** → Use attention to align compressed video tokens with dialogue history, enabling subgoal-level context pruning.
  - related fix: Cross-modal transformer with attention aligning topological map features and language instruction embeddings to output subgoal sequences.
- **Learning_Training** → Distill long video context into compact memory tokens via privileged guidance loss.
  - related fix: Privileged Information Guidance (PIG): train a diffusion policy with privileged depth and collision information during training, then distill into a student policy that uses only RGB observations via a guidance loss.

## Patch

```diff
--- kv_cache_reuse.before.py
+++ kv_cache_reuse.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Multi-turn dialogue in long video streams causes unbounded context size and high inference cost due to full recomputation of long sequences.

+# Fix    : Reuse key-value (KV) caches from previous turns to avoid full recomputation, maintaining bounded context size and controlled inference cost.

+# Avoid  : Full recomputation of long sequences for each turn.

```
