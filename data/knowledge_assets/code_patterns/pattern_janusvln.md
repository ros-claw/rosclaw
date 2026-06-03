---
pattern_id: pattern_janusvln
applicable_symptoms: [janusvln]
domain: Memory_Reasoning
---

# VLN agents using RGB-only input struggle with spatial reasoning, leading to poor navigation success compared to methods that fuse multiple sensor modalities (depth, semantic maps).

**Domain**: `Memory_Reasoning`

## Fix

Dual implicit neural memory: maintain separate key-value caches for spatial-geometric encoder (3D priors) and visual-semantic encoder, retaining only initial tokens and sliding window tokens for efficient incremental updates.

## Anti-pattern

Fusing multiple input data types (RGB + depth + segmentation) or using RGB-only with full token retention.

## Cross-domain analogies

- **Perception_Vision** → Fuse visually salient landmarks with depth cues to enhance spatial reasoning.
  - related fix: Prefer instruction design using landmarks that are visually salient and detectable by the agent's sensor suite (e.g., large objects, distinct colors).
- **Planning_Decision** → Use learned visual embeddings to implicitly encode spatial cues, reducing need for explicit multi-sensor fusion.
  - related fix: Use a learned visual representation (e.g., Siamese network or contrastive embedding) to match current observation to goal image, combined with an exploration policy (e.g., frontier-based or RL) that moves to reduce embedding distance.
- **Learning_Training** → Use self-supervised pseudo-label generation from RGB to infer spatial features, mimicking depth/semantic modalities.
  - related fix: Use unsupervised adversarial training with self-supervised learning (e.g., rotation prediction) to generate pseudo-labels for robust training against adversarial perturbations.

## Patch

```diff
--- janusvln.before.py
+++ janusvln.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents using RGB-only input struggle with spatial reasoning, leading to poor navigation success compared to methods that fuse multiple sensor modalities (depth, semantic maps).

+# Fix    : Dual implicit neural memory: maintain separate key-value caches for spatial-geometric encoder (3D priors) and visual-semantic encoder, retaining only initial tokens and sliding window tokens for efficient incremental updates.

+# Avoid  : Fusing multiple input data types (RGB + depth + segmentation) or using RGB-only with full token retention.

```
