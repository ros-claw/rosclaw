---
pattern_id: pattern_deepmind_perceiver
applicable_symptoms: [deepmind_perceiver]
domain: Perception_Vision
---

# Standard transformers scale quadratically with input size, making them impractical for high-dimensional multimodal data like images, video, and point clouds.

**Domain**: `Perception_Vision`

## Fix

Use a cross-attention bottleneck: project arbitrary input to a fixed-size latent array via cross-attention, then process with iterative self-attention in latent space.

## Anti-pattern

Applying standard transformer directly to raw pixels or high-dimensional inputs without a bottleneck.

## Cross-domain analogies

- **Planning_Decision** → Fine-tune a pretrained model on compressed simulated experience to reduce input dimensionality.
  - related fix: Fine-tune a pre-trained VLM on simulated embodied experience to act as a navigation policy, conditioning on visual history and goals (as in FiLM-Nav).
- **Learning_Training** → Use pretrained vision-language models for hierarchical spatial decomposition to reduce transformer input dimensions.
  - related fix: Use Marky, a multilingual instruction generator that produces visually grounded instruction-trajectory pairs at scale (4.2M pairs) by leveraging pretrained vision-language models and spatial alignment.
- **Control_Locomotion** → Hierarchical decomposition with local attention windows guided by global context.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- deepmind_perceiver.before.py
+++ deepmind_perceiver.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Standard transformers scale quadratically with input size, making them impractical for high-dimensional multimodal data like images, video, and point clouds.

+# Fix    : Use a cross-attention bottleneck: project arbitrary input to a fixed-size latent array via cross-attention, then process with iterative self-attention in latent space.

+# Avoid  : Applying standard transformer directly to raw pixels or high-dimensional inputs without a bottleneck.

```
