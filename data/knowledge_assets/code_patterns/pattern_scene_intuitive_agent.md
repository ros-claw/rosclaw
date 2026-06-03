---
pattern_id: pattern_scene_intuitive_agent
applicable_symptoms: [scene_intuitive_agent]
domain: Planning_Decision
---

# Embodied agents fail to navigate to remote target objects specified by high-level instructions because they cannot associate global scene semantics with local object features.

**Domain**: `Planning_Decision`

## Fix

Two-stage training: first learn scene grounding (global visual-language alignment), then fine-tune object grounding (local noun-phrase alignment), using a memory-augmented attentive action decoder with episodic buffer and cross-modal attention.

## Anti-pattern

Single-stage training without explicit separation of scene and object grounding leads to poor generalization to unseen targets.

## Cross-domain analogies

- **Perception_Vision** → Use panoramic ray properties to constrain local feature sampling with global semantic geometry.
  - related fix: Apply spherical geometry-aware constraints to regularize sampling offsets, leveraging panoramic ray properties for distortion-aware alignment without explicit undistortion.
- **Learning_Training** → Use a discriminator to distinguish global semantic coherence from local feature plans, training the planner to fool it.
  - related fix: Adversarial Variational Autoencoder (aVAE) combining VAE with adversarial training: use a discriminator to distinguish real data from VAE-generated samples, and train the VAE encoder/decoder to fool the discriminator.
- **Control_Locomotion** → Use hierarchical decomposition to resolve global semantics into local action sequences.
  - related fix: Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Patch

```diff
--- scene_intuitive_agent.before.py
+++ scene_intuitive_agent.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Embodied agents fail to navigate to remote target objects specified by high-level instructions because they cannot associate global scene semantics with local object features.

+# Fix    : Two-stage training: first learn scene grounding (global visual-language alignment), then fine-tune object grounding (local noun-phrase alignment), using a memory-augmented attentive action decoder with episodic buffer and cross-modal attention.

+# Avoid  : Single-stage training without explicit separation of scene and object grounding leads to poor generalization to unseen targets.

```
