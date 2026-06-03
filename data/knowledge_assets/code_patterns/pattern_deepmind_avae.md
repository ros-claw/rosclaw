---
pattern_id: pattern_deepmind_avae
applicable_symptoms: [deepmind_avae]
domain: Learning_Training
---

# Standard VAEs produce blurry samples and fail to capture sharp, realistic data distributions.

**Domain**: `Learning_Training`

## Fix

Adversarial Variational Autoencoder (aVAE) combining VAE with adversarial training: use a discriminator to distinguish real data from VAE-generated samples, and train the VAE encoder/decoder to fool the discriminator.

## Anti-pattern

Standard VAE with only KL-divergence and reconstruction loss yields blurry outputs.

## Cross-domain analogies

- **Perception_Vision** → Integrate learned semantic representations into VAE latent space to enforce sharp, realistic feature reconstruction.
  - related fix: VISR: a framework integrating visual perception with semantic reasoning using learned representations
- **Planning_Decision** → Use adaptive latent channel gating to switch between coarse and fine-grained decoding based on reconstruction complexity.
  - related fix: Adaptive Chain-of-Thought mechanism that dynamically switches between fast reactive (System 1) and slow deliberative (System 2) reasoning based on task complexity.
- **Control_Locomotion** → Train a VAE with domain randomization and sim-to-real transfer to map noisy latents to sharp outputs.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- deepmind_avae.before.py
+++ deepmind_avae.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Standard VAEs produce blurry samples and fail to capture sharp, realistic data distributions.

+# Fix    : Adversarial Variational Autoencoder (aVAE) combining VAE with adversarial training: use a discriminator to distinguish real data from VAE-generated samples, and train the VAE encoder/decoder to fool the discriminator.

+# Avoid  : Standard VAE with only KL-divergence and reconstruction loss yields blurry outputs.

```
