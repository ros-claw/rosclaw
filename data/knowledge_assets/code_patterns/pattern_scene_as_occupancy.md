---
pattern_id: pattern_scene_as_occupancy
applicable_symptoms: [scene_as_occupancy]
domain: Perception_Vision
---

# 3D scene understanding from 2D images fails to capture full geometry and semantics for autonomous driving.

**Domain**: `Perception_Vision`

## Fix

Learn an occupancy network that predicts 3D occupancy and semantics from multi-camera images using a transformer-based 3D decoder.

## Anti-pattern

Traditional 2D-to-3D lifting with depth estimation and voxel projection loses information due to depth ambiguity.

## Cross-domain analogies

- **Planning_Decision** → Use multi-modal diffusion conditioning to fuse 2D images with latent 3D features for continuous geometry-semantic reconstruction.
  - related fix: Use a Diffusion Transformer policy with multi-modal conditioning (pixel goals + latent features) as System 1 in a dual-system architecture to generate smooth, continuous trajectories in real time.
- **Learning_Training** → Use adversarial training to refine 3D geometry and semantics from 2D images.
  - related fix: Adversarial Variational Autoencoder (aVAE) combining VAE with adversarial training: use a discriminator to distinguish real data from VAE-generated samples, and train the VAE encoder/decoder to fool the discriminator.
- **Control_Locomotion** → Incorporate geometric priors (depth/lidar) into vision policy to dynamically adapt scene parsing and reconstruction.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- scene_as_occupancy.before.py
+++ scene_as_occupancy.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: 3D scene understanding from 2D images fails to capture full geometry and semantics for autonomous driving.

+# Fix    : Learn an occupancy network that predicts 3D occupancy and semantics from multi-camera images using a transformer-based 3D decoder.

+# Avoid  : Traditional 2D-to-3D lifting with depth estimation and voxel projection loses information due to depth ambiguity.

```
