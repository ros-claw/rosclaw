---
pattern_id: pattern_embodiedocc_scannet_benchmark
applicable_symptoms: [embodiedocc_scannet_benchmark]
domain: Perception_Vision
---

# Global occupancy benchmarks fail to evaluate egocentric, partial-observability prediction for embodied agents.

**Domain**: `Perception_Vision`

## Fix

Re-annotate ScanNet scenes with local occupancy grids aligned to the camera frame, supporting both static and temporal prediction tasks.

## Anti-pattern

Using global scene-level occupancy benchmarks (e.g., Semantic Scene Completion on NYUv2) for embodied evaluation.

## Cross-domain analogies

- **Planning_Decision** → Use multi-modal diffusion to generate egocentric occupancy predictions conditioned on partial observations.
  - related fix: Use a Diffusion Transformer policy with multi-modal conditioning (pixel goals + latent features) as System 1 in a dual-system architecture to generate smooth, continuous trajectories in real time.
- **Learning_Training** → Apply multi-scale dropout to occupancy predictions, masking spatial regions to prevent overfitting to global benchmarks.
  - related fix: Apply dropout operations at multiple feature scales (activations, channels, spatial regions, entire feature maps) to regularize training.
- **Control_Locomotion** → Use a separate safety-critic policy to override global benchmarks with egocentric risk thresholds.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- embodiedocc_scannet_benchmark.before.py
+++ embodiedocc_scannet_benchmark.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Global occupancy benchmarks fail to evaluate egocentric, partial-observability prediction for embodied agents.

+# Fix    : Re-annotate ScanNet scenes with local occupancy grids aligned to the camera frame, supporting both static and temporal prediction tasks.

+# Avoid  : Using global scene-level occupancy benchmarks (e.g., Semantic Scene Completion on NYUv2) for embodied evaluation.

```
