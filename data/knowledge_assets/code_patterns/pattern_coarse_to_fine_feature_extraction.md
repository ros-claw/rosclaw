---
pattern_id: pattern_coarse_to_fine_feature_extraction
applicable_symptoms: [coarse_to_fine_feature_extraction]
domain: Perception_Vision
---

# Single-scale feature extraction from volumetric grids loses either global layout or local detail, degrading navigation and manipulation performance.

**Domain**: `Perception_Vision`

## Fix

Use a coarse-to-fine pyramid (e.g., U-Net or FPN) that downsamples to capture coarse layout and upsamples to recover fine details, then fuse or sequentially feed multi-scale features.

## Anti-pattern

Using a single-resolution 3D CNN that either misses small obstacles or fails to capture long-range spatial structure.

## Cross-domain analogies

- **Planning_Decision** → Use multi-scale closed-loop verification to adaptively fuse global and local volumetric features.
  - related fix: Use a cooperative dialog framework where an oracle provides step-by-step guidance and the agent can ask clarifying questions, grounding instructions in real-time visual observations.
- **Learning_Training** → Use multi-scale feature extraction with a consistency penalty across scales to preserve both local and global information.
  - related fix: Functional regularisation: add a penalty on changes to the network's input-output mapping (e.g., using KL divergence or L2 distance on outputs) when training on new tasks.
- **Control_Locomotion** → Use multi-scale hierarchical feature extraction to balance global layout and local detail.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- coarse_to_fine_feature_extraction.before.py
+++ coarse_to_fine_feature_extraction.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Single-scale feature extraction from volumetric grids loses either global layout or local detail, degrading navigation and manipulation performance.

+# Fix    : Use a coarse-to-fine pyramid (e.g., U-Net or FPN) that downsamples to capture coarse layout and upsamples to recover fine details, then fuse or sequentially feed multi-scale features.

+# Avoid  : Using a single-resolution 3D CNN that either misses small obstacles or fails to capture long-range spatial structure.

```
