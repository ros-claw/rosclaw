---
pattern_id: pattern_decoupling_training
applicable_symptoms: [decoupling_training]
domain: Learning_Training
---

# End-to-end training of a VLM with a navigation policy causes catastrophic forgetting of the VLM's broad semantic knowledge.

**Domain**: `Learning_Training`

## Fix

Train System 1 (VLM) and System 2 (local navigation policy) separately: freeze or fine-tune the VLM on high-level tasks, and train the navigation policy via RL or IL on environment-specific interactions.

## Anti-pattern

End-to-end training of VLM and navigation policy jointly.

## Cross-domain analogies

- **Perception_Vision** → Use multi-scale feature distillation with a coarse-to-fine pyramid to retain broad semantics while learning navigation.
  - related fix: Use a coarse-to-fine pyramid (e.g., U-Net or FPN) that downsamples to capture coarse layout and upsamples to recover fine details, then fuse or sequentially feed multi-scale features.
- **Planning_Decision** → Use learned neural components to selectively fine-tune the VLM while freezing its core semantic knowledge.
  - related fix: Use learned neural components (e.g., branching policies, primal heuristics) to accelerate MIP solving.
- **Control_Locomotion** → Closed-loop verification of semantic memory against local policy gradients prevents catastrophic forgetting.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- decoupling_training.before.py
+++ decoupling_training.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: End-to-end training of a VLM with a navigation policy causes catastrophic forgetting of the VLM's broad semantic knowledge.

+# Fix    : Train System 1 (VLM) and System 2 (local navigation policy) separately: freeze or fine-tune the VLM on high-level tasks, and train the navigation policy via RL or IL on environment-specific interactions.

+# Avoid  : End-to-end training of VLM and navigation policy jointly.

```
