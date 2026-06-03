---
pattern_id: pattern_vln_bert
applicable_symptoms: [vln_bert]
domain: Learning_Training
---

# VLN agents struggle to generalize from static image-text pairs to dynamic navigation tasks, leading to poor path-instruction compatibility scoring.

**Domain**: `Learning_Training`

## Fix

Two-stage curriculum: pretrain on large-scale web-scraped image-text pairs (Conceptual Captions) then fine-tune on embodied path-instruction data.

## Anti-pattern

Directly training on embodied data without web pretraining yields lower VLN performance.

## Cross-domain analogies

- **Perception_Vision** → Use 3D-GS to generate dynamic trajectory-conditioned views from sparse real paths for training.
  - related fix: Construct high-fidelity datasets using 3D Gaussian Splatting (3D-GS) to generate photorealistic novel-view synthetic images from sparse real captures, preserving fine-grained textures and lighting details.
- **Planning_Decision** → Use closed-loop verification to score path-instruction compatibility from dynamic rollouts.
  - related fix: Self-improving embodied reasoning loop: collect failure trajectories, generate corrective reasoning via LLM self-reflection, and fine-tune the policy on augmented data.
- **Control_Locomotion** → Train an end-to-end policy mapping visual-linguistic inputs directly to navigation actions.
  - related fix: Train a single neural network policy via deep reinforcement learning that maps raw depth camera images directly to motor commands, bypassing hand-crafted perception and control layers.

## Patch

```diff
--- vln_bert.before.py
+++ vln_bert.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents struggle to generalize from static image-text pairs to dynamic navigation tasks, leading to poor path-instruction compatibility scoring.

+# Fix    : Two-stage curriculum: pretrain on large-scale web-scraped image-text pairs (Conceptual Captions) then fine-tune on embodied path-instruction data.

+# Avoid  : Directly training on embodied data without web pretraining yields lower VLN performance.

```
