---
pattern_id: pattern_janusvln_decoupling_semantics_and_spatiality_with_dual_implicit_memory_for_visio
applicable_symptoms: [janusvln_decoupling_semantics_and_spatiality_with_dual_implicit_memory_for_visio]
domain: Planning_Decision
---

# VLN agents fail to decouple semantic and spatial cues, leading to poor navigation when instructions require both object recognition and geometric reasoning.

**Domain**: `Planning_Decision`

## Fix

Dual implicit memory with separate 2D semantic encoder (Qwen2.5-VL) and 3D spatial encoder (VGGT), updated via sliding window for dynamic incremental history.

## Anti-pattern

Single-stream MLLM that conflates semantics and spatiality in one representation.

## Cross-domain analogies

- **Perception_Vision** → Use a multi-scale feature pyramid to separately encode semantic and spatial cues at coarse and fine resolutions.
  - related fix: Use a coarse-to-fine pyramid (e.g., U-Net or FPN) that downsamples to capture coarse layout and upsamples to recover fine details, then fuse or sequentially feed multi-scale features.
- **Learning_Training** → Use synthetic data with decoupled semantic-spatial annotations to train disentangled representations.
  - related fix: Use large-scale synthetic data generation (e.g., ScaleVLN with 4M+ instructions) to augment training.
- **Control_Locomotion** → Use lightweight hierarchical decomposition to separate semantic and spatial processing streams.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- janusvln_decoupling_semantics_and_spatiality_with_dual_implicit_memory_for_visio.before.py
+++ janusvln_decoupling_semantics_and_spatiality_with_dual_implicit_memory_for_visio.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to decouple semantic and spatial cues, leading to poor navigation when instructions require both object recognition and geometric reasoning.

+# Fix    : Dual implicit memory with separate 2D semantic encoder (Qwen2.5-VL) and 3D spatial encoder (VGGT), updated via sliding window for dynamic incremental history.

+# Avoid  : Single-stream MLLM that conflates semantics and spatiality in one representation.

```
