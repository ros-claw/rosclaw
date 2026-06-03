---
pattern_id: pattern_embodiedocc_scannet
applicable_symptoms: [embodiedocc_scannet]
domain: Perception_Vision
---

# Egocentric occupancy prediction models lack large-scale annotated datasets for training and evaluation, limiting sim-to-real transfer.

**Domain**: `Perception_Vision`

## Fix

EmbodiedOcc-ScanNet: a large-scale egocentric occupancy dataset derived from ScanNet with voxel-level occupancy labels from first-person perspective.

## Anti-pattern

Using only synthetic or small-scale datasets for occupancy prediction.

## Cross-domain analogies

- **Planning_Decision** → Use pretrained multimodal LLMs for zero-shot occupancy prediction via in-context learning.
  - related fix: Use a pretrained multimodal LLM (Gemini-Pro-Vision) as the navigation policy backbone, processing visual observations and language instructions jointly via in-context learning.
- **Learning_Training** → Augment synthetic occupancy data with realistic sensor artifacts and domain randomization to improve sim-to-real transfer.
  - related fix: Augment synthetic depth images with noise patterns (Gaussian blur, quantization artifacts, dropout) during training.
- **Control_Locomotion** → Distill diverse synthetic depth priors via DAgger and RL fine-tuning for occupancy prediction.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- embodiedocc_scannet.before.py
+++ embodiedocc_scannet.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Egocentric occupancy prediction models lack large-scale annotated datasets for training and evaluation, limiting sim-to-real transfer.

+# Fix    : EmbodiedOcc-ScanNet: a large-scale egocentric occupancy dataset derived from ScanNet with voxel-level occupancy labels from first-person perspective.

+# Avoid  : Using only synthetic or small-scale datasets for occupancy prediction.

```
