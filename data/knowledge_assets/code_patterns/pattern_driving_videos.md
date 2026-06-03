---
pattern_id: pattern_driving_videos
applicable_symptoms: [driving_videos]
domain: Learning_Training
---

# VLN agents trained on static datasets fail to generalize to diverse outdoor scenes due to limited visual diversity.

**Domain**: `Learning_Training`

## Fix

Use driving videos to automatically generate navigation instructions and action labels for data augmentation.

## Anti-pattern

Training only on existing VLN datasets without real-world video augmentation.

## Cross-domain analogies

- **Perception_Vision** → Pretrain with cross-modal contrastive loss on diverse outdoor visual-language pairs.
  - related fix: Cross-modal alignment pretraining using contrastive or attention-based losses to align visual object features with language tokens.
- **Planning_Decision** → Use a unified multi-task architecture integrating diverse outdoor scene data and semantic querying.
  - related fix: Use the SG3D benchmark to evaluate navigation and question-answering performance, and adopt the MTU3D multi-task unified architecture that integrates mapping, planning, and obstacle avoidance into a single model.
- **Control_Locomotion** → Pre-train a diverse library of visual navigation behaviors via RL, decoupling perception from task planning.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- driving_videos.before.py
+++ driving_videos.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents trained on static datasets fail to generalize to diverse outdoor scenes due to limited visual diversity.

+# Fix    : Use driving videos to automatically generate navigation instructions and action labels for data augmentation.

+# Avoid  : Training only on existing VLN datasets without real-world video augmentation.

```
