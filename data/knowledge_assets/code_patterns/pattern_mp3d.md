---
pattern_id: pattern_mp3d
applicable_symptoms: [mp3d]
domain: Planning_Decision
---

# Object-goal navigation agents fail to generalize across multi-floor buildings and unseen floor plans due to partial observability and long-horizon tasks.

**Domain**: `Planning_Decision`

## Fix

Use adaptive skill composition (e.g., ASCENT) that learns to combine navigation skills for different floors and rooms.

## Anti-pattern

Single flat policy trained on single-floor scenes without hierarchical structure.

## Cross-domain analogies

- **Perception_Vision** → Use cross-modal joint processing to fuse visual and semantic floor-plan cues for hierarchical long-horizon navigation.
  - related fix: Use a Vision-Language Model (VLM) that jointly processes visual and textual data for cross-modal reasoning, as in NavForesee.
- **Learning_Training** → Use self-occlusion-aware ray casting to generate multi-floor occupancy priors for long-horizon planning.
  - related fix: Synthetic depth image generation with self-occlusion-aware ray casting and noise-aware modeling calibrated from real depth sensor characteristics.
- **Control_Locomotion** → Distill multi-expert navigation policies using DAgger with depth-based memory for long-horizon floor plan generalization.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- mp3d.before.py
+++ mp3d.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Object-goal navigation agents fail to generalize across multi-floor buildings and unseen floor plans due to partial observability and long-horizon tasks.

+# Fix    : Use adaptive skill composition (e.g., ASCENT) that learns to combine navigation skills for different floors and rooms.

+# Avoid  : Single flat policy trained on single-floor scenes without hierarchical structure.

```
