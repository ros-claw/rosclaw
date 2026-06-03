---
pattern_id: pattern_deformable_cross_attention
applicable_symptoms: [deformable_cross_attention]
domain: Perception_Vision
---

# Standard cross-attention over full image feature grids is computationally expensive and inefficient for integrating visual information into 3D Gaussian representations.

**Domain**: `Perception_Vision`

## Fix

Use deformable cross-attention with learned sampling points to selectively attend to relevant image features instead of the entire grid.

## Anti-pattern

Standard cross-attention over full image feature grids

## Cross-domain analogies

- **Planning_Decision** → Use hierarchical decomposition to attend only to task-relevant image regions.
  - related fix: CapNav benchmark: evaluate VLMs on 5 agent types with distinct constraints across 45 scenes, 473 tasks, and 2,365 QA pairs to test capability-aware navigation.
- **Learning_Training** → Shared embedding space with sparse attention reduces cross-modal computation for 3D Gaussian integration.
  - related fix: Unified multi-task model co-trained on all VLNVerse benchmark tasks (goal-oriented navigation, language-guided exploration, instruction following) using shared transformer-based cross-modal attention and common visual-linguistic embedding space.
- **Control_Locomotion** → Adaptive sparse attention guided by terrain-like depth saliency maps.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- deformable_cross_attention.before.py
+++ deformable_cross_attention.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Standard cross-attention over full image feature grids is computationally expensive and inefficient for integrating visual information into 3D Gaussian representations.

+# Fix    : Use deformable cross-attention with learned sampling points to selectively attend to relevant image features instead of the entire grid.

+# Avoid  : Standard cross-attention over full image feature grids

```
