---
pattern_id: pattern_minivln_efficient_vision_and_language_navigation_by_progressive_knowledge_distil
applicable_symptoms: [minivln_efficient_vision_and_language_navigation_by_progressive_knowledge_distil]
domain: Learning_Training
---

# Large VLN models are too heavy for deployment, yet smaller models suffer accuracy loss.

**Domain**: `Learning_Training`

## Fix

Two-stage progressive knowledge distillation: first distill from large teacher to medium student, then from medium to small student, achieving 1/7 model size with same accuracy.

## Anti-pattern

Directly training a small model from scratch or single-stage distillation fails to match large model accuracy.

## Cross-domain analogies

- **Perception_Vision** → Fuse multi-scale model outputs with a sensor-like coverage strategy to handle blind spots in accuracy.
  - related fix: Multi-modal occupancy grid fusion that integrates RGB, depth, lidar, and proprioception with a sensor layout strategy to maximize coverage and handle self-occlusion.
- **Planning_Decision** → Decompose the VLN task into hierarchical stages to offload heavy reasoning to a smaller model.
  - related fix: Four-stage pipeline: Visual State Description → Reflection and Reasoning → Language Plan Generation → Executable Plan Generation
- **Control_Locomotion** → Closed-loop verification using local lightweight models to correct high-level predictions in real-time.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- minivln_efficient_vision_and_language_navigation_by_progressive_knowledge_distil.before.py
+++ minivln_efficient_vision_and_language_navigation_by_progressive_knowledge_distil.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Large VLN models are too heavy for deployment, yet smaller models suffer accuracy loss.

+# Fix    : Two-stage progressive knowledge distillation: first distill from large teacher to medium student, then from medium to small student, achieving 1/7 model size with same accuracy.

+# Avoid  : Directly training a small model from scratch or single-stage distillation fails to match large model accuracy.

```
