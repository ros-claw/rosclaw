---
pattern_id: pattern_unified_multi_task_model_vlnverse
applicable_symptoms: [unified_multi_task_model_vlnverse]
domain: Learning_Training
---

# Separate task-specific models for VLN sub-tasks suffer from catastrophic forgetting and poor sample efficiency, limiting generalization to real-world navigation.

**Domain**: `Learning_Training`

## Fix

Unified multi-task model co-trained on all VLNVerse benchmark tasks (goal-oriented navigation, language-guided exploration, instruction following) using shared transformer-based cross-modal attention and common visual-linguistic embedding space.

## Anti-pattern

Training separate models for each VLN sub-task independently.

## Cross-domain analogies

- **Perception_Vision** → Use 3D-GS to generate photorealistic synthetic trajectories for continual multi-task rehearsal.
  - related fix: Construct high-fidelity datasets using 3D Gaussian Splatting (3D-GS) to generate photorealistic novel-view synthetic images from sparse real captures, preserving fine-grained textures and lighting details.
- **Planning_Decision** → Use an LLM-based Advisor to monitor cross-task context and trigger selective memory consolidation.
  - related fix: Use an LLM-based Advisor module that continuously monitors system state and task progress, evaluating contextual cues (e.g., unexpected sensor readings, partial failures) to issue a replanning request only when necessary.
- **Control_Locomotion** → Multi-expert distillation with DAgger and RL fine-tuning.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- unified_multi_task_model_vlnverse.before.py
+++ unified_multi_task_model_vlnverse.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Separate task-specific models for VLN sub-tasks suffer from catastrophic forgetting and poor sample efficiency, limiting generalization to real-world navigation.

+# Fix    : Unified multi-task model co-trained on all VLNVerse benchmark tasks (goal-oriented navigation, language-guided exploration, instruction following) using shared transformer-based cross-modal attention and common visual-linguistic embedding space.

+# Avoid  : Training separate models for each VLN sub-task independently.

```
