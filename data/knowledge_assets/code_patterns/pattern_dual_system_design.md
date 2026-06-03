---
pattern_id: pattern_dual_system_design
applicable_symptoms: [dual_system_design]
domain: Planning_Decision
---

# End-to-end VLN models couple reasoning and action in a single monolithic model, leading to poor interpretability and inability to adapt local actions based on high-level context.

**Domain**: `Planning_Decision`

## Fix

Dual-system architecture: separate high-level reasoning (System 2) for strategic planning from low-level control (System 1) for reactive execution, with decoupled training for each system.

## Anti-pattern

End-to-end pipelines that couple reasoning and action in a single monolithic model.

## Cross-domain analogies

- **Perception_Vision** → Use hierarchical decomposition to separate cross-modal reasoning from action execution.
  - related fix: Use a Vision-Language Model (VLM) that jointly processes visual and textual data for cross-modal reasoning, as in NavForesee.
- **Learning_Training** → Use back-translation to decouple reasoning and action via modular instruction-path generation.
  - related fix: Use back-translation: generate new instructions from paths and new paths from instructions using a pre-trained model, combined with environmental dropout to create diverse training triplets.
- **Control_Locomotion** → Distill high-level reasoning and local action into separate experts with closed-loop fine-tuning.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- dual_system_design.before.py
+++ dual_system_design.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: End-to-end VLN models couple reasoning and action in a single monolithic model, leading to poor interpretability and inability to adapt local actions based on high-level context.

+# Fix    : Dual-system architecture: separate high-level reasoning (System 2) for strategic planning from low-level control (System 1) for reactive execution, with decoupled training for each system.

+# Avoid  : End-to-end pipelines that couple reasoning and action in a single monolithic model.

```
