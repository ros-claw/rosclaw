---
pattern_id: pattern_cvdn_cooperative_vision_and_dialog_navigation
applicable_symptoms: [cvdn_cooperative_vision_and_dialog_navigation]
domain: Planning_Decision
---

# VLN agents fail to follow dynamic, ambiguous instructions that require clarification, as they rely on static pre-generated instructions.

**Domain**: `Planning_Decision`

## Fix

Use a cooperative dialog framework where an oracle provides step-by-step guidance and the agent can ask clarifying questions, grounding instructions in real-time visual observations.

## Anti-pattern

Standard VLN tasks with static human-generated instructions that do not allow interactive clarification.

## Cross-domain analogies

- **Perception_Vision** → Use shared transformer layers with modality-specific encoders to jointly embed dynamic instructions and real-time visual feedback.
  - related fix: Use a multimodal versatile network (MMV) with shared transformer layers and modality-specific encoders to learn joint embeddings across modalities.
- **Learning_Training** → Regularize policy outputs against prior instructions to penalize deviation when integrating new clarifications.
  - related fix: Functional regularisation: add a penalty on changes to the network's input-output mapping (e.g., using KL divergence or L2 distance on outputs) when training on new tasks.
- **Control_Locomotion** → Pre-train a library of reusable dialogue primitives via RL, decoupling clarification skill acquisition from task planning.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- cvdn_cooperative_vision_and_dialog_navigation.before.py
+++ cvdn_cooperative_vision_and_dialog_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to follow dynamic, ambiguous instructions that require clarification, as they rely on static pre-generated instructions.

+# Fix    : Use a cooperative dialog framework where an oracle provides step-by-step guidance and the agent can ask clarifying questions, grounding instructions in real-time visual observations.

+# Avoid  : Standard VLN tasks with static human-generated instructions that do not allow interactive clarification.

```
