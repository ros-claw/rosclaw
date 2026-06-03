---
pattern_id: pattern_march_in_chat_interactive_prompting_for_remote_embodied_referring_expression
applicable_symptoms: [march_in_chat_interactive_prompting_for_remote_embodied_referring_expression]
domain: Planning_Decision
---

# Remote embodied referring expression agents fail to resolve ambiguous references in long-horizon tasks due to limited interactive feedback.

**Domain**: `Planning_Decision`

## Fix

March-in-Chat (MiC): interactive prompting that allows the agent to ask clarifying questions and receive human responses during navigation.

## Anti-pattern

Static instruction following without online clarification.

## Cross-domain analogies

- **Perception_Vision** → Use a pre-trained vision-language model for shared embedding to resolve ambiguous references via zero-shot grounding.
  - related fix: Use a pre-trained vision-language model (e.g., CLIP) trained on large-scale image-text pairs to learn a shared embedding space, enabling zero-shot transfer to unseen tasks and objects.
- **Learning_Training** → Use two-stage training: supervised pretraining then reinforcement fine-tuning with interactive reward.
  - related fix: Two-stage training: first supervised fine-tuning on expert demonstrations, then reinforcement fine-tuning with policy gradient (e.g., PPO) to maximize task completion reward
- **Control_Locomotion** → Train a separate verification policy to override ambiguous decisions when confidence is low.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- march_in_chat_interactive_prompting_for_remote_embodied_referring_expression.before.py
+++ march_in_chat_interactive_prompting_for_remote_embodied_referring_expression.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Remote embodied referring expression agents fail to resolve ambiguous references in long-horizon tasks due to limited interactive feedback.

+# Fix    : March-in-Chat (MiC): interactive prompting that allows the agent to ask clarifying questions and receive human responses during navigation.

+# Avoid  : Static instruction following without online clarification.

```
