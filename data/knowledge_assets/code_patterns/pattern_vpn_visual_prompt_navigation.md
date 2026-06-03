---
pattern_id: pattern_vpn_visual_prompt_navigation
applicable_symptoms: [vpn_visual_prompt_navigation]
domain: Planning_Decision
---

# VLN agents struggle with ambiguous language instructions from non-expert users, leading to navigation failures.

**Domain**: `Planning_Decision`

## Fix

Use visual prompts (2D top-view map annotations) instead of natural language to guide navigation, implemented via VPNet baseline.

## Anti-pattern

Natural language instructions cause ambiguity and high user expertise requirements.

## Cross-domain analogies

- **Perception_Vision** → Fuse multimodal language inputs into a unified semantic representation for robust instruction grounding.
  - related fix: Propose a multimodal occupancy perception system that fuses vision, depth, and other sensor data into a unified occupancy representation for humanoid robots.
- **Learning_Training** → Pretrain on diverse human-annotated instruction data to learn robust language grounding before fine-tuning on navigation.
  - related fix: Pretrain a vision-language model on large-scale web data to learn cross-modal alignment, then fine-tune on navigation tasks
- **Control_Locomotion** → Closed-loop verification reconciles local perception with high-level commands to resolve ambiguous instructions.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- vpn_visual_prompt_navigation.before.py
+++ vpn_visual_prompt_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents struggle with ambiguous language instructions from non-expert users, leading to navigation failures.

+# Fix    : Use visual prompts (2D top-view map annotations) instead of natural language to guide navigation, implemented via VPNet baseline.

+# Avoid  : Natural language instructions cause ambiguity and high user expertise requirements.

```
