---
pattern_id: pattern_high_level_planner
applicable_symptoms: [high_level_planner]
domain: Planning_Decision
---

# VLN agent ignores landmark cues in long instructions

**Domain**: `Planning_Decision`

## Fix

Dynamic sub-instruction selection: high-level planner decomposes long-horizon instructions into contextually relevant sub-instructions based on current visual observations

## Anti-pattern

Using full instruction without filtering irrelevant parts

## Cross-domain analogies

- **Perception_Vision** → Use closed-loop verification to confirm landmark perception before executing navigation steps.
  - related fix: Prefer instruction design using landmarks that are visually salient and detectable by the agent's sensor suite (e.g., large objects, distinct colors).
- **Learning_Training** → Use human demonstration data to train behavioral cloning of landmark-attending navigation policies.
  - related fix: Use large-scale human demonstration dataset (Habitat-Web) collected via web-based interface to train a behavioral cloning policy for object-goal navigation.
- **Control_Locomotion** → Map visual features to actions, enabling real-time landmark-guided navigation.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- high_level_planner.before.py
+++ high_level_planner.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent ignores landmark cues in long instructions

+# Fix    : Dynamic sub-instruction selection: high-level planner decomposes long-horizon instructions into contextually relevant sub-instructions based on current visual observations

+# Avoid  : Using full instruction without filtering irrelevant parts

```
