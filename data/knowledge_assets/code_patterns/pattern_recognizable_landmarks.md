---
pattern_id: pattern_recognizable_landmarks
applicable_symptoms: [recognizable_landmarks]
domain: Perception_Vision
---

# VLN agent fails to follow instructions that refer to landmarks it cannot perceive, causing navigation failures.

**Domain**: `Perception_Vision`

## Fix

Prefer instruction design using landmarks that are visually salient and detectable by the agent's sensor suite (e.g., large objects, distinct colors).

## Anti-pattern

Using instructor-described landmarks without considering agent's perceptual capabilities.

## Cross-domain analogies

- **Planning_Decision** → Joint training of landmark perception with instruction grounding to align visual and linguistic features.
  - related fix: Joint training of high-level action prediction with low-level action training, where the model learns both coarse goals and fine-grained motor commands simultaneously.
- **Learning_Training** → Use closed-loop verification to generate synthetic landmark references for unperceived objects.
  - related fix: Self-Refining Data Flywheel: generate synthetic navigation trajectories via a teacher policy, filter with a learned verifier, and iteratively retrain the student policy on the augmented data.
- **Control_Locomotion** → Use closed-loop verification to re-query perception when landmark detection fails.
  - related fix: Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Patch

```diff
--- recognizable_landmarks.before.py
+++ recognizable_landmarks.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent fails to follow instructions that refer to landmarks it cannot perceive, causing navigation failures.

+# Fix    : Prefer instruction design using landmarks that are visually salient and detectable by the agent's sensor suite (e.g., large objects, distinct colors).

+# Avoid  : Using instructor-described landmarks without considering agent's perceptual capabilities.

```
