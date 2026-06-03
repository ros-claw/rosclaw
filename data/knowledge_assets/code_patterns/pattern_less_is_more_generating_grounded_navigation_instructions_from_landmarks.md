---
pattern_id: pattern_less_is_more_generating_grounded_navigation_instructions_from_landmarks
applicable_symptoms: [less_is_more_generating_grounded_navigation_instructions_from_landmarks]
domain: Planning_Decision
---

# VLN agents fail to follow long instructions because they ignore landmark cues and rely on low-level action sequences.

**Domain**: `Planning_Decision`

## Fix

Generate grounded navigation instructions by selecting a minimal set of distinctive landmarks along the path and describing actions relative to those landmarks.

## Anti-pattern

Using dense, step-by-step instructions that cause agents to overfit to low-level actions and ignore landmarks.

## Cross-domain analogies

- **Perception_Vision** → Train VLN agents on trajectories with simulated landmark noise and occlusion to force landmark reliance.
  - related fix: Generate synthetic depth images with simulated sensor noise, self-occlusion, and lighting effects to match real sensor distribution.
- **Learning_Training** → Use domain randomization on landmark saliency to force agent reliance on high-level cues.
  - related fix: Domain randomization, system identification, or sim-to-real transfer techniques
- **Control_Locomotion** → Map camera images to actions for obstacle avoidance, so map landmark cues to decisions for instruction following.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- less_is_more_generating_grounded_navigation_instructions_from_landmarks.before.py
+++ less_is_more_generating_grounded_navigation_instructions_from_landmarks.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to follow long instructions because they ignore landmark cues and rely on low-level action sequences.

+# Fix    : Generate grounded navigation instructions by selecting a minimal set of distinctive landmarks along the path and describing actions relative to those landmarks.

+# Avoid  : Using dense, step-by-step instructions that cause agents to overfit to low-level actions and ignore landmarks.

```
