---
pattern_id: pattern_orionnav_online_planning_for_robot_autonomy_with_context_aware_llm_and_open_voca
applicable_symptoms: [orionnav_online_planning_for_robot_autonomy_with_context_aware_llm_and_open_voca]
domain: Planning_Decision
---

# Long-horizon navigation in unknown environments fails due to lack of semantic context and dynamic replanning capability.

**Domain**: `Planning_Decision`

## Fix

Online planning with LLM (GPT-4-Turbo) and open-vocabulary semantic scene graphs, integrating LiDAR-SLAM, RGBD semantic mapping (FC-CLIP), and ROS2 navigation stack with exploration.

## Anti-pattern

Classical navigation planners that ignore semantic object information and cannot adapt to novel goals or dynamic obstacles.

## Cross-domain analogies

- **Perception_Vision** → Use Laplacian variance to filter low-information planning states, discarding those with poor semantic context.
  - related fix: Laplacian Variance Filtering (LVF): compute variance of Laplacian of each frame; discard or deweight frames with low variance (high blur) to reduce jitter-induced motion blur.
- **Learning_Training** → Use causal intervention to remove irrelevant semantic context for dynamic replanning.
  - related fix: Use causal counterfactual reasoning to remove the influence of sensitive attributes on predictions by intervening on the causal graph.
- **Control_Locomotion** → Use blocked-action heuristic to trigger semantic-aware replanning when path fails.
  - related fix: Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Patch

```diff
--- orionnav_online_planning_for_robot_autonomy_with_context_aware_llm_and_open_voca.before.py
+++ orionnav_online_planning_for_robot_autonomy_with_context_aware_llm_and_open_voca.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Long-horizon navigation in unknown environments fails due to lack of semantic context and dynamic replanning capability.

+# Fix    : Online planning with LLM (GPT-4-Turbo) and open-vocabulary semantic scene graphs, integrating LiDAR-SLAM, RGBD semantic mapping (FC-CLIP), and ROS2 navigation stack with exploration.

+# Avoid  : Classical navigation planners that ignore semantic object information and cannot adapt to novel goals or dynamic obstacles.

```
