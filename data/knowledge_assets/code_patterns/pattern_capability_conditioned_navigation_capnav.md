---
pattern_id: pattern_capability_conditioned_navigation_capnav
applicable_symptoms: [capability_conditioned_navigation_capnav]
domain: Planning_Decision
---

# VLN agents fail to adapt navigation plans when agent mobility, dimensions, or interaction abilities vary (e.g., cannot fit through narrow passages, cannot open cabinets).

**Domain**: `Planning_Decision`

## Fix

CapNav benchmark: evaluate VLMs on 5 agent types with distinct constraints across 45 scenes, 473 tasks, and 2,365 QA pairs to test capability-aware navigation.

## Anti-pattern

Standard VLN benchmarks ignore agent capabilities, leading to overestimated generalization.

## Cross-domain analogies

- **Perception_Vision** → Use shared transformer layers with modality-specific encoders to learn joint embeddings across diverse agent capabilities.
  - related fix: Use a multimodal versatile network (MMV) with shared transformer layers and modality-specific encoders to learn joint embeddings across modalities.
- **Learning_Training** → Use causal intervention to remove mobility constraints' influence on navigation policy.
  - related fix: Use causal counterfactual reasoning to remove the influence of sensitive attributes on predictions by intervening on the causal graph.
- **Control_Locomotion** → Incorporate agent capability maps into planning to dynamically adapt routes and actions.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- capability_conditioned_navigation_capnav.before.py
+++ capability_conditioned_navigation_capnav.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to adapt navigation plans when agent mobility, dimensions, or interaction abilities vary (e.g., cannot fit through narrow passages, cannot open cabinets).

+# Fix    : CapNav benchmark: evaluate VLMs on 5 agent types with distinct constraints across 45 scenes, 473 tasks, and 2,365 QA pairs to test capability-aware navigation.

+# Avoid  : Standard VLN benchmarks ignore agent capabilities, leading to overestimated generalization.

```
