---
pattern_id: pattern_a_collection_of_resources_for_getting_started_in_icsscada_cybersecurity
applicable_symptoms: [a_collection_of_resources_for_getting_started_in_icsscada_cybersecurity]
domain: Learning_Training
---

# Newcomers to ICS/SCADA cybersecurity struggle to find a structured learning path due to the field's unique mission, risks, and threats compared to IT security.

**Domain**: `Learning_Training`

## Fix

Curated resource collection covering threats, physical processes, and control system fundamentals, with emphasis on mission-specific differences from IT security.

## Anti-pattern

Applying IT cybersecurity practices (patching, EDR, passwords) directly to OT/ICS environments without adapting to mission function and risk profiles.

## Cross-domain analogies

- **Perception_Vision** → Use spherical geometry-aware constraints to structure a risk-aligned learning path for ICS/SCADA cybersecurity.
  - related fix: Apply spherical geometry-aware constraints to regularize sampling offsets, leveraging panoramic ray properties for distortion-aware alignment without explicit undistortion.
- **Planning_Decision** → Structured multi-scenario evaluation benchmarks mapping learner constraints to adaptive training paths.
  - related fix: CapNav benchmark: evaluate VLMs on 5 agent types with distinct constraints across 45 scenes, 473 tasks, and 2,365 QA pairs to test capability-aware navigation.
- **Control_Locomotion** → Use a structured benchmark with standardized tasks to train newcomers on ICS-specific risks and threats.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- a_collection_of_resources_for_getting_started_in_icsscada_cybersecurity.before.py
+++ a_collection_of_resources_for_getting_started_in_icsscada_cybersecurity.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Newcomers to ICS/SCADA cybersecurity struggle to find a structured learning path due to the field's unique mission, risks, and threats compared to IT security.

+# Fix    : Curated resource collection covering threats, physical processes, and control system fundamentals, with emphasis on mission-specific differences from IT security.

+# Avoid  : Applying IT cybersecurity practices (patching, EDR, passwords) directly to OT/ICS environments without adapting to mission function and risk profiles.

```
