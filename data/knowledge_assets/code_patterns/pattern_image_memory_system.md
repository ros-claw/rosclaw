---
pattern_id: pattern_image_memory_system
applicable_symptoms: [image_memory_system]
domain: Memory_Reasoning
---

# Embodied agents lack visual context for long-horizon task decomposition and replanning, leading to brittle plans that ignore past observations.

**Domain**: `Memory_Reasoning`

## Fix

Store visual observations (raw images or extracted features) in a structured memory that can be queried by a reasoning module during task decomposition and replanning.

## Anti-pattern

Using only current visual input without retaining past observations for context.

## Cross-domain analogies

- **Perception_Vision** → Fuse episodic memory with visual attention to prioritize salient past observations for replanning.
  - related fix: Implement a sensing intelligence pipeline that fuses multiple sensor modalities (e.g., vision, depth, IMU) into a coherent representation, with attention mechanisms to prioritize salient input.
- **Planning_Decision** → Use frontier queries as discrete visual memory anchors for context-driven replanning.
  - related fix: Use frontier cells as discrete spatial hypotheses queried via frontier_queries to guide exploration without full map reconstruction.
- **Learning_Training** → Use video-only input to force implicit memory, eliminating explicit state dependency.
  - related fix: Use video-only input modality (no depth or map) combined with domain randomization to eliminate sensor fidelity and geometry transfer gaps

## Patch

```diff
--- image_memory_system.before.py
+++ image_memory_system.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Embodied agents lack visual context for long-horizon task decomposition and replanning, leading to brittle plans that ignore past observations.

+# Fix    : Store visual observations (raw images or extracted features) in a structured memory that can be queried by a reasoning module during task decomposition and replanning.

+# Avoid  : Using only current visual input without retaining past observations for context.

```
