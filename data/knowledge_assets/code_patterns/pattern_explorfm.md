---
pattern_id: pattern_explorfm
applicable_symptoms: [explorfm]
domain: Planning_Decision
---

# Semantic navigation in unknown environments fails to prioritize frontiers relevant to user-specified objects or terrain types, especially beyond depth sensor range.

**Domain**: `Planning_Decision`

## Fix

Use a vision-language foundation model (ExploRFM) that jointly predicts traversability, visual frontiers, and open-vocabulary object similarity from a single camera image, then scores frontier nodes in a navigation graph by thresholding and combining these maps.

## Anti-pattern

Using only depth-based frontier detection without semantic relevance scoring.

## Cross-domain analogies

- **Perception_Vision** → Use 3D-GS to generate photorealistic long-range semantic maps from sparse sensor data.
  - related fix: Construct high-fidelity datasets using 3D Gaussian Splatting (3D-GS) to generate photorealistic novel-view synthetic images from sparse real captures, preserving fine-grained textures and lighting details.
- **Learning_Training** → Use closed-loop data aggregation to iteratively refine frontier prioritization based on policy-encountered terrain.
  - related fix: Iteratively collect new data under the current policy's distribution, query the expert for corrective actions, and aggregate this data into the training set (DAgger).
- **Control_Locomotion** → Pre-train a library of semantic frontier priors via RL, decoupling object-driven exploration from high-level planning.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- explorfm.before.py
+++ explorfm.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Semantic navigation in unknown environments fails to prioritize frontiers relevant to user-specified objects or terrain types, especially beyond depth sensor range.

+# Fix    : Use a vision-language foundation model (ExploRFM) that jointly predicts traversability, visual frontiers, and open-vocabulary object similarity from a single camera image, then scores frontier nodes in a navigation graph by thresholding and combining these maps.

+# Avoid  : Using only depth-based frontier detection without semantic relevance scoring.

```
