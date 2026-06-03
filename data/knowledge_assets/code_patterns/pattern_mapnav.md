---
pattern_id: pattern_mapnav
applicable_symptoms: [mapnav]
domain: Planning_Decision
---

# VLN agents that process long video histories suffer from high storage and computational overhead, limiting real-time deployment.

**Domain**: `Planning_Decision`

## Fix

Replace historical frame sequences with an Annotated Semantic Map (ASM) that is constructed at episode start and updated each timestep, using a VLM to add textual labels for key regions.

## Anti-pattern

Processing long video histories frame-by-frame without a compact spatial representation.

## Cross-domain analogies

- **Perception_Vision** → Use learned semantic representations to compress video history into compact task-relevant features.
  - related fix: VISR: a framework integrating visual perception with semantic reasoning using learned representations
- **Learning_Training** → Use hierarchical decomposition to compress long video histories into compact scene graphs.
  - related fix: ScaleVLN: large-scale synthetic data generation by combining 3D scene graphs with LLM-generated instructions and augmenting with panoramic views and object-level grounding.
- **Control_Locomotion** → Train an end-to-end policy that maps compressed video features directly to actions, bypassing full history storage.
  - related fix: Train a single neural network policy via deep reinforcement learning that maps raw depth camera images directly to motor commands, bypassing hand-crafted perception and control layers.

## Patch

```diff
--- mapnav.before.py
+++ mapnav.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents that process long video histories suffer from high storage and computational overhead, limiting real-time deployment.

+# Fix    : Replace historical frame sequences with an Annotated Semantic Map (ASM) that is constructed at episode start and updated each timestep, using a VLM to add textual labels for key regions.

+# Avoid  : Processing long video histories frame-by-frame without a compact spatial representation.

```
