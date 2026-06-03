---
pattern_id: pattern_bridging_the_indoor_outdoor_gap_vision_centric_instruction_guided_embodied_navig
applicable_symptoms: [bridging_the_indoor_outdoor_gap_vision_centric_instruction_guided_embodied_navig]
domain: Planning_Decision
---

# Vision-language navigation agents fail when transitioning from outdoor to indoor environments due to lack of training data covering the full trajectory.

**Domain**: `Planning_Decision`

## Fix

BridgeNavDataset: a dataset with 55K street-view images, 100+ hours video, and 55K trajectory-instruction pairs for outdoor-to-indoor navigation.

## Anti-pattern

Existing VLN datasets focus on either indoor or outdoor scenes, not the transition.

## Cross-domain analogies

- **Perception_Vision** → Use multi-modal attention to fuse outdoor and indoor visual cues for robust trajectory transitions.
  - related fix: Implement a sensing intelligence pipeline that fuses multiple sensor modalities (e.g., vision, depth, IMU) into a coherent representation, with attention mechanisms to prioritize salient input.
- **Learning_Training** → Use synthetic multi-scene trajectory generation with LLM-based instruction augmentation.
  - related fix: ScaleVLN: large-scale synthetic data generation by combining 3D scene graphs with LLM-generated instructions and augmenting with panoramic views and object-level grounding.
- **Control_Locomotion** → Use blocked-action heuristic to trigger alternative pathfinding when outdoor-indoor transitions fail.
  - related fix: Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Patch

```diff
--- bridging_the_indoor_outdoor_gap_vision_centric_instruction_guided_embodied_navig.before.py
+++ bridging_the_indoor_outdoor_gap_vision_centric_instruction_guided_embodied_navig.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Vision-language navigation agents fail when transitioning from outdoor to indoor environments due to lack of training data covering the full trajectory.

+# Fix    : BridgeNavDataset: a dataset with 55K street-view images, 100+ hours video, and 55K trajectory-instruction pairs for outdoor-to-indoor navigation.

+# Avoid  : Existing VLN datasets focus on either indoor or outdoor scenes, not the transition.

```
