---
pattern_id: pattern_sim_to_real_transfer
applicable_symptoms: [sim_to_real_transfer]
domain: Learning_Training
---

# Policy trained in simulation fails when deployed on real hardware due to domain gap

**Domain**: `Learning_Training`

## Fix

Domain randomization, system identification, or sim-to-real transfer techniques

## Anti-pattern

Training only on nominal simulation parameters without randomization

## Cross-domain analogies

- **Perception_Vision** → Derive a large-scale real-world dataset with paired sim-to-real labels to bridge the domain gap.
  - related fix: EmbodiedOcc-ScanNet: a large-scale egocentric occupancy dataset derived from ScanNet with voxel-level occupancy labels from first-person perspective.
- **Planning_Decision** → Use decision-driven domain randomization to balance sim fidelity, deployment cost, and safety.
  - related fix: Use Decision-Driven Semantic Object Exploration (DD-SOE) algorithm, which provides a sequential decision-making framework that balances semantic information gain, localization cost, and safety to guide exploration behavior.
- **Control_Locomotion** → Use camera images to augment training inputs, bridging sim-to-real domain gaps via perception.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- sim_to_real_transfer.before.py
+++ sim_to_real_transfer.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Policy trained in simulation fails when deployed on real hardware due to domain gap

+# Fix    : Domain randomization, system identification, or sim-to-real transfer techniques

+# Avoid  : Training only on nominal simulation parameters without randomization

```
