---
pattern_id: pattern_deepmind_enformer
applicable_symptoms: [deepmind_enformer]
domain: Learning_Training
---

# Convolutional models cannot capture long-range regulatory interactions beyond ~100kb in genomic sequences

**Domain**: `Learning_Training`

## Fix

Use a convolutional stem followed by Transformer blocks with global attention to model dependencies across 100kb+ distances

## Anti-pattern

Pure convolutional architectures with limited receptive field

## Cross-domain analogies

- **Perception_Vision** → Fuse multi-scale genomic attention heads to prioritize distal regulatory interactions.
  - related fix: Implement a sensing intelligence pipeline that fuses multiple sensor modalities (e.g., vision, depth, IMU) into a coherent representation, with attention mechanisms to prioritize salient input.
- **Planning_Decision** → Tokenized encoding with dynamic budget sampling enables long-range genomic context.
  - related fix: Unified architecture with identifier tokens encoding embodiment and temporal context, plus dynamic token sampling under budget constraints
- **Control_Locomotion** → Train a secondary attention module to override local convolution outputs when long-range genomic context exceeds a threshold.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- deepmind_enformer.before.py
+++ deepmind_enformer.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Convolutional models cannot capture long-range regulatory interactions beyond ~100kb in genomic sequences

+# Fix    : Use a convolutional stem followed by Transformer blocks with global attention to model dependencies across 100kb+ distances

+# Avoid  : Pure convolutional architectures with limited receptive field

```
