---
pattern_id: pattern_memory_guided_decision_making
applicable_symptoms: [memory_guided_decision_making]
domain: Memory_Reasoning
---

# Reactive policies that map only immediate sensory input to actions fail under partial observability or ambiguous sensor readings, leading to greedy or repetitive errors in long-term navigation.

**Domain**: `Memory_Reasoning`

## Fix

Maintain a bounded internal state (e.g., dynamic bounded memory queue) that stores recent observations and actions, and fuse it with current observation via attention or recurrent layer before action selection.

## Anti-pattern

Using purely reactive policies without memory, which ignore historical context and cause inconsistent decisions.

## Cross-domain analogies

- **Perception_Vision** → Train memory on simulated partial observations with injected ambiguity to match real-world occlusion patterns.
  - related fix: Generate synthetic depth images with simulated sensor noise, self-occlusion, and lighting effects to match real sensor distribution.
- **Planning_Decision** → Use a learned forward model of latent state dynamics to predict future observations and guide policy under partial observability.
  - related fix: Learn a perceptive forward dynamics model that predicts future states from visual observations and robot state, then use it in a model predictive control framework for safe, platform-aware navigation.
- **Learning_Training** → Inject self-occlusion-aware latent state modeling to disambiguate partial observations.
  - related fix: Synthetic depth image generation with self-occlusion-aware ray casting and noise-aware modeling calibrated from real depth sensor characteristics.

## Patch

```diff
--- memory_guided_decision_making.before.py
+++ memory_guided_decision_making.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Reactive policies that map only immediate sensory input to actions fail under partial observability or ambiguous sensor readings, leading to greedy or repetitive errors in long-term navigation.

+# Fix    : Maintain a bounded internal state (e.g., dynamic bounded memory queue) that stores recent observations and actions, and fuse it with current observation via attention or recurrent layer before action selection.

+# Avoid  : Using purely reactive policies without memory, which ignore historical context and cause inconsistent decisions.

```
