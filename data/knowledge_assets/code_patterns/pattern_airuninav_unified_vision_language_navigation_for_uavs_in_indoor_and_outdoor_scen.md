---
pattern_id: pattern_airuninav_unified_vision_language_navigation_for_uavs_in_indoor_and_outdoor_scen
applicable_symptoms: [airuninav_unified_vision_language_navigation_for_uavs_in_indoor_and_outdoor_scen]
domain: Planning_Decision
---

# UAV navigation in indoor and outdoor scenes lacks a unified vision-language framework, leading to poor generalization across environments.

**Domain**: `Planning_Decision`

## Fix

Unified architecture with text tokenizer, video encoder (history+current), connector, and LLM (Qwen 2 7B) outputting discrete actions: indoor (stop, forward, left, right) and outdoor (add ascend, descend, leftward, rightward).

## Anti-pattern

Separate indoor/outdoor navigation systems without shared representation.

## Cross-domain analogies

- **Perception_Vision** → Use a transformer-based 3D decoder to predict unified occupancy and semantics from vision-language inputs.
  - related fix: Learn an occupancy network that predicts 3D occupancy and semantics from multi-camera images using a transformer-based 3D decoder.
- **Learning_Training** → Distribute vision-language processing across multiple parallel workers with synchronized context updates.
  - related fix: Distribute PPO training across multiple workers with synchronized gradient updates (DD-PPO).
- **Control_Locomotion** → Pre-train a library of reusable vision-language navigation primitives decoupled from high-level planning.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- airuninav_unified_vision_language_navigation_for_uavs_in_indoor_and_outdoor_scen.before.py
+++ airuninav_unified_vision_language_navigation_for_uavs_in_indoor_and_outdoor_scen.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: UAV navigation in indoor and outdoor scenes lacks a unified vision-language framework, leading to poor generalization across environments.

+# Fix    : Unified architecture with text tokenizer, video encoder (history+current), connector, and LLM (Qwen 2 7B) outputting discrete actions: indoor (stop, forward, left, right) and outdoor (add ascend, descend, leftward, rightward).

+# Avoid  : Separate indoor/outdoor navigation systems without shared representation.

```
