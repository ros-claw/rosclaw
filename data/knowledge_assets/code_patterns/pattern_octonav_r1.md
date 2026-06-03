---
pattern_id: pattern_octonav_r1
applicable_symptoms: [octonav_r1]
domain: Planning_Decision
---

# Multimodal chain-of-thought reasoning in VLN generates excessive synthetic visual tokens, making real-time navigation impractical.

**Domain**: `Planning_Decision`

## Fix

FantasyVLN: a more efficient variant that reduces token inflation without sacrificing reasoning quality.

## Anti-pattern

OctoNav-R1's step-by-step imagined visual observations before action selection.

## Cross-domain analogies

- **Perception_Vision** → Restrict reasoning to only visually salient, sensor-detectable landmarks to limit token generation.
  - related fix: Prefer instruction design using landmarks that are visually salient and detectable by the agent's sensor suite (e.g., large objects, distinct colors).
- **Learning_Training** → Use egocentric action-conditioned token pruning to reduce synthetic visual tokens in real-time reasoning.
  - related fix: Use VLN-Ego dataset: large-scale egocentric video + expert action pairs from Habitat simulator for imitation learning (behavioral cloning) of navigation policies directly from first-person observations.
- **Control_Locomotion** → Pre-train a compact library of navigation primitives to decouple reasoning from real-time visual token generation.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- octonav_r1.before.py
+++ octonav_r1.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Multimodal chain-of-thought reasoning in VLN generates excessive synthetic visual tokens, making real-time navigation impractical.

+# Fix    : FantasyVLN: a more efficient variant that reduces token inflation without sacrificing reasoning quality.

+# Avoid  : OctoNav-R1's step-by-step imagined visual observations before action selection.

```
