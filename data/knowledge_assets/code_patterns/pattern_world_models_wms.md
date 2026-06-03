---
pattern_id: pattern_world_models_wms
applicable_symptoms: [world_models_wms]
domain: Learning_Training
---

# Agents require large amounts of real-world interaction to learn, limiting sample efficiency and safe exploration.

**Domain**: `Learning_Training`

## Fix

Train a neural world model that predicts future latent states and rewards from current observations and actions, enabling model-based planning and mental simulation without direct environment interaction.

## Anti-pattern

Model-free RL that learns purely from trial-and-error without an internal predictive model.

## Cross-domain analogies

- **Perception_Vision** → Use dual-view interaction sampling to reduce real-world trials via complementary synthetic and real data.
  - related fix: Dual-view visual prompt: combine two complementary spatial views into a single prompt at inference time, applied on top of a VLA model.
- **Planning_Decision** → Use interactive human queries to reduce unsafe real-world exploration needs.
  - related fix: March-in-Chat (MiC): interactive prompting that allows the agent to ask clarifying questions and receive human responses during navigation.
- **Control_Locomotion** → Train a safety critic to intervene when high-risk actions are proposed, enabling efficient safe exploration.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- world_models_wms.before.py
+++ world_models_wms.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Agents require large amounts of real-world interaction to learn, limiting sample efficiency and safe exploration.

+# Fix    : Train a neural world model that predicts future latent states and rewards from current observations and actions, enabling model-based planning and mental simulation without direct environment interaction.

+# Avoid  : Model-free RL that learns purely from trial-and-error without an internal predictive model.

```
