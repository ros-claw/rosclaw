---
pattern_id: pattern_the_industrial_control_system_cyber_kill_chain
applicable_symptoms: [the_industrial_control_system_cyber_kill_chain]
domain: Systems_Compute
---

# Industrial control systems are vulnerable to multi-stage cyber attacks that can disrupt critical infrastructure.

**Domain**: `Systems_Compute`

## Fix

Implement the ICS Cyber Kill Chain framework to detect and block attacks at each stage: reconnaissance, weaponization, delivery, exploitation, installation, command and control, and actions on objectives.

## Anti-pattern

Traditional IT-centric kill chains fail to account for ICS-specific attack vectors and operational technology constraints.

## Cross-domain analogies

- **Perception_Vision** → Use shared transformer layers with modality-specific encoders to detect multi-stage attack patterns across diverse control signals.
  - related fix: Use a multimodal versatile network (MMV) with shared transformer layers and modality-specific encoders to learn joint embeddings across modalities.
- **Planning_Decision** → Use closed-loop verification benchmarks to detect multi-stage attack patterns in control systems.
  - related fix: Use Target-Bench benchmark to evaluate and improve semantic reasoning by measuring target-approaching metrics in path planning tasks.
- **Learning_Training** → Use synthetic attack trajectory generation from unlabeled system logs to augment training data for anomaly detection.
  - related fix: Train a speaker model to generate synthetic instruction–trajectory pairs from unannotated visual paths, then augment the original training set with these synthetic pairs.

## Patch

```diff
--- the_industrial_control_system_cyber_kill_chain.before.py
+++ the_industrial_control_system_cyber_kill_chain.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Industrial control systems are vulnerable to multi-stage cyber attacks that can disrupt critical infrastructure.

+# Fix    : Implement the ICS Cyber Kill Chain framework to detect and block attacks at each stage: reconnaissance, weaponization, delivery, exploitation, installation, command and control, and actions on objectives.

+# Avoid  : Traditional IT-centric kill chains fail to account for ICS-specific attack vectors and operational technology constraints.

```
