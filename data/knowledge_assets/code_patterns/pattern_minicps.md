---
pattern_id: pattern_minicps
applicable_symptoms: [minicps]
domain: World_Physics
---

# CPS security research lacks realistic simulation environments for testing attacks and defenses.

**Domain**: `World_Physics`

## Fix

Use MiniCPS framework combining mininet network emulation with physical process and control device simulation for real-time CPS simulation.

## Anti-pattern

Using static or purely software-based CPS models without network emulation.

## Cross-domain analogies

- **Perception_Vision** → Use passive reconnaissance and geolocation APIs to map real-world CPS assets into simulation environments.
  - related fix: Use Shodan, BinaryEdge, and WHOISXMLAPI to passively scan for ICS/IoT devices, geolocate them via Google Maps and device indicators, and report to CERT.
- **Planning_Decision** → Use sequential decision-making to balance information gain, safety, and cost in CPS attack-defense simulation.
  - related fix: Use Decision-Driven Semantic Object Exploration (DD-SOE) algorithm, which provides a sequential decision-making framework that balances semantic information gain, localization cost, and safety to guide exploration behavior.
- **Learning_Training** → Generate synthetic attack-defense scenarios at scale using a transformer model.
  - related fix: Train a transformer agent on 4.2 million synthetic instruction-trajectory pairs generated at scale, reducing reliance on human demonstrations.

## Patch

```diff
--- minicps.before.py
+++ minicps.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: CPS security research lacks realistic simulation environments for testing attacks and defenses.

+# Fix    : Use MiniCPS framework combining mininet network emulation with physical process and control device simulation for real-time CPS simulation.

+# Avoid  : Using static or purely software-based CPS models without network emulation.

```
