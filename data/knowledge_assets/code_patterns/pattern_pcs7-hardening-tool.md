---
pattern_id: pattern_pcs7-hardening-tool
applicable_symptoms: [pcs7-hardening-tool]
domain: Systems_Compute
---

# Siemens Simatic PCS 7 stations have insecure configurations (weak GPO, permissive folder sharing, short passwords, no password complexity) that expose industrial control systems to cyber attacks.

**Domain**: `Systems_Compute`

## Fix

Run the PCS7-Hardening-Tool PowerShell script as administrator to assess and then remediate security configurations: enforce strict GPO, remove Everyone share permissions, set password minimum length to 14 characters, and enable password complexity.

## Anti-pattern

Relying on default Siemens PCS 7 security settings without hardening.

## Cross-domain analogies

- **Perception_Vision** → Use synthetic adversarial network traffic to train a closed-loop verification agent that enforces secure configurations.
  - related fix: Generate synthetic images from landmark text descriptions via a text-to-image diffusion model, and train the agent with an auxiliary grounding loss that aligns instruction representations with imagination embeddings
- **Planning_Decision** → Hierarchical decomposition of security into goal setting, access alignment, and action enforcement.
  - related fix: Decompose navigation decision into three stages: imagination (generate goal representation), observation selection (align goal with sensory input), and action determination (compute motor commands).
- **Learning_Training** → Use counterfactual configuration audits contrasting secure and insecure states to highlight critical hardening gaps.
  - related fix: Use counterfactual trajectory demonstrations: generate and analyze alternative paths that could have been taken, then contrast them with expert trajectories to focus learning on the most critical features for navigation cost inference.

## Patch

```diff
--- pcs7-hardening-tool.before.py
+++ pcs7-hardening-tool.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Siemens Simatic PCS 7 stations have insecure configurations (weak GPO, permissive folder sharing, short passwords, no password complexity) that expose industrial control systems to cyber attacks.

+# Fix    : Run the PCS7-Hardening-Tool PowerShell script as administrator to assess and then remediate security configurations: enforce strict GPO, remove Everyone share permissions, set password minimum length to 14 characters, and enable password complexity.

+# Avoid  : Relying on default Siemens PCS 7 security settings without hardening.

```
