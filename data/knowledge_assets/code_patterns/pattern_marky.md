---
pattern_id: pattern_marky
applicable_symptoms: [marky]
domain: Learning_Training
---

# Lack of large-scale, high-quality multilingual instruction-trajectory pairs for training navigation agents, leading to poor generalization in unseen environments.

**Domain**: `Learning_Training`

## Fix

Use Marky, a multilingual instruction generator that produces visually grounded instruction-trajectory pairs at scale (4.2M pairs) by leveraging pretrained vision-language models and spatial alignment.

## Anti-pattern

Simple sequence-to-sequence models that produce hallucinated or spatially imprecise instructions.

## Cross-domain analogies

- **Perception_Vision** → Use text-to-image generation to create synthetic multilingual instruction-trajectory pairs for training.
  - related fix: Generate synthetic images from landmark text descriptions via a text-to-image diffusion model, and train the agent with an auxiliary grounding loss that aligns instruction representations with imagination embeddings
- **Planning_Decision** → Use hierarchical CFG decomposition to generate structured multilingual instruction-trajectory pairs.
  - related fix: Use a Context-Free Grammar (CFG) to decompose navigation instructions into hierarchical categories (landmarks, directions, actions) and evaluate each category independently.
- **Control_Locomotion** → Pre-train a diverse library of multilingual navigation trajectories via reinforcement learning, decoupling skill acquisition from task planning.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- marky.before.py
+++ marky.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Lack of large-scale, high-quality multilingual instruction-trajectory pairs for training navigation agents, leading to poor generalization in unseen environments.

+# Fix    : Use Marky, a multilingual instruction generator that produces visually grounded instruction-trajectory pairs at scale (4.2M pairs) by leveraging pretrained vision-language models and spatial alignment.

+# Avoid  : Simple sequence-to-sequence models that produce hallucinated or spatially imprecise instructions.

```
