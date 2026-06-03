---
pattern_id: pattern_rl_locomotion_policy
applicable_symptoms: [rl_locomotion_policy]
domain: Control_Locomotion
---

# Legged robot locomotion policies are often too heavy for real-time inference on embedded hardware, causing high latency and unstable gaits.

**Domain**: `Control_Locomotion`

## Fix

Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Anti-pattern

Using complex, non-compact neural network architectures that cannot run in real-time on onboard hardware.

## Cross-domain analogies

- **Perception_Vision** → Selectively skip or downweight high-latency control steps to maintain real-time gait stability.
  - related fix: Laplacian Variance Filtering (LVF): compute variance of Laplacian of each frame; discard or deweight frames with low variance (high blur) to reduce jitter-induced motion blur.
- **Planning_Decision** → Hierarchical decomposition with lightweight local policies guided by sparse high-level priors.
  - related fix: Scenario-oriented object navigation with graph-based exploration: build a semantic graph of explored regions, use a high-level policy to select frontier nodes based on object-context priors, and a low-level policy to navigate to chosen nodes.
- **Learning_Training** → Distill a privileged teacher into a lightweight student policy for real-time inference.
  - related fix: Privileged Information Guidance (PIG): train a diffusion policy with privileged depth and collision information during training, then distill into a student policy that uses only RGB observations via a guidance loss.

## Patch

```diff
--- rl_locomotion_policy.before.py
+++ rl_locomotion_policy.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Legged robot locomotion policies are often too heavy for real-time inference on embedded hardware, causing high latency and unstable gaits.

+# Fix    : Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

+# Avoid  : Using complex, non-compact neural network architectures that cannot run in real-time on onboard hardware.

```
