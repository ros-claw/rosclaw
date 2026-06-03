---
pattern_id: ppo_entropy_collapse_guard
safety_label: Entropy_Collapse
applicable_symptoms: [ppo_entropy_collapse_guard]
domain: Learning_Training
source: curated
---

# PPO entropy crashes to zero and the policy fixates on a degenerate action

**Domain**: `Learning_Training`
**Safety label**: `Entropy_Collapse`

## Fix

Hold a minimum entropy bonus (coefficient ≥ 0.01) throughout training; add a target-KL trust region (early-stop the inner update when `mean_kl > 1.5 * target_kl`); and decay the learning rate (linear or cosine) so late-stage updates can't overrun the cliff. Reset the advantage running mean every N epochs to avoid frozen normalisation.

## Anti-pattern

Cranking the policy LR or removing the entropy bonus to 'commit' to the current winner — accelerates the collapse and corrupts the value-function moments so even a restart shows the same degenerate basin.

## Cross-domain analogies (curated)

- **Control_Locomotion** → Same family as anti-windup: bound the magnitude of the update step the way you bound the actuator — KL = LR-equivalent for policies.
  - related fix: Treat target-KL as the policy's `tau_max`; stop the inner loop the moment KL exceeds it.
- **Memory_Reasoning** → Entropy bonus plays the role of an attention sink: a tiny always-on term that prevents the distribution from collapsing onto one mode.
  - related fix: Keep a floor on the entropy coefficient the way KV-cache keeps `sink_tokens` — never decay it to zero.

## Patch

```diff
--- ppo_entropy_collapse_guard.before.py+++ ppo_entropy_collapse_guard.after.py@@ -1,3 +1,8 @@-loss = policy_loss + 0.5 * value_loss   # no entropy bonus
+ent_coef = max(0.01, 0.02 - 1e-6 * step)
+loss = policy_loss + 0.5 * value_loss - ent_coef * entropy
 loss.backward()
-optimizer.step()                          # no KL early-stop, no LR decay
+if mean_kl > 1.5 * target_kl:             # target-KL trust region
+    break                                  # early-stop the inner PPO epoch
+for g in optimizer.param_groups:           # linear LR decay
+    g['lr'] = lr0 * (1 - step / total_steps)
+optimizer.step()

```
