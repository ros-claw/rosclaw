# Correlated residual gradient: numeric proposal backend

`rosclaw.growth.correlated_residual_gradient` accepts frozen arrays and finite
residual network layers. It has no robot, scene, ROS, executor, activation,
checkpoint loading or permit interface. Torch remains optional and lazy.

The backend reconstructs the actual AR(1) behavior likelihood, including
trajectory resets. Previous candidate means remain in the gradient graph.
It optimizes a clipped PPO objective with conditional KL penalty and checks
both conditional and marginal KL against a fixed 0.005 budget. CPU float64
is the reference; accepted steps must reduce the objective and remain below
0.0049. At most 160 steps and 13 backtracking reductions are allowed.

Defaults match the existing downstream smooth motor numerical reference:
raw residual cap 0.05, learning rate 0.0001, rho 0.9. Explicit experiments may
declare cap up to 0.2 and learning rate up to 0.0004. These are latent units,
not joint torque limits. Expanded parameters have **no physical qualification**.

The optional terminal critic uses four whole-trajectory cross-fit folds and
ridge 0.01. It is a Monte-Carlo terminal critic, not a bootstrapped continual
actor-critic. Callers must establish complete physical data provenance and
independently evaluate protected skills, safety, fresh holdouts and promotion.
All result authority flags are false; `physical_batch_verified` is false.

The Soccer adapter's numerical migration fixture compares the default engine
with the unchanged historical optimizer. Network weights, critic weights,
accepted-step count and loss history match exactly; compiled-inference KL
matches within 1e-12. Synthetic tests do not establish football improvement.
Process-wide CPU RNG, thread and deterministic-algorithm settings are restored
on success and exceptions. No CUDA random state is intentionally changed.
