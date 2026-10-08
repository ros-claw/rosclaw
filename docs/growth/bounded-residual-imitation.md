# Bounded residual imitation

`fit_bounded_residual_imitation` is task-neutral, optional supervised numeric
learning. It fits a tanh residual over an immutable supplied baseline and gate.
It does not run physics, update a deployed policy, or authenticate teachers.
Run fitting in an isolated process; PyTorch remains optional until fitting.

The downstream adapter owns feature causality, episode identity, dataset
provenance, training/holdout separation, teacher selection and physical exams.
Future success labels may be offline weights, never inference features.
This fitter returns plain numeric layers with no activation authority.

All declared rows are checked, including zero-weight rows. Only positive-weight
rows affect gradients. Zero-weight failure rows are **not** negative examples;
keeping their provenance is not learning avoidance from them. Repeated minibatch
updates are not new physical episodes. No temporal likelihood, PPO, critic,
KL trust region, motion-style discriminator or continual-learning guarantee
is implemented. A better training loss is not physical improvement.

The configuration bounds optimization steps, batch size, learning rate and
residual magnitude. Final predictions have the form
`baseline + residual_cap * gate * tanh(network(context))`. This numeric cap
does not certify downstream joint, actuator or body safety. Exact zero gates
leave the supplied baseline unchanged. All simulator execution limits and
independent validation remain the adapter's responsibility.

Input arrays and original layers are owned and never modified. Fitting restores
thread counts, CPU/selected CUDA RNG and deterministic-algorithm settings.
GPU fitting requires explicit deterministic cuBLAS configuration. Invalid
numeric inputs and nonfinite losses/gradients reject fitting.
