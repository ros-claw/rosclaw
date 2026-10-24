# Frozen critic scaling for optional SIM learning

Source motivation: [Learning values across many orders of magnitude](https://arxiv.org/abs/1602.07714).
This reuses the fixed output-preserving target-normalization principle already
used by the experimental `growth.normalized_value_regression` implementation.
It is not an implementation of adaptive online PopArt or a new RL framework.

`control.fixed_value_scale` provides pure NumPy scalar affine conversion. The
existing G1 recurrent twin critics optionally configure frozen statistics from
training returns before their first optimizer step. The actor and its optimizer
are untouched. Critic outputs and actor objectives remain in original units;
critic residuals are divided by their head's scale. Rewards are not clipped or
relabelled. A scale floor of one avoids magnifying all-zero fall labels.

The affine reparameterization preserves initial raw-unit predictions within
floating precision (not an arbitrary-precision equality claim). Scale buffers
and the configured marker survive checkpoints; legacy checkpoints missing all
buffers load only identity scales. Partially present, invalid, twin-disagreeing
or current/target-disagreeing scales are rejected. Learned Adam moments are not
silently rescaled: configuring after critic optimization is rejected.

This addresses one observed numerical issue, not counterfactual data coverage,
recurrent burn-in, entropy time units, task success or real-robot safety. The
caller must authenticate the training/validation split and validate calibration
before activating actor changes. All related G1 policy execution remains SIM_ONLY.
