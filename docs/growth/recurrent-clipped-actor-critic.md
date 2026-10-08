# Numeric recurrent clipped actor–critic fitting

`growth.recurrent_clipped_actor_critic` updates a causal GRU residual actor and
a persistent neural Monte Carlo value estimator from complete episode arrays.
It has no robot, simulator, transport, execution or promotion interface. The
dimensions, observations, physical rewards and evidence remain downstream.

Unlike positive-weight imitation, supplied negative advantages contribute to
the actor's clipped likelihood objective. Every row, including zero-advantage
rows, must first reconstruct its actual conditional Gaussian behavior density.
For AR(1) exploration, candidate conditioning is
`mu_t + rho * (actual_previous_action - candidate_mu_previous)`: reusing a fixed
old noise offset would give the wrong new-policy likelihood. The first frame
uses the marginal noise scale; later frames use the innovation scale. GRU and
conditioning state reset at every episode boundary.

The actor sees only context, its own causal state, and supplied frozen baseline
and gates. Frozen advantages and Monte Carlo returns are loss labels, not actor
inputs. The critic fits the supplied MC returns; this is not TD, GAE or a claim
that the caller collected an on-policy batch. Both actor and critic parameters
are returned for an explicitly reviewed next update.

Each joint Adam proposal is checked on every supplied row for marginal and
conditional mean KL relative to the original behavior. An excessive proposal
is rolled back jointly, including critic and optimizer state, and fitting
stops. These are numeric trust-region checks on logged observations, not an
environment safety proof, a bound on unseen states, or a projected-action KL.
The likelihood belongs to the latent Gaussian proposal, not clipped joint
commands or torques.

Torch is optional and imported only after input and behavior-density validation.
The function owns its numeric inputs, preserves RNG/thread/determinism settings,
validates fitted parameters, and returns source/input/parameter hashes, exact
KL measurements, positive/negative advantage counts and accepted/rejected update
counts. Physical provenance and on-policy provenance remain explicitly
unverified; runtime, promotion and hardware authorization remain false.

Synthetic tests verify genuine actor/value updates, negative-advantage influence,
protected zero gates, exact independent causal KL reconstruction, joint rollback,
repeatability and rejected malformed/non-finite/wrong-density inputs. These tests
are not physical rollout evidence or football performance.

## Independent value regression

`growth.normalized_value_regression.fit_normalized_value_regression` supplies a
separate offline critic optimizer and budget, without modifying the joint v1
learner or accepting actor parameters. Its contexts, labels and initial critic
are caller-authenticated. It does not compute actor advantages, authenticate
physical provenance, or authorize a policy update.

Complete supplied targets define fixed mean `m` and scale `s` (with an explicit
positive floor). The initial last layer is transformed to `W/s, (b-m)/s`;
original-unit predictions `s * normalized_prediction + m` must remain unchanged.
Only normalized regression error is optimized. The exported last layer is
folded back to `s*W, s*b+m`, so existing value evaluators receive original-unit
parameters, not normalized predictions. All labels are retained, including
negative and failure returns; targets are not clipped or redefined.

This uses the output-preserving parametrization described in
[PopArt](https://arxiv.org/abs/1602.07714), but statistics stay fixed for each
fit: it is **not adaptive online PopArt**, actor-critic RL, or a continual
optimizer checkpoint. Frozen whole-episode minibatches, independent Adam and
gradient clipping cannot inherit the actor's KL rollback. A caller still needs
properly versioned data/model commitments, frozen advantage computation,
protected actor updates and unchanged physical exams to use the result.

The numeric receipt binds source, parameter-contract source, inputs, initial
and fitted critic parameters, normalization statistics and actual optimizer
steps. Initial output preservation and final training error are measured on
every supplied frame. Held-out calibration, physical gain and all execution
authority remain false. Synthetic learning and regression tests do not certify
robot stability; lower training MSE cannot be substituted for a physical gate.

## Context-excluded temporal MC credit

`growth.cross_fitted_value_advantages.cross_fitted_value_advantages` consumes
complete chronological value targets and already completed normalized value
fits. It does not train anything. Unlike the historical terminal-return
cross-fit adapters, targets may change at each frame inside an episode.

The caller predeclares K folds (2..10): `episode_context_id % K`. Repeated
trajectories of one context cannot cross folds. Each supplied fold binds its
ordered context exclusions, exact float64 training inputs, initial/fitted
critic parameters, source versions, full configuration, statistics, counts,
and finite loss history. The adapter recomputes original-unit training MSE and
predicts only the excluded episodes. A full-data fit or changed training mean
cannot be relabelled as an excluded fit. Empty folds fail; they are not repaired
after results arrive. No best-fold or best-model selection occurs.

All supplied target rows, including failures, remain unchanged. Raw credit is
`target - excluded_value`; normalized advantages use one global mean and a
fixed standard-deviation floor of `1e-6`. Returned predictions, raw/normalized
advantages and episode fold IDs are privately owned read-only arrays. The
receipt includes each contributing fit hash and source/input/parameter
bindings. This is offline MC baseline subtraction, not online TD/GAE or an
on-policy claim. Actor observations receive neither context IDs nor future
labels through this API; it does not modify the existing actor learner.

Numeric receipt consistency does not authenticate optimizer execution, an
initial critic's previous training history, or physical trajectories. The
caller must authenticate the archived data and fit receipts separately.
Cross-fitting on consumed contexts is not a private Fresh exam. The result
grants no actor update, runtime, promotion or hardware authority; deployment
still requires the unchanged physical safety, retention and improvement gates.

Value evaluation batches whole episodes with at most 65,536 chronological
frames per hidden-layer allocation (the input horizon is at most 4,096). For
the fixed 64-unit float64 critic, each hidden array is at most 32 MiB. Input
arrays, outputs, masks, matrix-operation temporaries and receipt serialization
still require additional memory; this is not a total-process memory ceiling.
Batching changes neither frame order nor target statistics and does not
truncate long batches, omit failures, refit a critic or modify the policy.
