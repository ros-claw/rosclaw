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
