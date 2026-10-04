# Explicit proposal trust-region regression

`rosclaw.growth.proposal_advantage_regression` is a task-neutral numerical
research primitive. It does not execute, activate, approve, or register a policy.
Its only execution ceiling is `PROPOSAL_ONLY_NO_RUNTIME`.

## Motivation and boundary

The existing bounded advantage regression retains its fixed 0.005 KL contract.
It and every existing motor-model validator are unchanged. An explicitly larger
proposal budget must not be represented as a legacy-approved motor model.

The new API permits `maximum_mean_kl` from 0.005 to 0.5. The budget bounds both
full-batch conditional AR KL and marginal KL, not trajectory risk, unseen-state
behavior, physical stability, or skill retention. Both KL values are checked
again after optimization. The line-search acceptance margin is 98% of the
declared budget; at the default, the original literal 0.0049 is retained to avoid
floating-point changes to legacy numerical behavior.

The objective remains AWR-inspired marginal action likelihood plus conditional
KL regularization. It is not TD-lambda AWR, online actor-critic, or iid PPO.
Conditional behavior likelihood, episode boundaries, frozen gates, finite
inputs, residual caps, and optional Torch dependency retain their original
checks. No simulator, checkpoint transport, hardware dependency, or activation
authority is added.

## Using the primitive

```python
from rosclaw.growth.proposal_advantage_regression import (
    ProposalAdvantageRegressionConfig,
    fit_proposal_advantage_residual,
)

config = ProposalAdvantageRegressionConfig(maximum_mean_kl=0.05)
# Supply independently validated arrays using the same causal batch contract
# as bounded_advantage_regression; this function does not validate physics.
proposal = fit_proposal_advantage_residual(config=config, **batch)
assert proposal["runtime_execution_authorized"] is False
```

Any downstream simulation experiment needs a distinct, sealed proposal model,
explicit provenance and budget, unchanged actuator/physics limits, real native
rollouts and independent audits. Physical skill retention must be tested rather
than inferred from a KL number. A larger budget can produce worse behavior.
Nothing here qualifies a candidate for runtime or real hardware.

## Verified locally

- Default configuration matches legacy outputs, weights, and loss history
  exactly; only algorithm identity and explicit proposal metadata differ.
- A synthetic, density-consistent larger-action batch uses a larger declared
  budget while preserving both bounds and zero-gate protected outputs.
- Tests reject invalid budgets, runtime execution ceilings, changed behavior
  likelihoods, nonfinite inputs, invalid gates and reset masks.
- Torch RNG, thread count and deterministic settings are restored on failure.

These are numerical unit tests, not evidence of improved football, universal
retention, or end-to-end physical safety.

## Source-pinned prediction-only inference

`CompiledContextPrediction` is a generic optional compilation surface for the
existing context-disjoint predictor. The unchanged full reference validates a
private model copy before immutable numeric arrays are allocated. Each query
checks bounded copied features, finite intermediates, input alignment and both
dependency source hashes. No training, simulator, policy, motor transport or
activation interface is introduced; Torch is not imported by compilation.

Tests compare original and compiled predictions exactly for multiple batch
sizes, preserve caller ownership, reject nonfinite/type/authority forgeries,
and reject source drift. Targeted prediction tests: 15 passed; complete Growth
suite: 358 passed. The existing missing `asyncio_mode` pytest plugin warning
remains. These are numerical tests, not native rollout qualification or a
claim of improved robot control. Actual downstream trained models still need
separate reference-parity evidence before use in simulation proposals.

`bounded_response_proposal` is a separate numerical local-quadratic primitive:
it accepts finite bounded response matrices, error vectors and objective
weights, solves a regularized system and clips the numerical increment. A
predicted cost regression is rejected back to zero. Every output explicitly
denies physical validation, runtime execution, promotion and hardware
authorization. State meaning, temporal slew limits, frozen-skill protection,
actual model validity and physical replay belong downstream. No robot order,
football rule or simulator is embedded. Seven focused tests pass; the complete
Growth suite now has 365 passing tests, with the same existing plugin warning.
