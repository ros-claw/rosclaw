"""Explicit larger inner-optimization budget over ONE frozen proposal batch.

No old default or artifact contract changes. More accepted updates are not new
rollouts, online RL, physical retention, deployment, or execution permission.
The original solver retains its single initial behavior reference and KL cap.
"""

from dataclasses import dataclass, fields
from typing import Any

from rosclaw.growth.proposal_advantage_regression import (
    ProposalAdvantageRegressionConfig,
    fit_proposal_advantage_residual,
)


@dataclass(frozen=True)
class ExtendedProposalRegressionConfig(ProposalAdvantageRegressionConfig):
    steps: int = 1600

    def validate(self) -> None:
        if type(self.steps) is not int or not 1 <= self.steps <= 2560:
            raise ValueError("explicit bounded extended inner-optimizer budget required")
        # Validate EVERY original field; only the separately bounded step limit
        # is extended. Do not widen caps, likelihood, device, KL or authority.
        original = {
            f.name: getattr(self, f.name) for f in fields(ProposalAdvantageRegressionConfig)
        }
        original["steps"] = min(self.steps, 160)
        ProposalAdvantageRegressionConfig(**original).validate()


def fit_extended_proposal_residual(
    *,
    layers: Any,
    context: Any,
    baseline: Any,
    gates: Any,
    actions: Any,
    marginal_std: Any,
    first: Any,
    advantages: Any,
    old_log_probability: Any,
    config: ExtendedProposalRegressionConfig,
    sample_weights: Any | None = None,
) -> dict[str, Any]:
    """Same validated solver, one reference, explicit non-legacy receipt.

    Early rejection/convergence remains enabled. The accepted-step count is
    not the number of attempted optimizer steps or independent samples.
    Callers must separately authenticate the data and physically test output.
    """
    if type(config) is not ExtendedProposalRegressionConfig:
        raise ValueError("explicit extended proposal configuration required")
    config.validate()
    result = fit_proposal_advantage_residual(
        layers=layers,
        context=context,
        baseline=baseline,
        gates=gates,
        actions=actions,
        marginal_std=marginal_std,
        first=first,
        advantages=advantages,
        old_log_probability=old_log_probability,
        config=config,
        sample_weights=sample_weights,
    )
    result["base_numeric_algorithm"] = result["algorithm"]
    result["algorithm"] = "EXPLICIT_EXTENDED_PROPOSAL_REGRESSION_V1"
    result["requested_inner_optimizer_steps"] = config.steps
    result["single_initial_behavior_reference"] = True
    result["kl_budget_reset_between_passes"] = False
    result["accepted_step_count_not_total_attempts"] = True
    result["additional_physical_episodes_executed"] = 0
    result["online_rl_claimed"] = False
    return result
