"""Domain randomization for sim-to-real transfer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from rosclaw_sim.mjlab_env import MjlabEnv


@dataclass
class RandomizationRange:
    """Range for randomizing a parameter.

    Attributes:
        low: Lower bound of the range.
        high: Upper bound of the range.
        distribution: Distribution type ("uniform", "log_uniform", "gaussian").
    """

    low: float
    high: float
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"

    def sample(self, shape: tuple[int, ...], device: str = "cpu") -> torch.Tensor:
        """Sample from the distribution.

        Args:
            shape: Shape of the output tensor.
            device: Device for the tensor.

        Returns:
            Sampled tensor.
        """
        if self.distribution == "uniform":
            return torch.rand(shape, device=device) * (self.high - self.low) + self.low
        elif self.distribution == "log_uniform":
            log_low = np.log(self.low) if self.low > 0 else -10
            log_high = np.log(self.high)
            return torch.exp(
                torch.rand(shape, device=device) * (log_high - log_low) + log_low
            )
        elif self.distribution == "gaussian":
            mean = (self.low + self.high) / 2
            std = (self.high - self.low) / 4  # 95% within range
            return torch.randn(shape, device=device) * std + mean
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass
class RandomizationConfig:
    """Configuration for domain randomization.

    Domain randomization helps bridge the sim-to-real gap by randomizing
    physical parameters during training, making policies more robust.

    Attributes:
        body_mass: Randomization range for body masses.
        body_inertia: Randomization range for body inertia.
        joint_friction: Randomization range for joint friction.
        joint_damping: Randomization range for joint damping.
        joint_armature: Randomization range for joint armature.
        actuator_gain: Randomization range for actuator gains (kp).
        actuator_bias: Randomization range for actuator bias.
        gravity: Randomization range for gravity vector.
        apply_on_reset: Whether to apply randomization on reset.
        apply_during_sim: Whether to apply randomization during simulation.
        randomization_frequency: How often to randomize (in steps).
    """

    # Mass properties
    body_mass: RandomizationRange | None = None
    body_inertia: RandomizationRange | None = None

    # Joint properties
    joint_friction: RandomizationRange | None = None
    joint_damping: RandomizationRange | None = None
    joint_armature: RandomizationRange | None = None

    # Actuator properties
    actuator_gain: RandomizationRange | None = None  # kp
    actuator_bias: RandomizationRange | None = None  # kd/effects

    # Environment properties
    gravity: RandomizationRange | None = None

    # Randomization schedule
    apply_on_reset: bool = True
    apply_during_sim: bool = False
    randomization_frequency: int = 100

    @classmethod
    def default_robot_config(cls) -> RandomizationConfig:
        """Create default configuration for robot manipulation.

        These ranges are conservative and suitable for most robot arms.
        """
        return cls(
            body_mass=RandomizationRange(0.95, 1.05),  # +/- 5%
            body_inertia=RandomizationRange(0.9, 1.1),  # +/- 10%
            joint_friction=RandomizationRange(0.0, 0.1),  # Small friction
            joint_damping=RandomizationRange(0.9, 1.1),  # +/- 10%
            joint_armature=RandomizationRange(0.95, 1.05),  # +/- 5%
            actuator_gain=RandomizationRange(0.9, 1.1),  # +/- 10%
            actuator_bias=RandomizationRange(0.95, 1.05),  # +/- 5%
            apply_on_reset=True,
            apply_during_sim=False,
        )

    @classmethod
    def aggressive_config(cls) -> RandomizationConfig:
        """Create aggressive randomization for robust policies."""
        return cls(
            body_mass=RandomizationRange(0.8, 1.2),  # +/- 20%
            body_inertia=RandomizationRange(0.7, 1.3),  # +/- 30%
            joint_friction=RandomizationRange(0.0, 0.3),  # Higher friction range
            joint_damping=RandomizationRange(0.7, 1.3),  # +/- 30%
            joint_armature=RandomizationRange(0.8, 1.2),  # +/- 20%
            actuator_gain=RandomizationRange(0.8, 1.2),  # +/- 20%
            actuator_bias=RandomizationRange(0.8, 1.2),  # +/- 20%
            gravity=RandomizationRange(9.0, 10.0),  # Variable gravity
            apply_on_reset=True,
            apply_during_sim=True,
            randomization_frequency=50,
        )

    @classmethod
    def from_preset(cls, preset: Literal["default", "aggressive", "none"]) -> RandomizationConfig:
        """Create configuration from preset name.

        Args:
            preset: Preset name ("default", "aggressive", "none").

        Returns:
            RandomizationConfig for the preset.
        """
        if preset == "default":
            return cls.default_robot_config()
        elif preset == "aggressive":
            return cls.aggressive_config()
        elif preset == "none":
            return cls()  # Empty config
        else:
            raise ValueError(f"Unknown preset: {preset}")


class DomainRandomization:
    """Domain randomization for sim-to-real transfer.

    Applies randomization to simulation parameters to improve policy
    robustness when deployed on real robots.

    Example:
        >>> config = RandomizationConfig.default_robot_config()
        >>> dr = DomainRandomization(config)
        >>>
        >>> # Apply randomization
        >>> dr.apply(env)
        >>>
        >>> # Apply randomization to specific environments
        >>> dr.apply(env, env_ids=torch.tensor([0, 2, 4]))
    """

    def __init__(self, config: RandomizationConfig | None = None):
        self.config = config or RandomizationConfig()
        self._step_count = 0
        self._default_values: dict[str, torch.Tensor] = {}

    def apply(
        self,
        env: MjlabEnv,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Apply domain randomization to the environment.

        Args:
            env: Mjlab environment to randomize.
            env_ids: Specific environment IDs to randomize. If None, all.

        Returns:
            Dictionary of applied randomizations.
        """
        if not env.is_initialized:
            raise RuntimeError("Environment must be initialized before applying DR")

        applied = {}

        # Determine which environments to randomize
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)

        # Randomize body mass
        if self.config.body_mass is not None:
            applied["body_mass"] = self._randomize_body_mass(env, env_ids)

        # Randomize body inertia
        if self.config.body_inertia is not None:
            applied["body_inertia"] = self._randomize_body_inertia(env, env_ids)

        # Randomize joint friction
        if self.config.joint_friction is not None:
            applied["joint_friction"] = self._randomize_joint_friction(env, env_ids)

        # Randomize joint damping
        if self.config.joint_damping is not None:
            applied["joint_damping"] = self._randomize_joint_damping(env, env_ids)

        # Randomize joint armature
        if self.config.joint_armature is not None:
            applied["joint_armature"] = self._randomize_joint_armature(env, env_ids)

        # Randomize actuator gains
        if self.config.actuator_gain is not None:
            applied["actuator_gain"] = self._randomize_actuator_gain(env, env_ids)

        # Randomize gravity
        if self.config.gravity is not None:
            applied["gravity"] = self._randomize_gravity(env, env_ids)

        self._step_count += 1

        return applied

    def _randomize_body_mass(
        self, env: MjlabEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Randomize body masses."""
        num_bodies = env._env.sim.mj_model.nbody
        shape = (len(env_ids), num_bodies)

        samples = self.config.body_mass.sample(shape, env.device)

        # Get default masses
        default_mass = torch.tensor(
            env._env.sim.mj_model.body_mass.copy(),
            device=env.device,
            dtype=torch.float32,
        )

        # Apply randomization
        env._env.sim.model.body_mass[env_ids] = default_mass.unsqueeze(0) * samples

        # Recompute derived quantities
        env._env.sim.recompute_constants("set_const")

        return samples

    def _randomize_body_inertia(
        self, env: MjlabEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Randomize body inertia."""
        num_bodies = env._env.sim.mj_model.nbody
        # Inertia is (nbody, 3) for principal moments
        shape = (len(env_ids), num_bodies, 3)

        samples = self.config.body_inertia.sample(shape, env.device)

        default_inertia = torch.tensor(
            env._env.sim.mj_model.body_inertia.copy(),
            device=env.device,
            dtype=torch.float32,
        )

        env._env.sim.model.body_inertia[env_ids] = default_inertia.unsqueeze(0) * samples

        env._env.sim.recompute_constants("set_const")

        return samples

    def _randomize_joint_friction(
        self, env: MjlabEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Randomize joint friction."""
        num_joints = env._env.sim.mj_model.njnt
        shape = (len(env_ids), num_joints)

        samples = self.config.joint_friction.sample(shape, env.device)

        default_friction = torch.tensor(
            env._env.sim.mj_model.jnt_frictionloss.copy(),
            device=env.device,
            dtype=torch.float32,
        )

        # Add friction (frictionloss is additive)
        env._env.sim.model.jnt_frictionloss[env_ids] = default_friction.unsqueeze(0) + samples

        return samples

    def _randomize_joint_damping(
        self, env: MjlabEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Randomize joint damping."""
        num_joints = env._env.sim.mj_model.njnt
        shape = (len(env_ids), num_joints)

        samples = self.config.joint_damping.sample(shape, env.device)

        default_damping = torch.tensor(
            env._env.sim.mj_model.dof_damping.copy(),
            device=env.device,
            dtype=torch.float32,
        )

        env._env.sim.model.dof_damping[env_ids] = default_damping.unsqueeze(0) * samples

        return samples

    def _randomize_joint_armature(
        self, env: MjlabEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Randomize joint armature."""
        num_joints = env._env.sim.mj_model.njnt
        shape = (len(env_ids), num_joints)

        samples = self.config.joint_armature.sample(shape, env.device)

        default_armature = torch.tensor(
            env._env.sim.mj_model.dof_armature.copy(),
            device=env.device,
            dtype=torch.float32,
        )

        env._env.sim.model.dof_armature[env_ids] = default_armature.unsqueeze(0) * samples

        env._env.sim.recompute_constants("set_const_0")

        return samples

    def _randomize_actuator_gain(
        self, env: MjlabEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Randomize actuator gains (kp)."""
        num_actuators = env._env.sim.mj_model.nu
        shape = (len(env_ids), num_actuators)

        samples = self.config.actuator_gain.sample(shape, env.device)

        # actuator_gainprm is (nu, 3) where first element is kp
        default_gain = torch.tensor(
            env._env.sim.mj_model.actuator_gainprm[:, 0].copy(),
            device=env.device,
            dtype=torch.float32,
        )

        # Only randomize first parameter (kp)
        env._env.sim.model.actuator_gainprm[env_ids, :, 0] = (
            default_gain.unsqueeze(0) * samples
        )

        return samples

    def _randomize_gravity(
        self, env: MjlabEnv, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Randomize gravity."""
        # Gravity is shared across environments, so we sample once
        sample = self.config.gravity.sample((), env.device).item()

        # Standard gravity is (0, 0, -9.81)
        env._env.sim.model.opt.gravity[2] = -sample

        return torch.tensor([sample], device=env.device)

    def should_randomize(self) -> bool:
        """Check if randomization should be applied on this step."""
        if not self.config.apply_during_sim:
            return False
        return self._step_count % self.config.randomization_frequency == 0

    def reset(self) -> None:
        """Reset the randomization step counter."""
        self._step_count = 0


class CurriculumDomainRandomization(DomainRandomization):
    """Domain randomization with curriculum learning.

    Starts with small randomization ranges and gradually increases them
    as training progresses.

    Example:
        >>> base_config = RandomizationConfig.default_robot_config()
        >>> dr = CurriculumDomainRandomization(base_config)
        >>>
        >>> # Update curriculum progress (0.0 to 1.0)
        >>> dr.set_progress(0.5)  # Halfway through curriculum
        >>> dr.apply(env)
    """

    def __init__(
        self,
        final_config: RandomizationConfig,
        initial_factor: float = 0.1,
        total_iterations: int = 10000,
    ):
        super().__init__(final_config)
        self.final_config = final_config
        self.initial_factor = initial_factor
        self.total_iterations = total_iterations
        self.current_iteration = 0

    def set_progress(self, progress: float) -> None:
        """Set curriculum progress (0.0 to 1.0).

        Args:
            progress: Progress through curriculum (0.0 = start, 1.0 = full).
        """
        self.current_iteration = int(progress * self.total_iterations)

    def update(self) -> None:
        """Update iteration counter and adjust randomization strength."""
        self.current_iteration = min(self.current_iteration + 1, self.total_iterations)

        # Calculate interpolation factor
        progress = self.current_iteration / self.total_iterations
        factor = self.initial_factor + (1.0 - self.initial_factor) * progress

        # Scale all randomization ranges
        self._scale_config(factor)

    def _scale_config(self, factor: float) -> None:
        """Scale randomization ranges by factor."""
        for attr_name in [
            "body_mass",
            "body_inertia",
            "joint_friction",
            "joint_damping",
            "joint_armature",
            "actuator_gain",
            "actuator_bias",
        ]:
            final_range = getattr(self.final_config, attr_name)
            if final_range is not None:
                # Scale the range
                center = (final_range.low + final_range.high) / 2
                half_range = (final_range.high - final_range.low) / 2 * factor

                new_range = RandomizationRange(
                    low=center - half_range,
                    high=center + half_range,
                    distribution=final_range.distribution,
                )
                setattr(self.config, attr_name, new_range)
