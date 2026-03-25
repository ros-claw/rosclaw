"""SO101 Simulation Demo - VLA control in MuJoCo simulation.

This demo shows the complete VLA pipeline:
1. User provides natural language instruction
2. VLA generates action sequence from camera image
3. Digital Twin validates in simulation
4. Actions are visualized and can be replayed
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rosclaw_vla import VLAService, VLAConfig

logger = logging.getLogger(__name__)


class SO101SimDemo:
    """SO101 Robot Simulation Demo with VLA control."""

    def __init__(
        self,
        vla_config: VLAConfig | None = None,
        use_gui: bool = True,
        record_video: bool = False,
    ):
        """Initialize demo.

        Args:
            vla_config: VLA service configuration.
            use_gui: Enable MuJoCo GUI.
            record_video: Record simulation video.
        """
        self.vla_config = vla_config or VLAConfig()
        self.use_gui = use_gui
        self.record_video = record_video

        self.vla_service: VLAService | None = None
        self.sim_env = None
        self.episodes: list[dict] = []

    async def initialize(self) -> None:
        """Initialize VLA service and simulation."""
        logger.info("Initializing SO101 Sim Demo...")

        # Initialize VLA service
        self.vla_service = VLAService(self.vla_config)
        await self.vla_service.initialize()

        # Initialize MuJoCo simulation
        try:
            self.sim_env = self._create_sim_env()
            logger.info("Simulation environment created")
        except ImportError as e:
            logger.warning(f"Could not create sim environment: {e}")
            logger.info("Running in VLA-only mode")

        logger.info("Demo initialized successfully")

    def _create_sim_env(self):
        """Create MuJoCo simulation environment."""
        try:
            import mujoco
            import mujoco.viewer

            # Load SO101 robot model
            # Using standard LeRobot SO101 XML or similar
            xml_path = Path(__file__).parent / "models" / "so101.xml"

            if not xml_path.exists():
                # Create minimal SO101 model
                xml_str = self._get_default_so101_xml()
                model = mujoco.MjModel.from_xml_string(xml_str)
            else:
                model = mujoco.MjModel.from_xml_path(str(xml_path))

            data = mujoco.MjData(model)

            return {
                "model": model,
                "data": data,
                "mujoco": mujoco,
            }

        except ImportError:
            logger.error("MuJoCo not installed. Install with: pip install mujoco")
            raise

    def _get_default_so101_xml(self) -> str:
        """Get default SO101 robot XML for MuJoCo."""
        return """
<mujoco model="so101">
  <compiler angle="radian" meshdir="." autolimits="true"/>

  <option timestep="0.002" iterations="50" solver="Newton" gravity="0 0 -9.81">
    <flag warmstart="enable"/>
  </option>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.8 1.0" rgb2="0.2 0.4 0.8" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <material name="robot" rgba="0.8 0.6 0.4 1"/>
  </asset>

  <worldbody>
    <light directional="true" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" pos="0 0 5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" material="grid"/>

    <!-- Simple robot base -->
    <body name="base" pos="0 0 0.1">
      <freejoint name="base_joint"/>
      <geom type="cylinder" size="0.1 0.05" rgba="0.3 0.3 0.3 1"/>

      <!-- Arm link 1 -->
      <body name="link1" pos="0 0 0.05">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" material="robot"/>

        <!-- Arm link 2 -->
        <body name="link2" pos="0.3 0 0">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.025" material="robot"/>

          <!-- Arm link 3 -->
          <body name="link3" pos="0 0 0.3">
            <joint name="joint3" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.02" material="robot"/>

            <!-- Gripper base -->
            <body name="gripper" pos="0 0 0.25">
              <joint name="joint4" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
              <geom type="box" size="0.04 0.02 0.01" rgba="0.5 0.5 0.5 1"/>

              <!-- Camera -->
              <camera name="wrist_cam" pos="0.05 0 0.02" xyaxes="0 -1 0 0 0 1" fovy="60"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Red cube for manipulation -->
    <body name="red_cube" pos="0.4 0 0.025">
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" rgba="1 0 0 1" mass="0.1"/>
    </body>

    <!-- Green cube -->
    <body name="green_cube" pos="0.3 0.2 0.025">
      <freejoint name="green_cube_joint"/>
      <geom name="green_cube_geom" type="box" size="0.025 0.025 0.025" rgba="0 1 0 1" mass="0.1"/>
    </body>

    <!-- Blue cube -->
    <body name="blue_cube" pos="0.3 -0.2 0.025">
      <freejoint name="blue_cube_joint"/>
      <geom name="blue_cube_geom" type="box" size="0.025 0.025 0.025" rgba="0 0 1 1" mass="0.1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="joint1" name="motor1" gear="100" ctrllimited="true" ctrlrange="-100 100"/>
    <motor joint="joint2" name="motor2" gear="100" ctrllimited="true" ctrlrange="-100 100"/>
    <motor joint="joint3" name="motor3" gear="100" ctrllimited="true" ctrlrange="-100 100"/>
    <motor joint="joint4" name="motor4" gear="50" ctrllimited="true" ctrlrange="-50 50"/>
  </actuator>
</mujoco>
"""

    async def run_episode(
        self,
        instruction: str,
        max_steps: int = 100,
        step_delay: float = 0.05,
    ) -> dict:
        """Run a single episode.

        Args:
            instruction: Natural language instruction.
            max_steps: Maximum steps to run.
            step_delay: Delay between steps.

        Returns:
            Episode data dictionary.
        """
        logger.info(f"Starting episode: {instruction}")

        episode_data = {
            "instruction": instruction,
            "observations": [],
            "actions": [],
            "vla_outputs": [],
        }

        for step in range(max_steps):
            # Get observation (camera image)
            obs = self._get_observation()
            episode_data["observations"].append(obs)

            # Get VLA prediction
            vla_output = await self.vla_service.predict(
                image=obs["image"],
                instruction=instruction,
                proprioception=obs["proprioception"],
            )

            episode_data["vla_outputs"].append({
                "actions": vla_output.actions.tolist(),
                "confidence": vla_output.confidence,
                "latency_ms": vla_output.latency_ms,
            })

            # Apply first action
            action = vla_output.actions[0]
            episode_data["actions"].append(action.tolist())

            if self.sim_env:
                self._apply_action(action)

            logger.info(f"Step {step}: action={action[:3]}, conf={vla_output.confidence:.2f}")

            # Check if done (simple heuristic)
            if self._check_success(instruction, obs):
                logger.info("Task completed successfully!")
                break

            await asyncio.sleep(step_delay)

        self.episodes.append(episode_data)
        return episode_data

    def _get_observation(self) -> dict:
        """Get current observation from simulation."""
        if not self.sim_env:
            # Return dummy observation
            return {
                "image": Image.new("RGB", (224, 224), color="gray"),
                "proprioception": np.zeros(7),
            }

        # Render from camera
        model = self.sim_env["model"]
        data = self.sim_env["data"]
        mujoco = self.sim_env["mujoco"]

        # Step simulation physics
        mujoco.mj_step(model, data)

        # Render camera view
        renderer = mujoco.Renderer(model, height=224, width=224)
        renderer.update_scene(data, camera="wrist_cam")
        pixels = renderer.render()

        # Get proprioception (joint positions)
        qpos = data.qpos[:7].copy()

        return {
            "image": Image.fromarray(pixels),
            "proprioception": qpos,
        }

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply action to simulation."""
        if not self.sim_env:
            return

        model = self.sim_env["model"]
        data = self.sim_env["data"]

        # Apply action as control
        # action: [x, y, z, roll, pitch, yaw, gripper]
        # Convert to joint commands (simplified)
        ctrl = np.zeros(model.nu)
        ctrl[:min(len(action), model.nu)] = action[:model.nu]
        data.ctrl[:] = ctrl * 10  # Scale for simulation

    def _check_success(self, instruction: str, obs: dict) -> bool:
        """Check if task is complete."""
        # Simple heuristic based on instruction
        instruction_lower = instruction.lower()

        if "pick" in instruction_lower and "red" in instruction_lower:
            # Check if red cube is lifted
            # This would require tracking cube position
            pass

        return False

    async def shutdown(self) -> None:
        """Shutdown demo."""
        logger.info("Shutting down demo...")

        if self.vla_service:
            await self.vla_service.shutdown()

        logger.info("Demo shutdown complete")

    def save_episodes(self, path: Path) -> None:
        """Save episode data to file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.episodes, f, indent=2)

        logger.info(f"Saved {len(self.episodes)} episodes to {path}")


async def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(description="SO101 VLA Simulation Demo")
    parser.add_argument(
        "--instruction",
        type=str,
        default="Pick up the red cube",
        help="Task instruction",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openvla/openvla-7b",
        help="VLA model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/cuda/cpu)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable GUI",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--save-episodes",
        type=str,
        default=None,
        help="Path to save episode data",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create config
    config = VLAConfig(
        model_name=args.model,
        device=args.device,
        streaming=False,
    )

    # Create and run demo
    demo = SO101SimDemo(
        vla_config=config,
        use_gui=not args.no_gui,
    )

    try:
        await demo.initialize()
        await demo.run_episode(
            instruction=args.instruction,
            max_steps=args.max_steps,
        )

        if args.save_episodes:
            demo.save_episodes(Path(args.save_episodes))

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await demo.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
