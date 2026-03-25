"""SO101 Real Robot Demo - VLA control on physical robot.

This demo runs VLA inference on a physical SO101 robot.
Requires robot hardware connection via RoboClaw/ROS.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from rosclaw_vla import VLAService, VLAConfig

logger = logging.getLogger(__name__)


class SO101RealDemo:
    """SO101 Real Robot Demo with VLA control."""

    def __init__(
        self,
        vla_config: VLAConfig | None = None,
        robot_ip: str = "localhost",
        camera_id: int = 0,
    ):
        """Initialize demo.

        Args:
            vla_config: VLA service configuration.
            robot_ip: Robot controller IP address.
            camera_id: Camera device ID.
        """
        self.vla_config = vla_config or VLAConfig()
        self.robot_ip = robot_ip
        self.camera_id = camera_id

        self.vla_service: VLAService | None = None
        self.robot_interface = None
        self.camera = None

    async def initialize(self) -> None:
        """Initialize VLA service and robot connection."""
        logger.info("Initializing SO101 Real Robot Demo...")

        # Initialize VLA service
        self.vla_service = VLAService(self.vla_config)
        await self.vla_service.initialize()

        # Initialize robot interface
        try:
            self.robot_interface = await self._connect_robot()
            logger.info(f"Connected to robot at {self.robot_ip}")
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            raise

        # Initialize camera
        try:
            self.camera = await self._init_camera()
            logger.info(f"Camera initialized (ID: {self.camera_id})")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise

        # Set VLA callback for logging
        self.vla_service.set_callback(self._on_vla_output)

        logger.info("Demo initialized successfully")

    async def _connect_robot(self):
        """Connect to robot controller."""
        # Try different robot interfaces in order of preference

        # 1. Try ROS2/ROS control
        try:
            return await self._connect_ros()
        except Exception as e:
            logger.debug(f"ROS connection failed: {e}")

        # 2. Try direct RoboClaw
        try:
            return await self._connect_roboclaw()
        except Exception as e:
            logger.debug(f"RoboClaw connection failed: {e}")

        # 3. Try MCP server
        try:
            return await self._connect_mcp()
        except Exception as e:
            logger.debug(f"MCP connection failed: {e}")

        raise RuntimeError("Could not connect to robot via any interface")

    async def _connect_ros(self):
        """Connect via ROS/ROS2."""
        import rclpy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        rclpy.init()
        node = rclpy.create_node("vla_demo")

        # Publisher for commands
        cmd_pub = node.create_publisher(
            Float64MultiArray,
            "/so101/joint_commands",
            10,
        )

        # Subscriber for state
        joint_states = {}

        def state_callback(msg):
            joint_states["positions"] = list(msg.position)
            joint_states["velocities"] = list(msg.velocity)

        node.create_subscription(
            JointState,
            "/so101/joint_states",
            state_callback,
            10,
        )

        return {
            "type": "ros2",
            "node": node,
            "cmd_pub": cmd_pub,
            "joint_states": joint_states,
        }

    async def _connect_roboclaw(self):
        """Connect via direct RoboClaw interface."""
        # This would use the roboclaw driver
        logger.info("Using RoboClaw direct connection")

        # Placeholder - actual implementation would import roboclaw library
        return {
            "type": "roboclaw",
            "connected": True,
        }

    async def _connect_mcp(self):
        """Connect via MCP server."""
        # This would use the rosclaw_mcp client
        logger.info("Using MCP server connection")

        return {
            "type": "mcp",
            "connected": True,
        }

    async def _init_camera(self):
        """Initialize camera."""
        try:
            import cv2

            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera {self.camera_id}")

            return {
                "type": "opencv",
                "cap": cap,
            }

        except ImportError:
            logger.warning("OpenCV not available, trying alternative camera")
            return await self._init_realsense()

    async def _init_realsense(self):
        """Initialize Intel RealSense camera."""
        try:
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            pipeline.start(config)

            return {
                "type": "realsense",
                "pipeline": pipeline,
            }

        except ImportError:
            raise RuntimeError("No camera backend available")

    def _get_observation(self) -> dict:
        """Get current observation from robot."""
        # Get camera image
        image = self._get_camera_image()

        # Get proprioception
        proprioception = self._get_proprioception()

        return {
            "image": image,
            "proprioception": proprioception,
        }

    def _get_camera_image(self) -> Image.Image:
        """Get image from camera."""
        if self.camera["type"] == "opencv":
            import cv2
            ret, frame = self.camera["cap"].read()
            if not ret:
                raise RuntimeError("Failed to capture frame")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)

        elif self.camera["type"] == "realsense":
            import pyrealsense2 as rs
            frames = self.camera["pipeline"].wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Failed to get color frame")
            image = np.asanyarray(color_frame.get_data())
            return Image.fromarray(image)

        else:
            raise RuntimeError(f"Unknown camera type: {self.camera['type']}")

    def _get_proprioception(self) -> np.ndarray:
        """Get robot proprioceptive state."""
        if self.robot_interface["type"] == "ros2":
            states = self.robot_interface["joint_states"]
            if "positions" in states:
                return np.array(states["positions"])

        # Default zeros
        return np.zeros(7)

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply action to robot."""
        # Apply safety limits
        action = self._safety_filter(action)

        if self.robot_interface["type"] == "ros2":
            from std_msgs.msg import Float64MultiArray

            msg = Float64MultiArray()
            msg.data = action.tolist()
            self.robot_interface["cmd_pub"].publish(msg)

        elif self.robot_interface["type"] == "roboclaw":
            # Send to roboclaw
            pass

        elif self.robot_interface["type"] == "mcp":
            # Send via MCP
            pass

    def _safety_filter(self, action: np.ndarray) -> np.ndarray:
        """Apply safety filtering to actions."""
        # Clip to safe ranges
        safe_action = action.copy()

        # Position limits (example values for SO101)
        position_limits = [
            (-2.5, 2.5),   # Base rotation
            (-1.5, 1.5),   # Shoulder
            (-1.5, 1.5),   # Elbow
            (-2.5, 2.5),   # Wrist rotation
            (-1.0, 1.0),   # Wrist flex
            (-1.5, 1.5),   # Gripper
        ]

        for i, (min_val, max_val) in enumerate(position_limits):
            if i < len(safe_action):
                safe_action[i] = np.clip(safe_action[i], min_val, max_val)

        # Velocity limiting
        max_vel = 0.5  # rad/s
        safe_action = np.clip(safe_action, -max_vel, max_vel)

        return safe_action

    def _on_vla_output(self, output) -> None:
        """Callback for VLA outputs."""
        logger.debug(
            f"VLA output: latency={output.latency_ms:.1f}ms, "
            f"confidence={output.confidence:.2f}"
        )

    async def run_episode(
        self,
        instruction: str,
        max_steps: int = 100,
        step_delay: float = 0.1,
    ) -> dict:
        """Run a single episode on real robot.

        Args:
            instruction: Natural language instruction.
            max_steps: Maximum steps.
            step_delay: Delay between steps.

        Returns:
            Episode data.
        """
        logger.info(f"Starting episode: {instruction}")
        logger.warning("REAL ROBOT MODE - Ensure safety measures are in place!")

        # Safety confirmation
        input("Press ENTER to start (Ctrl+C to abort)...")

        episode_data = {
            "instruction": instruction,
            "observations": [],
            "actions": [],
            "vla_outputs": [],
        }

        try:
            for step in range(max_steps):
                # Get observation
                obs = self._get_observation()

                # Get VLA prediction
                vla_output = await self.vla_service.predict(
                    image=obs["image"],
                    instruction=instruction,
                    proprioception=obs["proprioception"],
                )

                # Apply first action
                action = vla_output.actions[0]
                self._apply_action(action)

                # Log
                logger.info(
                    f"Step {step}: action={action[:3]}, "
                    f"latency={vla_output.latency_ms:.1f}ms"
                )

                # Store data
                episode_data["actions"].append(action.tolist())
                episode_data["vla_outputs"].append({
                    "latency_ms": vla_output.latency_ms,
                    "confidence": vla_output.confidence,
                })

                await asyncio.sleep(step_delay)

        except KeyboardInterrupt:
            logger.info("Episode interrupted")

        finally:
            # Stop robot
            self._apply_action(np.zeros(7))

        return episode_data

    async def shutdown(self) -> None:
        """Shutdown demo."""
        logger.info("Shutting down...")

        if self.vla_service:
            await self.vla_service.shutdown()

        if self.camera:
            if self.camera["type"] == "opencv":
                self.camera["cap"].release()
            elif self.camera["type"] == "realsense":
                self.camera["pipeline"].stop()

        if self.robot_interface and self.robot_interface["type"] == "ros2":
            import rclpy
            self.robot_interface["node"].destroy_node()
            rclpy.shutdown()

        logger.info("Shutdown complete")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SO101 Real Robot VLA Demo")
    parser.add_argument(
        "--instruction",
        type=str,
        default="Pick up the red cube",
        help="Task instruction",
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        default="localhost",
        help="Robot IP address",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device ID",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openvla/openvla-7b",
        help="VLA model name",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = VLAConfig(
        model_name=args.model,
        device="auto",
    )

    demo = SO101RealDemo(
        vla_config=config,
        robot_ip=args.robot_ip,
        camera_id=args.camera_id,
    )

    try:
        await demo.initialize()
        await demo.run_episode(
            instruction=args.instruction,
            max_steps=args.max_steps,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        await demo.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
