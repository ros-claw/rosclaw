#!/usr/bin/env python3
"""
ROSClaw V4 End-to-End Demo

Full pipeline: Conversation → VLA → Digital Twin Verification → Robot Execution
Demonstrates the complete 7-layer architecture:
  Layer 7: RL Training (reward model)
  Layer 6: Cognitive Router (confidence-based routing)
  Layer 5: VLA Policy (OpenVLA)
  Layer 4: MCP Layer (robot control)
  Layer 3: Digital Twin (mjlab verification)
  Layer 2: Data Layer (ring buffer)
  Layer 1: Runtime (simulated)

Usage:
  python -m rosclaw_vla.demo.e2e_demo --mode sim --task "pick up the red cube"
  python -m rosclaw_vla.demo.e2e_demo --mode real --robot so101 --task "wave hello"
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DemoConfig:
    """Demo configuration"""
    mode: str = "sim"  # sim or real
    robot_type: str = "so101"
    task: str = "pick up the red cube"
    use_digital_twin: bool = True
    use_vla: bool = True
    safety_threshold: float = 0.8
    max_steps: int = 100
    render: bool = True


class ROSClawDemo:
    """
    ROSClaw V4 End-to-End Demonstration

    Shows the complete flow from natural language task to physical execution,
    with safety verification at each layer.
    """

    def __init__(self, config: DemoConfig):
        self.config = config
        self.layers: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.session_id: Optional[str] = None

    async def initialize(self):
        """Initialize all 7 layers"""
        logger.info("=" * 60)
        logger.info("ROSClaw V4 - Initializing 7-Layer Architecture")
        logger.info("=" * 60)

        # Layer 1: Runtime (simulated for demo)
        logger.info("[Layer 1] Runtime - Initializing ROS 2 runtime...")
        self.layers['runtime'] = self._init_runtime()

        # Layer 2: Data Layer
        logger.info("[Layer 2] Data - Initializing ring buffer...")
        self.layers['data'] = self._init_data_layer()

        # Layer 3: Digital Twin
        if self.config.use_digital_twin:
            logger.info("[Layer 3] Digital Twin - Initializing mjlab simulation...")
            self.layers['digital_twin'] = await self._init_digital_twin()

        # Layer 4: MCP Layer
        logger.info("[Layer 4] MCP - Initializing robot interface...")
        self.layers['mcp'] = await self._init_mcp_layer()

        # Layer 5: VLA Policy
        if self.config.use_vla:
            logger.info("[Layer 5] VLA - Loading OpenVLA policy...")
            self.layers['vla'] = await self._init_vla()

        # Layer 6: Cognitive Router
        logger.info("[Layer 6] Cognitive - Initializing confidence router...")
        self.layers['cognitive'] = self._init_cognitive()

        # Layer 7: RL Training (reward model for evaluation)
        logger.info("[Layer 7] RL - Initializing M-PRM reward model...")
        self.layers['rl'] = self._init_rl()

        logger.info("=" * 60)
        logger.info("All layers initialized successfully!")
        logger.info("=" * 60)

    def _init_runtime(self) -> dict:
        """Initialize Layer 1: Runtime"""
        return {
            'ros_version': '2 (Humble)',
            'dds': 'CycloneDDS',
            'clock': 'ROS Time',
            'status': 'ready'
        }

    def _init_data_layer(self) -> dict:
        """Initialize Layer 2: Data"""
        from rosclaw_core.data.ring_buffer import RingBuffer, EventTrigger

        # Create ring buffer with 60s capacity at 30Hz
        buffer = RingBuffer(capacity=1800, feature_dim=10)

        return {
            'ring_buffer': buffer,
            'triggers': [
                EventTrigger('success', 'Task completed successfully'),
                EventTrigger('collision', 'Collision detected'),
                EventTrigger('timeout', 'Execution timeout'),
            ],
            'status': 'ready'
        }

    async def _init_digital_twin(self) -> Optional[dict]:
        """Initialize Layer 3: Digital Twin"""
        try:
            from rosclaw_sim.digital_twin import DigitalTwin
            from rosclaw_sim.robot_loader import RobotLoader

            # Load robot model
            loader = RobotLoader()
            robot_model = loader.from_eurdf(
                f"{self.config.robot_type}.eurdf.yaml"
            )

            # Create Digital Twin
            twin = DigitalTwin(
                robot_model=robot_model,
                enable_rendering=self.config.render
            )
            await twin.initialize()

            return {
                'twin': twin,
                'loader': loader,
                'verification_count': 0,
                'status': 'ready'
            }
        except Exception as e:
            logger.warning(f"Digital Twin initialization failed: {e}")
            return None

    async def _init_mcp_layer(self) -> dict:
        """Initialize Layer 4: MCP"""
        if self.config.mode == "sim":
            # Use mock MCP for simulation
            return {
                'type': 'mock',
                'robot': self.config.robot_type,
                'tools': ['move_joint', 'move_cartesian', 'gripper_control', 'get_state'],
                'status': 'ready'
            }
        else:
            # Real robot connection
            return {
                'type': 'real',
                'robot': self.config.robot_type,
                'connection': 'pending',
                'status': 'ready'
            }

    async def _init_vla(self) -> Optional[dict]:
        """Initialize Layer 5: VLA"""
        try:
            from rosclaw_vla.policies.openvla import OpenVLAPolicy

            policy = OpenVLAPolicy(
                model_id="openvla/openvla-7b",
                enable_lora=True
            )
            await policy.load()

            return {
                'policy': policy,
                'inference_count': 0,
                'avg_latency_ms': 0.0,
                'status': 'ready'
            }
        except Exception as e:
            logger.warning(f"VLA initialization failed: {e}")
            logger.info("Falling back to primitive actions")
            return None

    def _init_cognitive(self) -> dict:
        """Initialize Layer 6: Cognitive Router"""
        return {
            'router': 'confidence_gated',
            'thresholds': {
                'fast_lane': 0.95,
                'neural_twin': 0.70,
                'digital_twin': 0.50,
                'llm_planning': 0.00
            },
            'decisions': [],
            'status': 'ready'
        }

    def _init_rl(self) -> dict:
        """Initialize Layer 7: RL"""
        try:
            from rosclaw_rl.rewards.mprm import MPRMRewardModel

            reward_model = MPRMRewardModel(
                video_weight=0.3,
                force_weight=0.3,
                tactile_weight=0.2,
                goal_weight=0.2
            )

            return {
                'reward_model': reward_model,
                'episode_rewards': [],
                'status': 'ready'
            }
        except Exception as e:
            logger.warning(f"RL initialization failed: {e}")
            return {'status': 'disabled'}

    async def run_task(self, task_description: str) -> dict:
        """
        Execute a task through all 7 layers

        Args:
            task_description: Natural language task (e.g., "pick up the red cube")

        Returns:
            Execution results and metrics
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"Task: {task_description}")
        logger.info("=" * 60)

        self.session_id = f"demo_{int(time.time())}"
        start_time = time.time()

        # Step 1: Layer 6 - Cognitive Router (Confidence Evaluation)
        logger.info("\n[Layer 6] Cognitive Router - Evaluating task...")
        confidence = await self._evaluate_confidence(task_description)
        routing_decision = self._route_by_confidence(confidence)
        logger.info(f"  Confidence: {confidence:.3f}")
        logger.info(f"  Routing: {routing_decision}")

        # Step 2: Layer 5 - VLA Policy Generation
        logger.info("\n[Layer 5] VLA Policy - Generating actions...")
        actions = await self._generate_actions(task_description, routing_decision)
        logger.info(f"  Generated {len(actions)} actions")

        # Step 3: Layer 3 - Digital Twin Verification
        if self.layers.get('digital_twin') and routing_decision in ['neural_twin', 'digital_twin']:
            logger.info("\n[Layer 3] Digital Twin - Verifying trajectory...")
            verification = await self._verify_trajectory(actions)
            logger.info(f"  Safe: {verification['safe']}")
            logger.info(f"  Estimated success: {verification['success_prob']:.2%}")

            if not verification['safe']:
                logger.warning("  Trajectory unsafe! Aborting.")
                return {'success': False, 'reason': 'unsafe_trajectory'}

        # Step 4: Layer 4 - MCP Execution
        logger.info(f"\n[Layer 4] MCP - Executing on {self.config.mode} robot...")
        execution_result = await self._execute_actions(actions)
        logger.info(f"  Steps executed: {execution_result['steps']}")
        logger.info(f"  Success: {execution_result['success']}")

        # Step 5: Layer 2 - Data Collection
        logger.info("\n[Layer 2] Data - Collecting episode data...")
        await self._collect_data(task_description, actions, execution_result)
        logger.info("  Data saved to ring buffer")

        # Step 6: Layer 7 - RL Evaluation
        if self.layers['rl'].get('status') == 'ready':
            logger.info("\n[Layer 7] RL - Evaluating with M-PRM...")
            reward = await self._evaluate_reward(execution_result)
            logger.info(f"  Reward: {reward:.3f}")

        # Compile results
        elapsed = time.time() - start_time
        results = {
            'session_id': self.session_id,
            'task': task_description,
            'mode': self.config.mode,
            'robot': self.config.robot_type,
            'confidence': confidence,
            'routing': routing_decision,
            'actions_generated': len(actions),
            'execution': execution_result,
            'elapsed_time_sec': elapsed,
            'layers_active': sum(1 for l in self.layers.values() if l.get('status') == 'ready'),
            'success': execution_result.get('success', False)
        }

        logger.info("\n" + "=" * 60)
        logger.info(f"Task Complete - Success: {results['success']}")
        logger.info(f"Elapsed: {elapsed:.2f}s")
        logger.info("=" * 60)

        return results

    async def _evaluate_confidence(self, task: str) -> float:
        """Evaluate task confidence using cognitive router"""
        # Simulate confidence calculation
        # In reality, this would use:
        # - Task familiarity (vector similarity to past tasks)
        # - Environment predictability
        # - VLA policy confidence

        base_confidence = 0.75

        # Task complexity adjustment
        if 'pick' in task.lower():
            base_confidence += 0.15
        if 'red' in task.lower() or 'blue' in task.lower():
            base_confidence += 0.05  # Color detection is reliable
        if 'complex' in task.lower() or 'assemble' in task.lower():
            base_confidence -= 0.20

        return min(0.99, max(0.1, base_confidence))

    def _route_by_confidence(self, confidence: float) -> str:
        """Route based on confidence level"""
        thresholds = self.layers['cognitive']['thresholds']

        if confidence >= thresholds['fast_lane']:
            return 'fast_lane'
        elif confidence >= thresholds['neural_twin']:
            return 'neural_twin'
        elif confidence >= thresholds['digital_twin']:
            return 'digital_twin'
        else:
            return 'llm_planning'

    async def _generate_actions(self, task: str, routing: str) -> list[dict]:
        """Generate actions using VLA or fallback"""
        vla_layer = self.layers.get('vla')

        if vla_layer and vla_layer.get('status') == 'ready':
            # Use VLA policy
            policy = vla_layer['policy']

            # Mock observation (would be real camera image)
            observation = np.zeros((224, 224, 3), dtype=np.uint8)

            try:
                actions = await policy.predict_actions(
                    observation=observation,
                    task=task,
                    num_actions=10
                )
                vla_layer['inference_count'] += 1
                return actions
            except Exception as e:
                logger.warning(f"VLA inference failed: {e}, using fallback")

        # Fallback: Generate primitive actions
        return self._generate_fallback_actions(task)

    def _generate_fallback_actions(self, task: str) -> list[dict]:
        """Generate fallback primitive actions"""
        primitives = []

        if 'pick' in task.lower():
            primitives = [
                {'type': 'move', 'target': 'approach', 'speed': 0.5},
                {'type': 'gripper', 'action': 'open'},
                {'type': 'move', 'target': 'grasp', 'speed': 0.3},
                {'type': 'gripper', 'action': 'close'},
                {'type': 'move', 'target': 'lift', 'speed': 0.5},
            ]
        elif 'place' in task.lower():
            primitives = [
                {'type': 'move', 'target': 'above_target', 'speed': 0.5},
                {'type': 'move', 'target': 'place', 'speed': 0.3},
                {'type': 'gripper', 'action': 'open'},
                {'type': 'move', 'target': 'retract', 'speed': 0.5},
            ]
        elif 'wave' in task.lower():
            primitives = [
                {'type': 'move', 'target': 'wave_up', 'speed': 0.8},
                {'type': 'move', 'target': 'wave_down', 'speed': 0.8},
                {'type': 'move', 'target': 'wave_up', 'speed': 0.8},
                {'type': 'move', 'target': 'home', 'speed': 0.5},
            ]
        else:
            primitives = [
                {'type': 'move', 'target': 'home', 'speed': 0.5},
            ]

        return primitives

    async def _verify_trajectory(self, actions: list[dict]) -> dict:
        """Verify trajectory in Digital Twin"""
        twin_layer = self.layers.get('digital_twin')
        if not twin_layer:
            return {'safe': True, 'success_prob': 0.5}

        twin = twin_layer['twin']

        # Convert actions to trajectory format
        trajectory = self._actions_to_trajectory(actions)

        # Verify in simulation
        result = await twin.verify_trajectory(trajectory)

        twin_layer['verification_count'] += 1

        return {
            'safe': result['collision_free'] and result['within_limits'],
            'success_prob': result.get('success_probability', 0.5),
            'collision_free': result['collision_free'],
            'within_limits': result['within_limits']
        }

    def _actions_to_trajectory(self, actions: list[dict]) -> list[np.ndarray]:
        """Convert actions to trajectory format"""
        # Mock trajectory generation
        trajectory = []
        for i, action in enumerate(actions):
            # Create dummy state vector (joint positions, velocities, etc.)
            state = np.random.randn(10) * 0.1
            trajectory.append(state)
        return trajectory

    async def _execute_actions(self, actions: list[dict]) -> dict:
        """Execute actions through MCP layer"""
        mcp = self.layers['mcp']

        executed = 0
        success = True

        for i, action in enumerate(actions[:self.config.max_steps]):
            logger.debug(f"  Executing action {i+1}: {action}")

            if self.config.mode == "sim":
                # Simulate execution
                await asyncio.sleep(0.1)
                executed += 1
            else:
                # Real robot execution
                try:
                    await self._send_mcp_command(action)
                    executed += 1
                except Exception as e:
                    logger.error(f"Execution failed: {e}")
                    success = False
                    break

        return {
            'steps': executed,
            'total_actions': len(actions),
            'success': success and executed == len(actions)
        }

    async def _send_mcp_command(self, action: dict):
        """Send command via MCP"""
        # Would call actual MCP tools
        pass

    async def _collect_data(self, task: str, actions: list[dict], result: dict):
        """Collect data to ring buffer"""
        data_layer = self.layers['data']
        buffer = data_layer['ring_buffer']

        # Create episode data
        episode_data = {
            'session_id': self.session_id,
            'task': task,
            'actions': actions,
            'result': result,
            'timestamp': time.time()
        }

        # Add to buffer
        buffer.add_episode(episode_data)

        # Trigger event if success/failure
        if result.get('success'):
            buffer.trigger_event('success', episode_data)
        else:
            buffer.trigger_event('failure', episode_data)

    async def _evaluate_reward(self, result: dict) -> float:
        """Evaluate reward using M-PRM"""
        rl_layer = self.layers['rl']
        if rl_layer.get('status') != 'ready':
            return 0.0

        reward_model = rl_layer['reward_model']

        # Mock observations
        video_frames = np.zeros((10, 224, 224, 3))
        force_data = np.random.randn(10, 6) * 0.1

        reward = await reward_model.compute_reward(
            video=video_frames,
            force=force_data,
            goal_achieved=result.get('success', False)
        )

        rl_layer['episode_rewards'].append(reward)
        return reward

    async def shutdown(self):
        """Clean shutdown of all layers"""
        logger.info("\nShutting down ROSClaw V4...")

        if self.layers.get('digital_twin'):
            await self.layers['digital_twin']['twin'].shutdown()

        if self.layers.get('vla'):
            await self.layers['vla']['policy'].unload()

        logger.info("Shutdown complete.")

    def print_summary(self, results: dict):
        """Print execution summary"""
        print("\n" + "=" * 60)
        print("ROSClaw V4 Demo Summary")
        print("=" * 60)
        print(f"Session ID: {results['session_id']}")
        print(f"Task: {results['task']}")
        print(f"Mode: {results['mode']}")
        print(f"Robot: {results['robot']}")
        print(f"\nConfidence: {results['confidence']:.3f}")
        print(f"Routing: {results['routing']}")
        print(f"Actions: {results['actions_generated']}")
        print(f"\nExecution Steps: {results['execution']['steps']}")
        print(f"Success: {'✓' if results['success'] else '✗'}")
        print(f"Time: {results['elapsed_time_sec']:.2f}s")
        print(f"Layers Active: {results['layers_active']}/7")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description='ROSClaw V4 End-to-End Demo')
    parser.add_argument('--mode', choices=['sim', 'real'], default='sim',
                       help='Execution mode: simulation or real robot')
    parser.add_argument('--robot', default='so101',
                       help='Robot type (so101, ur5, g1)')
    parser.add_argument('--task', default='pick up the red cube',
                       help='Task description')
    parser.add_argument('--no-dt', action='store_true',
                       help='Disable Digital Twin verification')
    parser.add_argument('--no-vla', action='store_true',
                       help='Disable VLA policy (use primitives)')
    parser.add_argument('--steps', type=int, default=100,
                       help='Maximum execution steps')

    args = parser.parse_args()

    # Create configuration
    config = DemoConfig(
        mode=args.mode,
        robot_type=args.robot,
        task=args.task,
        use_digital_twin=not args.no_dt,
        use_vla=not args.no_vla,
        max_steps=args.steps
    )

    # Run demo
    demo = ROSClawDemo(config)

    try:
        await demo.initialize()
        results = await demo.run_task(args.task)
        demo.print_summary(results)

        # Save results
        output_path = Path(f"/tmp/rosclaw_demo_{results['session_id']}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.exception("Demo failed")
        raise
    finally:
        await demo.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
