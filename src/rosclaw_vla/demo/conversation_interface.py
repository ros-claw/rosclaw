"""Conversation Interface - Natural language interface for robot control.

Provides an interactive CLI/GUI for users to control robots using
natural language instructions with VLA models.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from rosclaw_vla import VLAService, VLAConfig

logger = logging.getLogger(__name__)


class ConversationInterface:
    """Interactive conversation interface for VLA robot control.

    Features:
    - Natural language task specification
    - Real-time feedback on robot state
    - Multi-turn conversations for complex tasks
    - Safety confirmation for real robot execution
    """

    # Common task templates
    TASK_TEMPLATES = {
        "pick": [
            "Pick up the {object}",
            "Grasp the {object} and lift it",
            "Pick the {object} from the table",
        ],
        "place": [
            "Place the {object} on the {location}",
            "Put the {object} at {location}",
            "Move the {object} to the {location}",
        ],
        "push": [
            "Push the {object} to the {location}",
            "Slide the {object} toward the {location}",
        ],
        "stack": [
            "Stack the {object1} on top of the {object2}",
            "Place {object1} on {object2}",
        ],
    }

    # Safety keywords that require confirmation
    SAFETY_KEYWORDS = ["fast", "quick", "rapid", "force", "hard", "slam"]

    def __init__(
        self,
        vla_config: VLAConfig | None = None,
        sim_mode: bool = True,
        robot_connector: Callable | None = None,
    ):
        """Initialize conversation interface.

        Args:
            vla_config: VLA service configuration.
            sim_mode: Run in simulation mode (True) or real robot (False).
            robot_connector: Optional custom robot connection function.
        """
        self.vla_config = vla_config or VLAConfig()
        self.sim_mode = sim_mode
        self.robot_connector = robot_connector

        self.vla_service: VLAService | None = None
        self.robot = None
        self.camera = None
        self.conversation_history: list[dict] = []

    async def initialize(self) -> None:
        """Initialize the interface."""
        print("🤖 ROSClaw VLA Conversation Interface")
        print("=" * 50)

        # Initialize VLA service
        print("Loading VLA model...")
        self.vla_service = VLAService(self.vla_config)
        await self.vla_service.initialize()
        print(f"✅ Model loaded: {self.vla_config.model_name}")

        # Initialize robot connection
        if not self.sim_mode:
            print("Connecting to robot...")
            self.robot = await self._connect_robot()
            print("✅ Robot connected")
        else:
            print("ℹ️  Running in simulation mode")

        print("\nReady for instructions!")
        print("Type 'help' for available commands, 'quit' to exit.\n")

    async def _connect_robot(self):
        """Connect to robot hardware."""
        if self.robot_connector:
            return await self.robot_connector()

        # Default connection logic
        from so101_real import SO101RealDemo

        demo = SO101RealDemo(self.vla_config)
        await demo.initialize()
        return demo

    async def run(self) -> None:
        """Main conversation loop."""
        while True:
            try:
                # Get user input
                user_input = input("\n📝 You: ").strip()

                if not user_input:
                    continue

                # Process command
                if await self._handle_command(user_input):
                    break

            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"❌ Error: {e}")

    async def _handle_command(self, command: str) -> bool:
        """Handle user command.

        Returns:
            True if should exit, False otherwise.
        """
        command_lower = command.lower()

        # Exit commands
        if command_lower in ["quit", "exit", "q", "bye"]:
            print("👋 Goodbye!")
            return True

        # Help
        if command_lower == "help":
            self._print_help()
            return False

        # Status
        if command_lower in ["status", "info"]:
            self._print_status()
            return False

        # Templates
        if command_lower.startswith("template"):
            self._show_templates()
            return False

        # History
        if command_lower == "history":
            self._print_history()
            return False

        # Clear history
        if command_lower == "clear":
            self.conversation_history.clear()
            print("🗑️  Conversation history cleared")
            return False

        # Save session
        if command_lower.startswith("save"):
            parts = command.split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "session.json"
            self._save_session(filename)
            return False

        # Load session
        if command_lower.startswith("load"):
            parts = command.split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "session.json"
            self._load_session(filename)
            return False

        # Simulate (toggle sim mode)
        if command_lower == "simulate":
            self.sim_mode = True
            print("ℹ️  Switched to simulation mode")
            return False

        # Real robot
        if command_lower == "real":
            if self._confirm_safety():
                self.sim_mode = False
                print("⚠️  Switched to real robot mode")
            return False

        # Execute task
        await self._execute_task(command)
        return False

    def _confirm_safety(self) -> bool:
        """Confirm safety for real robot operation."""
        print("\n⚠️  SAFETY WARNING ⚠️")
        print("You are about to control a REAL ROBOT.")
        print("Ensure:")
        print("  - Emergency stop is accessible")
        print("  - Workspace is clear of obstacles")
        print("  - All safety systems are active")
        print()

        confirm = input("Type 'CONFIRM' to proceed: ")
        return confirm == "CONFIRM"

    async def _execute_task(self, instruction: str) -> None:
        """Execute a task instruction."""
        # Check for safety keywords
        if any(kw in instruction.lower() for kw in self.SAFETY_KEYWORDS):
            if not self.sim_mode:
                print("⚠️  Instruction contains safety-sensitive keywords")
                if not self._confirm_safety():
                    print("❌ Task cancelled")
                    return

        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": instruction,
        })

        print(f"\n🤖 Processing: '{instruction}'")

        # Get observation
        obs = await self._get_observation()

        # Run VLA inference
        try:
            vla_output = await self.vla_service.predict(
                image=obs["image"],
                instruction=instruction,
                proprioception=obs.get("proprioception"),
            )

            print(f"✅ Generated {len(vla_output.actions)} actions")
            print(f"   Latency: {vla_output.latency_ms:.1f}ms")
            print(f"   Confidence: {vla_output.confidence:.2f}")

            # Preview actions
            print(f"\n   First action: {vla_output.actions[0][:4]}")

            # Execute or simulate
            if self.sim_mode:
                print("\n[Simulation mode - actions previewed only]")
                print("   To execute, switch to 'real' mode")
            else:
                execute = input("\nExecute? [y/N]: ").lower() == "y"
                if execute:
                    await self._execute_actions(vla_output.actions)
                    print("✅ Task executed")
                else:
                    print("❌ Execution cancelled")

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": f"Generated {len(vla_output.actions)} actions",
                "actions": vla_output.actions.tolist(),
                "latency_ms": vla_output.latency_ms,
                "confidence": vla_output.confidence,
            })

        except Exception as e:
            logger.error(f"VLA inference failed: {e}")
            print(f"❌ VLA inference failed: {e}")

    async def _get_observation(self) -> dict:
        """Get current observation."""
        if self.sim_mode:
            # Generate synthetic observation
            return {
                "image": Image.new("RGB", (224, 224), color="gray"),
                "proprioception": np.zeros(7),
            }

        # Get from robot
        if self.robot:
            return self.robot._get_observation()

        raise RuntimeError("No robot connected")

    async def _execute_actions(self, actions: np.ndarray) -> None:
        """Execute actions on robot."""
        if self.robot and hasattr(self.robot, '_apply_action'):
            for action in actions[:10]:  # Execute first 10 actions
                self.robot._apply_action(action)
                await asyncio.sleep(0.1)

    def _print_help(self) -> None:
        """Print help message."""
        print("""
Available Commands:
  help              Show this help message
  status            Show system status
  templates         Show task templates
  history           Show conversation history
  clear             Clear conversation history
  save [file]       Save session to file
  load [file]       Load session from file
  simulate          Switch to simulation mode
  real              Switch to real robot mode
  quit/exit         Exit the interface

Task Examples:
  "Pick up the red cube"
  "Place the blue block on the table"
  "Push the green object to the left"
  "Stack the red cube on the blue cube"
""")

    def _print_status(self) -> None:
        """Print system status."""
        stats = self.vla_service.get_stats() if self.vla_service else {}

        print("\n" + "=" * 50)
        print("System Status")
        print("=" * 50)
        print(f"Mode: {'Simulation' if self.sim_mode else 'Real Robot'}")
        print(f"Model: {stats.get('model_name', 'N/A')}")
        print(f"Device: {stats.get('device', 'N/A')}")
        print(f"Initialized: {stats.get('initialized', False)}")
        print(f"Inferences: {stats.get('inference_count', 0)}")
        print(f"History entries: {len(self.conversation_history)}")
        print("=" * 50)

    def _show_templates(self) -> None:
        """Show task templates."""
        print("\nTask Templates:")
        for task_type, templates in self.TASK_TEMPLATES.items():
            print(f"\n{task_type.upper()}:")
            for template in templates:
                print(f"  - {template}")

    def _print_history(self) -> None:
        """Print conversation history."""
        if not self.conversation_history:
            print("No conversation history")
            return

        print("\nConversation History:")
        print("-" * 50)
        for entry in self.conversation_history:
            role = "📝 You" if entry["role"] == "user" else "🤖 Bot"
            print(f"{role}: {entry.get('content', '')}")
        print("-" * 50)

    def _save_session(self, filename: str) -> None:
        """Save conversation session."""
        data = {
            "config": {
                "model_name": self.vla_config.model_name,
                "device": self.vla_config.device,
                "sim_mode": self.sim_mode,
            },
            "history": self.conversation_history,
        }

        path = Path(filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"💾 Session saved to {path}")

    def _load_session(self, filename: str) -> None:
        """Load conversation session."""
        path = Path(filename)
        if not path.exists():
            print(f"❌ File not found: {path}")
            return

        with open(path) as f:
            data = json.load(f)

        self.conversation_history = data.get("history", [])
        print(f"📂 Session loaded from {path}")
        print(f"   {len(self.conversation_history)} entries restored")

    async def shutdown(self) -> None:
        """Shutdown the interface."""
        if self.vla_service:
            await self.vla_service.shutdown()


class WebInterface:
    """Web-based conversation interface (optional)."""

    def __init__(self, vla_config: VLAConfig | None = None, port: int = 8080):
        """Initialize web interface.

        Args:
            vla_config: VLA service configuration.
            port: Web server port.
        """
        self.vla_config = vla_config or VLAConfig()
        self.port = port
        self.vla_service: VLAService | None = None

    async def initialize(self) -> None:
        """Initialize web server."""
        try:
            from aiohttp import web
        except ImportError:
            raise RuntimeError("aiohttp required for web interface")

        self.vla_service = VLAService(self.vla_config)
        await self.vla_service.initialize()

        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_post("/api/predict", self._handle_predict)
        app.router.add_get("/api/status", self._handle_status)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", self.port)
        await site.start()

        print(f"🌐 Web interface running at http://localhost:{self.port}")

    async def _handle_index(self, request):
        """Serve main page."""
        from aiohttp import web

        html = """
<!DOCTYPE html>
<html>
<head>
    <title>ROSClaw VLA</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        #chat { border: 1px solid #ccc; height: 400px; overflow-y: auto; padding: 10px; }
        #input { width: 100%; padding: 10px; margin-top: 10px; }
        .user { color: blue; }
        .bot { color: green; }
    </style>
</head>
<body>
    <h1>🤖 ROSClaw VLA Interface</h1>
    <div id="chat"></div>
    <input type="text" id="input" placeholder="Enter instruction..." />
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');

        input.addEventListener('keypress', async (e) => {
            if (e.key === 'Enter' && input.value) {
                const msg = input.value;
                chat.innerHTML += `<p class="user">📝 You: ${msg}</p>`;
                input.value = '';

                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({instruction: msg})
                });
                const data = await response.json();
                chat.innerHTML += `<p class="bot">🤖 Bot: ${data.response}</p>`;
                chat.scrollTop = chat.scrollHeight;
            }
        });
    </script>
</body>
</html>
"""
        return web.Response(text=html, content_type="text/html")

    async def _handle_predict(self, request):
        """Handle prediction request."""
        from aiohttp import web

        data = await request.json()
        instruction = data.get("instruction", "")

        # Run VLA inference
        # TODO: Get actual image
        dummy_image = Image.new("RGB", (224, 224))

        vla_output = await self.vla_service.predict(
            image=dummy_image,
            instruction=instruction,
        )

        return web.json_response({
            "response": f"Generated {len(vla_output.actions)} actions",
            "latency_ms": vla_output.latency_ms,
            "confidence": vla_output.confidence,
        })

    async def _handle_status(self, request):
        """Handle status request."""
        from aiohttp import web

        stats = self.vla_service.get_stats() if self.vla_service else {}
        return web.json_response(stats)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="VLA Conversation Interface")
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
        "--sim",
        action="store_true",
        help="Start in simulation mode",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Use web interface",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Web interface port",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = VLAConfig(
        model_name=args.model,
        device=args.device,
    )

    if args.web:
        interface = WebInterface(config, port=args.port)
    else:
        interface = ConversationInterface(config, sim_mode=args.sim)

    try:
        await interface.initialize()
        if hasattr(interface, 'run'):
            await interface.run()
        else:
            # Keep web server running
            while True:
                await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await interface.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
