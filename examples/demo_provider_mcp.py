#!/usr/bin/env python3
"""
ROSClaw Provider-MCP Integration Demo

Demonstrates the semantic capability layer:
- Runtime hosts ProviderRegistry + CapabilityRouter + GuardPipeline
- MCPHub exposes semantic tools (observe_scene, locate_object, delegate_skill, verify_task_success)
- Tool calls route through CapabilityRouter to the best matching provider
- Fallback chain automatically activates if primary provider fails
"""

import asyncio
from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.core.event_bus import EventBus
from rosclaw.agent_runtime.mcp_hub import MCPHub


async def demo_semantic_tools():
    """Demo 1: MCPHub with semantic capability tools."""
    print("=" * 60)
    print("Demo 1: Semantic Capability Tools (Provider-Aware Mode)")
    print("=" * 60)

    # Create Runtime with provider layer
    runtime = Runtime(RuntimeConfig(robot_id="ur5e_001", enable_provider=True))
    runtime.initialize()

    # Attach MCPHub to Runtime
    hub = MCPHub(runtime.event_bus, robot_id="ur5e_001", runtime=runtime)
    hub.initialize()

    print(f"\nRegistered tools: {[t['name'] for t in hub.tools]}")

    # --- observe_scene ---
    print("\n--- observe_scene ---")
    result = await hub.handle_tool_call("observe_scene", {
        "image_topic": "/camera/color/image_raw",
        "query": "What objects are on the table?",
    })
    print(f"Status: {result['status']}")
    print(f"Capability: {result['capability']}")
    print(f"Provider: {result['provider']}")
    print(f"Result: {result['result']}")

    # --- locate_object ---
    print("\n--- locate_object ---")
    result = await hub.handle_tool_call("locate_object", {
        "object_name": "red cup",
        "image_topic": "/camera/color/image_raw",
    })
    print(f"Status: {result['status']}")
    print(f"Capability: {result['capability']}")
    print(f"Provider: {result['provider']}")
    obj = result['result'].get('objects', [{}])[0]
    print(f"Detected: {obj.get('label')} @ bbox={obj.get('bbox_2d')} conf={obj.get('confidence')}")

    # --- delegate_skill (grasp) ---
    print("\n--- delegate_skill (grasp) ---")
    result = await hub.handle_tool_call("delegate_skill", {
        "skill": "grasp",
        "target": {"object": "red cup", "bbox": [120, 80, 230, 200]},
        "constraints": {"force": 0.6, "approach": "top_down"},
    })
    print(f"Status: {result['status']}")
    print(f"Capability: {result['capability']}")
    print(f"Provider: {result['provider']}")
    print(f"Result: {result['result']}")

    # --- delegate_skill (pick_and_place) ---
    print("\n--- delegate_skill (pick_and_place) ---")
    result = await hub.handle_tool_call("delegate_skill", {
        "skill": "pick_and_place",
        "target": {"source": "red cup", "destination": "blue bin"},
    })
    print(f"Status: {result['status']}")
    print(f"Result: {result['result']}")

    # --- verify_task_success ---
    print("\n--- verify_task_success ---")
    result = await hub.handle_tool_call("verify_task_success", {
        "task_description": "The red cup was placed into the blue bin",
        "image_topic": "/camera/color/image_raw",
    })
    print(f"Status: {result['status']}")
    print(f"Capability: {result['capability']}")
    print(f"Success: {result['result'].get('success')}")
    print(f"Confidence: {result['result'].get('confidence')}")
    print(f"Reason: {result['result'].get('reason')}")

    hub.stop()
    runtime.stop()


async def demo_low_level_fallback():
    """Demo 2: MCPHub without Runtime falls back to low-level tools."""
    print("\n" + "=" * 60)
    print("Demo 2: Low-Level Fallback Mode (No Provider Layer)")
    print("=" * 60)

    bus = EventBus()
    hub = MCPHub(bus, robot_id="ur5e_001")  # No runtime attached
    hub.initialize()

    print(f"\nRegistered tools: {[t['name'] for t in hub.tools]}")

    # These are the original low-level tools
    print("\n--- move_joints (fallback) ---")
    result = await hub.handle_tool_call("move_joints", {
        "joint_positions": [0.1, -0.5, 1.2, -0.8, 0.3, 0.0],
        "duration": 2.0,
    })
    print(f"Result: {result}")

    print("\n--- grasp (fallback) ---")
    result = await hub.handle_tool_call("grasp", {"action": "close", "force": 0.7})
    print(f"Result: {result}")

    hub.stop()


async def demo_provider_layer_directly():
    """Demo 3: Direct capability router usage (bypass MCPHub)."""
    print("\n" + "=" * 60)
    print("Demo 3: Direct Capability Router Usage")
    print("=" * 60)

    runtime = Runtime(RuntimeConfig(robot_id="ur5e_001", enable_provider=True))
    runtime.initialize()

    from rosclaw.provider.core.request import ProviderRequest

    request = ProviderRequest(
        request_id="demo_001",
        capability="vlm.object_grounding",
        inputs={"query": "screwdriver", "camera_topic": "/camera/color/image_raw"},
        context={"robot": "ur5e_001"},
        constraints={"latency_ms": 500, "safety_level": "STRICT"},
    )

    print(f"\nRouting request: {request.capability}")
    decision = await runtime.capability_router.route(request)
    print(f"Selected provider: {decision.selected_provider}")
    print(f"Score: {decision.score:.2f}")
    print(f"Reason: {decision.reason}")
    print(f"Fallbacks: {decision.fallbacks}")

    print("\nInvoking provider...")
    response = await runtime.capability_router.invoke(request)
    print(f"Provider: {response.provider}")
    print(f"Latency: {response.latency_ms}ms")
    print(f"Result: {response.result}")

    runtime.stop()


async def demo_registry_stats():
    """Demo 4: Provider registry health and statistics."""
    print("\n" + "=" * 60)
    print("Demo 4: Provider Registry Statistics")
    print("=" * 60)

    runtime = Runtime(RuntimeConfig(robot_id="ur5e_001", enable_provider=True))
    runtime.initialize()

    stats = runtime.provider_registry.get_statistics()
    print(f"\nTotal providers: {stats['total_providers']}")
    print(f"Healthy: {stats['healthy_providers']}")
    print(f"Unhealthy: {stats['unhealthy_providers']}")
    print(f"By type: {stats['by_type']}")

    print("\nRegistered providers:")
    for name in runtime.provider_registry.list_providers():
        manifest = runtime.provider_registry.get_manifest(name)
        healthy = runtime.provider_registry.is_healthy(name)
        print(f"  - {name} (type={manifest.type}, capabilities={manifest.capabilities}, healthy={healthy})")

    runtime.stop()


async def demo_capability_client():
    """Demo 5: High-level task orchestration via CapabilityClient."""
    print("\n" + "=" * 60)
    print("Demo 5: CapabilityClient - Composite Task Orchestration")
    print("=" * 60)

    from rosclaw.provider.client import CapabilityClient

    runtime = Runtime(RuntimeConfig(robot_id="ur5e_001", enable_provider=True))
    runtime.initialize()

    client = CapabilityClient(runtime.capability_router)

    # --- pick up task ---
    print("\n--- Task: pick up the red cup ---")
    result = await client.run_task(
        task="pick up the red cup",
        robot="ur5e_001",
        scene_input={"camera_topic": "/camera/color/image_raw"},
    )
    print(f"Task: {result.task}")
    print(f"Status: {result.status}")
    print(f"Steps:")
    for step in result.steps:
        print(f"  - {step['capability']}: {step['status']} (provider={step.get('provider', 'n/a')})")
    print(f"Trace ID: {result.trace['trace_id']}")
    print(f"Total latency: {result.trace['total_latency_ms']}ms")
    print(f"Final result: {result.final_result}")

    # --- inspect task ---
    print("\n--- Task: what do you see on the table? ---")
    result = await client.run_task(
        task="what do you see on the table?",
        robot="ur5e_001",
        scene_input={"camera_topic": "/camera/color/image_raw"},
    )
    print(f"Task: {result.task}")
    print(f"Status: {result.status}")
    print(f"Steps: {[s['capability'] + ':' + s['status'] for s in result.steps]}")

    runtime.stop()


async def demo_provider_loader():
    """Demo 6: ProviderLoader scans directory and loads from YAML."""
    print("\n" + "=" * 60)
    print("Demo 6: ProviderLoader - YAML Auto-Discovery")
    print("=" * 60)

    import tempfile
    from pathlib import Path
    from rosclaw.provider.loader import ProviderLoader
    from rosclaw.provider.core.registry import ProviderRegistry

    registry = ProviderRegistry()
    loader = ProviderLoader(registry)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a test provider.yaml
        provider_dir = Path(tmpdir) / "ollama_vlm"
        provider_dir.mkdir()
        (provider_dir / "provider.yaml").write_text("""
name: ollama_vlm
version: "0.1.0"
type: vlm
capabilities:
  - vlm.scene_understanding
  - vlm.object_grounding
modalities:
  input: [image, text]
  output: [object_list]
runtime:
  backend: http
  endpoint: http://localhost:11434/api/generate
  device: cpu
safety:
  executable: false
  requires_guard: false
""")

        loaded = loader.scan_directory(tmpdir)
        print(f"\nLoaded providers from {tmpdir}: {loaded}")

        for name in loaded:
            manifest = registry.get_manifest(name)
            print(f"  - {name}: type={manifest.type}, backend={manifest.runtime.backend}")
            print(f"    capabilities: {manifest.capabilities}")
            print(f"    endpoint: {manifest.runtime.endpoint}")

    print("\nProviderLoader unload test:")
    loader.unload("ollama_vlm")
    print(f"  Registry after unload: {registry.list_providers()}")


async def main():
    await demo_semantic_tools()
    await demo_low_level_fallback()
    await demo_provider_layer_directly()
    await demo_registry_stats()
    await demo_capability_client()
    await demo_provider_loader()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
