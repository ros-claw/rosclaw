#!/usr/bin/env python3
"""Generate RealSense skill package skeletons for ROSClaw."""
from __future__ import annotations

import os
from pathlib import Path
from datetime import UTC, datetime

ROOT = Path(__file__).parent.resolve()
SKILLS_DIR = ROOT / "skills"

SKILLS = [
    {
        "name": "realsense_capture_frame",
        "display_name": "RealSense Capture Frame",
        "description": "Capture a single color image frame from a RealSense camera.",
        "category": "perception",
        "robots": ["realsense-d405", "realsense-d435i", "realsense-dual"],
        "sensors": ["color_camera"],
    },
    {
        "name": "realsense_capture_rgbd",
        "display_name": "RealSense Capture RGBD",
        "description": "Capture aligned RGB-D frames from a RealSense camera.",
        "category": "perception",
        "robots": ["realsense-d405", "realsense-d435i", "realsense-dual"],
        "sensors": ["color_camera", "depth_camera"],
    },
    {
        "name": "realsense_depth_health_check",
        "display_name": "RealSense Depth Health Check",
        "description": "Verify RealSense depth stream health and basic statistics.",
        "category": "perception",
        "robots": ["realsense-d405", "realsense-d435i", "realsense-dual"],
        "sensors": ["depth_camera"],
    },
    {
        "name": "realsense_imu_check",
        "display_name": "RealSense IMU Check",
        "description": "Check RealSense D435i IMU stream and variance.",
        "category": "perception",
        "robots": ["realsense-d435i", "realsense-dual"],
        "sensors": ["imu"],
    },
    {
        "name": "scene_risk_scan",
        "display_name": "Scene Risk Scan",
        "description": "Scan a scene for risks using mid-range RGB-D and VLM.",
        "category": "perception",
        "robots": ["realsense-d435i", "realsense-dual"],
        "sensors": ["color_camera", "depth_camera"],
    },
    {
        "name": "obstacle_visibility_check",
        "display_name": "Obstacle Visibility Check",
        "description": "Verify that obstacles are visible in the depth stream.",
        "category": "perception",
        "robots": ["realsense-d435i", "realsense-dual"],
        "sensors": ["color_camera", "depth_camera"],
    },
    {
        "name": "realsense_camera_info_check",
        "display_name": "RealSense Camera Info Check",
        "description": "Validate camera_info topics and intrinsic parameters.",
        "category": "perception",
        "robots": ["realsense-d405", "realsense-d435i", "realsense-dual"],
        "sensors": ["color_camera"],
    },
    {
        "name": "realsense_pointcloud_check",
        "display_name": "RealSense PointCloud Check",
        "description": "Verify colored point cloud generation from RealSense.",
        "category": "perception",
        "robots": ["realsense-d405", "realsense-d435i", "realsense-dual"],
        "sensors": ["color_camera", "depth_camera"],
    },
]


def render_skill_yaml(name: str, display_name: str, description: str, category: str, robots: list[str]) -> str:
    robots_tag = "\n    - ".join([""] + robots)
    return f'''schema_version: "rosclaw.skill.v1"
kind: "Skill"

metadata:
  name: "{name}"
  display_name: "{display_name}"
  namespace: "rosclaw-realsense"
  version: "0.1.0"
  stage: "draft"
  category: "{category}"
  tags:{robots_tag}
    - physical-ai
    - perception-only
  description: "{description}"
  license: "MIT"
  authors:
    - name: "ROSClaw Team"
      url: "https://github.com/ros-claw"

identity:
  skill_id: "rosclaw-realsense/{name}"
  package_name: "rosclaw-realsense/{name}"
  canonical_uri: "rosclaw://skills/rosclaw-realsense/{name}"
  git_repo: "https://github.com/rosclaw/realsense-skills"
  git_commit: null

task:
  intent: "{name}"
  natural_language:
    en: "Execute the {name} skill."
  input_contract:
    required:
      - robot_state
    optional: []
  output_contract:
    success_condition:
      - "task_completed == true"
    artifacts:
      - trace
      - runtime_events

execution:
  entrypoint:
    type: "behavior_tree"
    file: "behavior_tree.xml"
  runtime_adapter: "rosclaw.runtime.skill_executor"
  policy:
    type: "hybrid"
    config: "policies/policy.yaml"
  prompts:
    planner: "prompts/planner.md"
    executor: "prompts/executor.md"
    verifier: "prompts/verifier.md"
    recovery: "prompts/recovery.md"
    safety: "prompts/safety.md"

compatibility:
  eurdf: "e-urdf-compat.yaml"
  providers: "providers.yaml"
  safety: "safety.yaml"

evaluation:
  dojo: "dojo.yaml"
  darwin: "darwin_eval.yaml"
  tests: "tests/"

lineage:
  file: "lineage.yaml"

evidence:
  directory: "evidence/"
  latest_eval_report: null

status:
  promotion_state: "draft"
  last_eval_passed: false
  safe_to_run_on_real_robot: false
  recommended_runtime_mode: "sandbox_first"
'''


def render_eurdf_compat(robots: list[str], sensors: list[str]) -> str:
    robots_yaml = ""
    for robot in robots:
        sensor_list = "\n      - ".join([""] + sensors)
        robots_yaml += f'''  - robot: "{robot}"
    eurdf_profile: "rosclaw-realsense/{robot}@>=1.0.0"
    body_profile:
      min_dof: 0
    required_limbs: []
    required_sensors:{sensor_list}
    optional_sensors: []
    required_frames:
      - base_link
    action_interfaces:
      - sensor_msgs/Image
    physical_limits:
      min_battery_percent: 0
    environment_assumptions: {{}}
'''
    return f'''schema_version: "rosclaw.eurdf_compat.v1"

compatible_robots:
{robots_yaml}incompatible: []
'''


def render_providers() -> str:
    return '''schema_version: "rosclaw.providers.v1"

required_capabilities:
  vlm.object_grounding:
    primary: "cosmos-reason2-lan"
    fallback: []
    timeout_ms: 5000
  vlm.scene_understanding:
    primary: "cosmos-reason2-lan"
    fallback: []
    timeout_ms: 5000

providers:
  cosmos-reason2-lan:
    type: "vlm"
    endpoint: "http://192.168.1.105:8009/v1/chat/completions"
    health_check: "rosclaw provider health cosmos-reason2-lan"

routing_policy:
  default: "capability_first"
  prefer_local: true
  require_health_check: false
  fallback_on:
    - timeout
    - unavailable
'''


def render_safety() -> str:
    return '''schema_version: "rosclaw.safety.v1"

runtime_mode:
  default: "sandbox_first"
  allowed:
    - dry_run
    - replay
    - sandbox

hard_constraints:
  perception_only: true
  robot:
    min_battery_percent: 0
    require_estop_ready: false
  action:
    max_linear_velocity_mps: 0.0
  environment:
    disallow_unknown_obstacles: false

robot:
  min_battery_percent: 0
  require_estop_ready: false

action:
  max_linear_velocity_mps: 0.0

environment:
  disallow_unknown_obstacles: false

sandbox:
  required_checks:
    - perception_only_check
  block_on:
    - actuator_action_requested

failure_policy:
  on_sandbox_block: "abort_and_explain"
  on_low_confidence: "verify_or_abort"
  on_repeated_failure: "write_memory_and_stop"
  max_runtime_retries: 2
'''


def render_dojo(name: str, robots: list[str]) -> str:
    return f'''schema_version: "rosclaw.dojo.v1"

practice_sources:
  default_query:
    task: "{name}"
    robot: "{robots[0]}"
    min_success: 0
    include_failures: true
  storage:
    traces:
      - "~/.rosclaw/practice/traces"
    memory:
      - "~/.rosclaw/memory"

mining:
  candidate_prefix: "candidate"
  min_episodes: 5
  min_success_episodes: 1
  include_failure_recovery: true
  segmentation:
    method: "event_boundary_and_state_transition"
    event_markers:
      - task_start
      - success_verified
      - failure_event
  clustering:
    method: "effect_signature"
    features:
      - precondition
      - outcome_metric

replay:
  required: true
  sample_episodes: 3
  compare:
    - state_transition
    - outcome_metric

training:
  enabled: false
  output_dir: "policies/checkpoints"
'''


def render_darwin_eval(name: str, robots: list[str]) -> str:
    return f'''schema_version: "rosclaw.darwin_eval.v1"

suite:
  name: "{name}_eval"
  mode: "sandbox"
  robot: "{robots[0]}"
  tasks:
    - name: "basic_execution"
      episodes: 5

metrics:
  success_rate:
    required: true
    promote_threshold: 0.75
  no_fall_rate:
    required: true
    promote_threshold: 1.0
  sandbox_block_rate:
    required: true
    max_allowed: 0.15
  recovery_success_rate:
    required: true
    promote_threshold: 0.60

promotion_gates:
  candidate_to_validated:
    all_required_metrics_pass: true
    min_total_episodes: 5
    require_evidence_report: true
    require_lineage: true
    require_hash_lock: false

regression:
  compare_against: []
  fail_if_safety_regression: true
'''


def render_lineage(name: str) -> str:
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return f'''schema_version: "rosclaw.lineage.v1"

skill:
  name: "{name}"
  current_version: "0.1.0"
  current_stage: "draft"

origin:
  created_by: "rosclaw test generator"
  created_at: "{now}"
  source_practice_sessions: []

parents:
  skill_versions: []
  memory_records: []
  how_interventions: []

candidates: []

versions: []

rollbacks: []
'''


def render_behavior_tree(name: str, display_name: str) -> str:
    return f'''<root BTCPP_format="4">
  <BehaviorTree ID="{display_name}">
    <Sequence name="{name}_sequence">
      <CheckRobotReady name="check_robot_ready"/>
      <SandboxValidate name="precheck_safety"/>
      <ExecuteSkill name="execute_{name}"/>
      <VerifyOutcome name="verify_outcome"/>
      <Fallback name="recovery_if_failed">
        <MemoryRecall name="recall_prior_failure"/>
        <HowAdvise name="runtime_intervention"/>
        <RetrySkill name="retry_with_patch"/>
      </Fallback>
      <WriteMemory name="write_outcome"/>
    </Sequence>
  </BehaviorTree>
</root>
'''


def render_readme(name: str, display_name: str, description: str, robots: list[str], sensors: list[str]) -> str:
    robots_bullets = "\n".join(f"- {r}" for r in robots)
    sensors_bullets = "\n".join(f"- {s}" for s in sensors)
    return f'''# {display_name}

## What it does

{description}

## Supported robots

{robots_bullets}

## Required sensors

{sensors_bullets}

## Required providers

- See `providers.yaml`
- Primary VLM: `cosmos-reason2-lan`

## Safety constraints

- See `safety.yaml`
- Perception-only: no actuation
- Default runtime mode: `sandbox_first`

## How to run

```bash
rosclaw skill validate {name}
rosclaw skill install skills/{name}
```

## Evaluation evidence

See `evidence/reports/` for latest eval reports.

## Version history

### 0.1.0

- Initial draft for RealSense acceptance test.

## Known limitations

- Draft stage; validated only against mock/perception data.
'''


def render_changelog() -> str:
    return '''# Changelog

## 0.1.0

- Initial draft.
'''


def render_lock() -> str:
    return '''schema_version: "rosclaw.skill_lock.v1"
locked: false
'''


def generate() -> None:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SKILLS:
        name = spec["name"]
        display = spec["display_name"]
        desc = spec["description"]
        category = spec["category"]
        robots = spec["robots"]
        sensors = spec["sensors"]

        dest = SKILLS_DIR / name
        dest.mkdir(parents=True, exist_ok=True)

        (dest / "skill.yaml").write_text(render_skill_yaml(name, display, desc, category, robots))
        (dest / "e-urdf-compat.yaml").write_text(render_eurdf_compat(robots, sensors))
        (dest / "providers.yaml").write_text(render_providers())
        (dest / "safety.yaml").write_text(render_safety())
        (dest / "dojo.yaml").write_text(render_dojo(name, robots))
        (dest / "darwin_eval.yaml").write_text(render_darwin_eval(name, robots))
        (dest / "lineage.yaml").write_text(render_lineage(name))
        (dest / "behavior_tree.xml").write_text(render_behavior_tree(name, display))
        (dest / "README.md").write_text(render_readme(name, display, desc, robots, sensors))
        (dest / "CHANGELOG.md").write_text(render_changelog())
        (dest / ".rosclaw").mkdir(exist_ok=True)
        (dest / ".rosclaw" / "lock.yaml").write_text(render_lock())
        (dest / "tests").mkdir(exist_ok=True)
        (dest / "tests" / ".gitkeep").write_text("")
        (dest / "evidence").mkdir(exist_ok=True)
        (dest / "evidence" / ".gitkeep").write_text("")
        print(f"generated {dest}")


if __name__ == "__main__":
    generate()
