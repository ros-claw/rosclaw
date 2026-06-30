# Obstacle Visibility Check

## What it does

Verify that obstacles are visible in the depth stream.

## Supported robots

- realsense-d435i
- realsense-dual

## Required sensors

- color_camera
- depth_camera

## Required providers

- See `providers.yaml`
- Primary VLM: `cosmos-reason2-lan`

## Safety constraints

- See `safety.yaml`
- Perception-only: no actuation
- Default runtime mode: `sandbox_first`

## How to run

```bash
rosclaw skill validate obstacle_visibility_check
rosclaw skill install skills/obstacle_visibility_check
```

## Evaluation evidence

See `evidence/reports/` for latest eval reports.

## Version history

### 0.1.0

- Initial draft for RealSense acceptance test.

## Known limitations

- Draft stage; validated only against mock/perception data.
