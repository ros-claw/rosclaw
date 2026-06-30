# Scene Risk Scan

## What it does

Scan a scene for risks using mid-range RGB-D and VLM.

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
rosclaw skill validate scene_risk_scan
rosclaw skill install skills/scene_risk_scan
```

## Evaluation evidence

See `evidence/reports/` for latest eval reports.

## Version history

### 0.1.0

- Initial draft for RealSense acceptance test.

## Known limitations

- Draft stage; validated only against mock/perception data.
