# RealSense Capture Frame

## What it does

Capture a single color image frame from a RealSense camera.

## Supported robots

- realsense-d405
- realsense-d435i
- realsense-dual

## Required sensors

- color_camera

## Required providers

- See `providers.yaml`
- Primary VLM: `cosmos-reason2-lan`

## Safety constraints

- See `safety.yaml`
- Perception-only: no actuation
- Default runtime mode: `sandbox_first`

## How to run

```bash
rosclaw skill validate realsense_capture_frame
rosclaw skill install skills/realsense_capture_frame
```

## Evaluation evidence

See `evidence/reports/` for latest eval reports.

## Version history

### 0.1.0

- Initial draft for RealSense acceptance test.

## Known limitations

- Draft stage; validated only against mock/perception data.
