# Capability Apps

A ROSClaw App is a small, versioned task recipe. It declares Capability calls,
data flow, timeouts, and verification conditions. It is not an Agent framework,
driver, Skill replacement, permission grant, or alternate Runtime.

## Lifecycle

Install and inspect a bundled App:

```bash
rosclaw app install ros-claw/realsense-inspect
rosclaw app list
rosclaw app validate realsense-inspect --json
```

Run it through a configured daemon Body:

```bash
rosclaw app run realsense-inspect \
  --body lab-d405 \
  --snapshot sha256:BODY_SNAPSHOT \
  --mode SHADOW \
  --json
```

`REAL` additionally requires a capability-bound daemon Permit for every
physical step, an armed current daemon generation, and a registered verified
executor. A human operator issues each Permit with
`rosclaw daemon permit-issue`; the Agent cannot call that operation as itself.
Use `--permit CAPABILITY=PERMIT_ID`; never place permits in a manifest.

## Low-code Authoring

```bash
rosclaw app init my-inspection --path .
rosclaw app add camera.capture_rgbd \
  --app my-inspection/app.yaml \
  --save-as frame
rosclaw app validate my-inspection/app.yaml --json
```

An App may depend only on named Capabilities. Schema validation rejects
southbound details such as `/dev/ttyUSB0`, register numbers, `/cmd_vel`, and
system device paths. Runtime execution creates one UID-bound Agent Session,
submits each step as an `ActionEnvelope`, renews its bounded lease while
waiting, evaluates declared verification conditions, and closes the Session.

## Bundled Apps

| App | Capabilities | Current boundary |
|---|---|---|
| `ros-claw/realsense-inspect` | `camera.capture_rgbd`, `vlm.risk_assessment` | Manifest/runtime component-tested; no independently verified camera run. |
| `ros-claw/rh56-rps` | `rh56.single_step` | Manifest/runtime component-tested; production RH56 daemon Worker/Robot Integration migration and Agent hardware acceptance remain pending. |

The App store currently resolves bundled IDs and local paths. A remote App Hub
resolver is not implemented. `ros-claw/...` is accepted only for a known
bundled App; arbitrary path-like registry identifiers fail closed.

## Evidence Semantics

App success means its declared steps returned terminal receipts and its local
verification expressions passed. It does not upgrade canonical support status.
Hardware and Agent claims still require the independent evidence defined in
`src/rosclaw/product/status.yaml`.
