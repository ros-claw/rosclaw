# Robot Integrations

Robot Integration is the public name for the signed hardware onboarding unit
above individual e-URDF, Hardware MCP, skill, policy, and verification assets.
The internal schema remains `rosclaw.robot_pack.v1`. An Integration does not
replace those assets or create another execution framework; it locks their
relationship so an operator and `rosclawd` agree on the same body, device,
capability, adapter revision, policy, and evidence rules.

## Command Lifecycle

```bash
# Read-only enumeration; no stream is opened.
rosclaw robot discover --type camera --json

# Install the signed Integration metadata and verify every payload hash.
rosclaw robot install ros-claw/realsense-d400

# Optionally install the native adapter at the Pack's exact git commit.
rosclaw robot install ros-claw/realsense-d400 --install-adapter

# Bind one discovered physical identity to a persistent Body.
rosclaw robot configure realsense-d400 --instance lab-d405 --serial SERIAL

# H1 software/contract verification; no camera claim.
rosclaw robot verify realsense-d400 --stage contract

# Hardware-read acceptance requires a canonical rosclawd receipt.
rosclaw robot verify lab-d405 --stage read-only --receipt receipt.json
```

`robot install` is the primary Integration lifecycle. `robot add` is retained
as a compatibility alias and performs the same signed, transactional install.
Legacy e-URDF profile operations remain under the other `robot`/`eurdf`
commands; neither command silently installs native Adapter dependencies.

## RealSense D400 Integration

The first built-in Integration supports D405 and D435i on Linux x86_64 and aarch64.
It is perception-only and exposes one capability:

```text
camera.capture_rgbd
```

The Pack recognizes the upstream librealsense USB product IDs `0b5b` (D405)
and `0b3a` (D435i). The source of truth is librealsense
[`d400-private.h`](https://github.com/IntelRealSense/librealsense/blob/master/src/ds/d400/d400-private.h).

The Hardware MCP dependency is locked to:

```text
https://github.com/ros-claw/librealsense-mcp
fdea4c3cfd03e7acf1adb664a9ffca5733d44b59
```

`--install-adapter` may install native dependencies such as `pyrealsense2` and
OpenCV. It is explicit because platform support and system packages vary. The
generic equivalent is:

```bash
rosclaw mcp install librealsense-mcp \
  --from-git https://github.com/ros-claw/librealsense-mcp \
  --revision fdea4c3cfd03e7acf1adb664a9ffca5733d44b59
```

Git adapter upgrades are staged beside the active checkout. Clone, exact
revision checkout, commit comparison, and manifest validation must all pass
before an atomic source swap; a failed upgrade preserves the prior source,
runtime config, and registry record.

An already installed adapter with no commit provenance or a different commit
is `version_mismatch`; read-only verification fails closed.

Configuration and daemon startup do not trust the registry value alone. For a
production Git install they recheck the source URL, requested revision, actual
`HEAD`, tracked-file diff, and unknown untracked files. Interpreter/install
outputs such as `__pycache__`, `.pytest_cache`, and `*.egg-info` are ignored;
an added source file or changed tracked file makes the binding
`version_mismatch`.

## Discovery Semantics

The `pyrealsense2` backend reports model, serial, firmware, USB speed, physical
port, stable URI, and available stream profiles without starting a pipeline.
When the SDK is unavailable, Linux sysfs can still report VID/PID, model,
serial, USB bus location, and speed. That fallback is marked partial because
it cannot prove firmware or stream-profile readiness.

Zero discovered devices is a valid discovery result. It is not H3 evidence.
Offline configuration is available only with explicit operator input:

```bash
rosclaw robot configure realsense-d400 \
  --instance reserved-d405 \
  --model D405 \
  --serial EXPECTED_SERIAL \
  --allow-offline
```

The generated config records `offline_configured: true`; it cannot pass the
live identity, stream, adapter, or receipt checks.

## Trust and Installation

The Pack verifier performs all of the following before installation:

1. Strict Pydantic schema validation with unknown fields rejected.
2. Path containment and symlink rejection.
3. SHA-256 validation for every declared payload.
4. Rejection of untracked files and component/checksum disagreement.
5. A domain-separated detached Ed25519 signature against a scoped trusted key.
6. A second verification after the transactional copy.
7. An atomic lock update under a cross-process file lock.

The current key is version-scoped and pinned in the ROSClaw distribution. This
protects Pack content against local replacement after distribution; it does not
replace package/release provenance such as a signed Git tag, PyPI provenance,
or Sigstore attestation.

Persistent paths are:

```text
$ROSCLAW_HOME/robots/packs/<namespace>/<name>/<version>/
$ROSCLAW_HOME/robots/robot-packs.lock.json
$ROSCLAW_HOME/robots/instances/<instance>.yaml
$ROSCLAW_HOME/robots/evidence/<evidence-id>.json
```

## Daemon Boundary

At startup, `rosclawd --robot-id <instance>` loads the instance only if the
Integration signature, installed digest, Body snapshot, safety policy, and configured
identity all match. It registers only declared daemon-side executors. For the
RealSense Pack that is `camera.capture_rgbd:REAL`; no actuator executor exists.

The executor enforces:

- exact Body and configured serial;
- daemon-authored capability authorization and the immutable Body snapshot;
- the exact configured MCP server rather than another matching server name;
- no serial substitution in Agent arguments;
- a fresh, non-symlink artifact directory under
  `$ROSCLAW_HOME/artifacts/robot-packs`;
- both non-empty color and depth artifacts exist;
- capture-completion time within the action window, positive dimensions, and
  RGB-D alignment;
- SHA-256 hashes are placed in the physical observation;
- `PHYSICALLY_OBSERVED` is emitted only after those checks pass.

REAL requests still need the normal daemon permit, Body snapshot, capability
scope, and authorization. The Agent-facing MCP surface does not expose raw
`pyrealsense2`, arbitrary MCP invocation, or device paths.

## Support Tiers

| Tier | Meaning |
|---|---|
| `H0_INDEXED` | Metadata only. |
| `H1_CONTRACT_VERIFIED` | Signed Pack, schemas, dependencies, policy, Body profiles, and fail-closed loader pass. |
| `H2_SIMULATION_VERIFIED` | A physics-backed path passes; not applicable to this perception-only camera Pack. |
| `H3_HARDWARE_READ_VERIFIED` | Live identity, stream profiles, real artifacts, hashes, receipt, and independent physical observation pass. |
| `H4_HARDWARE_ACTUATION_VERIFIED` | Independent real actuation acceptance passes; not applicable to this Pack. |
| `H5_AGENT_BLACKBOX_VERIFIED` | An external Agent completes the guarded path without direct driver access. |
| `H6_REFERENCE_SUPPORTED` | A maintained reference release has passed H5 and its published support contract. |

A local read-only run can produce an `H3_HARDWARE_READ_VERIFIED` candidate.
It never edits canonical product status and never self-declares independent
observation. Until an independent hardware report is reviewed, the bundled
RealSense Integration remains H1 in `src/rosclaw/product/status.yaml`.

Candidate receipt validation requires the canonical REAL action/body contract,
an exclusive body resource lease, ordered timestamped control transitions,
driver ACK and verification decisions, and color/depth hashes and modification
times consistent with the receipt execution window.

## Current Limits

- No independent D405/D435i hardware evidence is committed yet.
- No RealSense H5 external-Agent run is committed yet.
- The current receipt records a host-side capture-completion timestamp; it does
  not yet carry a device-native frame timestamp or hardware clock correlation.
- Execution receipts are canonical Runtime records but are not yet signed by a
  machine identity; H3 therefore still requires independent evidence review.
- Native device ACL isolation must still be validated on the hardware host.
- The daemon permit/action ledger is durable and authenticated across restart,
  but automatic compaction, TPM-backed keys, and a remote witness are not implemented.
- The current Integration installer supports trusted built-in/local Integration
  directories; Hub publication and remote Robot Integration resolution remain open.
- RH56 has a real LeRobot/SerialModbus path and developer-observed hardware
  evidence, but it is not yet packaged as a production daemon-side signed Robot
  Integration Worker. Do not advertise an RH56 install command until that migration passes.
