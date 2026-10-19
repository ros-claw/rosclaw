# Explicit SDK contracts for read-only diagnostics

The retained V1 Native policy, Nav2 policy and Nav2 observation recipe keep the
original SDK image
`sha256:51ffb0e0aeb391f0efbef3013c7fc83586794c222b233e3dea688f3015ef1c6b`.
Their serialized fields and defaults remain unchanged. An unavailable image
fails; the operator never selects another tag or image automatically.

V2 is a separate operator-selected contract for the reconstructed SDK
`sha256:8cc3fbf1eaa57058f0c1386c061253a55651106bd46b98fb046439f966c9019b`:

| Input | Explicit schema |
| --- | --- |
| Native sensor-only policy | `rosclaw.rsk_diagnostic_probe_policy.v2` |
| Nav2 source-only policy | `rosclaw.rsk_nav2_diagnostic_probe_policy.v2` |
| Nav2 observation recipe | `rosclaw.rsk_nav2_observation_recipe.v2` |

Each V2 policy and recipe carries `sdk_image`, restricted to this exact digest.
Selecting V2 requires new original inputs, their hashes, the current frozen
source, and the existing entry gates. Changing historical V1 inputs does not
turn them into new evidence. A Nav2 policy/recipe image mismatch fails before
dispatch or consumption of the shared probe budget. Container creation,
owner-only cleanup and original-container review use the selected exact image.
The reviewer defaults to the old image when replaying V1 originals. A rehashed
container record with a different image still fails its selected contract.

The reconstructed image has a new identity despite matching the 1,590 historical
package versions, 33 Python wheel versions and sampled passive metadata overlay
bytes. Its portable archive and historical signed repository qualification are
separate evidence, rather than a claim that the old digest was reproduced.

V2 preserves the read-only operations, fixed role enumeration, no-device SDK
isolation and existing limits. Native diagnostics start no physics World. Nav2
diagnostics permit one owned source-only World with zero goals, commands or
model calls. No recipe accepts commands, scripts, arbitrary images or new
execution authority. Original diagnostic scope, source, run, snapshot, semantic
staging, complete asset inventories and wall-time checks remain required.

This compatibility change enables fresh component acceptance. It does not
grant Body/L1 admission, establish a live robot repair, satisfy pilot entry
attestations or demonstrate causal Memory benefit. The Core `src` tree is
unchanged.
