# Frozen state–prediction memory

`AnchorOutputMemory` is a task-neutral numerical training primitive. It imports
no runtime, robot transport, actuator, or simulator and grants no execution or
promotion authority.

## Why store outputs, not only zero a residual?

An initial residual gate can preserve the first frozen policy. After learning,
however, zeroing *all* accumulated plastic heads at newly successful states
would revert those states to the first policy and erase the new skill.

This memory records the **actual later-parent predictions**, paired with frozen
observations. At recorded observations its output is exactly that recorded
vector, even if a later learner proposes a different vector. Away from those
states it blends the proposal with a nearest recorded prediction using the
existing frozen-state distance gate.

## Contract

- Finite bounded arrays: 1–32,768 observations, 1–512 input dimensions and
  1–512 output dimensions; absolute numeric magnitude at most 1e6.
- Frozen encoder, parent policy, and source evidence content hashes are bound
  into serialization. They are identities, **not signatures or proof of trusted
  physical provenance**; callers must verify the evidence independently.
- The observation must include all causal context affecting the prediction.
  Identical observations with conflicting outputs are rejected, including
  signed-zero aliases. This can expose an incomplete/non-Markov observation
  contract, rather than silently choosing one teacher.
- Inputs and outputs are owned, immutable copies. Extension returns a new
  memory and preserves predecessor hash, all old observations, and outputs.
  Capacity overflow fails; no old states are silently discarded.
- Optional exact SciPy nearest-neighbor acceleration agrees with the NumPy
  reference path, including deterministic lowest-index ties.
- Nonfinite, misaligned, oversized, or wrong-encoder proposals are rejected.
  Even at a memorized state a malformed proposal is not silently accepted.
- Round-trip schema comparison rejects resealed changes to authority flags.
  `training_only=true`, `promotion_authorized=false`, and
  `hardware_authorized=false` remain fixed.

## What this does not establish

There is no unseen-state, distributional-retention, or physical-safety guarantee.
Nearest-reference selection can switch discontinuously between anchors, so
this is **not** a claim of smooth movement. Applications must apply their
independent output limits, temporal boundaries, and physical retention exams.

This is not unlimited lifelong memory. Exhausted capacity requires a separately
declared consolidation/compression experiment and full retention validation;
this API does not perform one automatically. A stored vector is a prediction,
not an approved action, Permit, driver, or execution receipt.
