# Final acceptance environment

Run date: 2026-08-06/07 UTC.

| Item | Observed value |
|---|---|
| Host | Ubuntu/Linux 6.17.0-1021-nvidia, aarch64 |
| Python | 3.13.12 (tests also ran in GitHub Actions on 3.11 and 3.12) |
| Docker | 29.2.1 |
| SeekDB image | `quay.io/oceanbase/seekdb:latest` |
| SeekDB server | `5.7.25-OceanBase seekdb-v1.3.0.0` |
| Acceptance database | `rosclaw_know_acceptance`, isolated from Memory and Practice |
| Know baseline | `ac4c2dd`, final merge `3e15dfb` |
| How baseline | `d52ecfd`, final merge `9b866d4` |
| Core baseline | `0e0a6bd8` |

The live suite used only read-only public-source operations. Third-party
repository code was never imported, installed, or executed.

Hardware constraints: no Unitree G1, LIMO, or RealSense device was available.
No physical task-success claim is made.

The supplied vLLM endpoint initially answered a smoke request, but the final
paired run first remained queued and later failed TCP connection. The paired
model-effect result is therefore recorded as unavailable, not passed.
