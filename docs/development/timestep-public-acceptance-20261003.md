# Public E05 acceptance

The post-reboot native Kimi E05 trial preserved two native RAW_EXACT replay
receipts and valid paired trajectories. Independent maximum common-time
position error improved from 5.3955 mm to 1.0791 mm, about 80%, but failed the
oracle's existing 0.5 mm requirement. The original prompt did not disclose
that requirement or the 30% improvement floor. Its FAIL record is retained;
no looser grading or retroactive promotion is applied.

The task now publishes the metric, units, both thresholds, recording density,
common-time requirements, duration and timestamp matching tolerance. A shared
`TIMESTEP_ACCEPTANCE` contract generates the quantitative prompt, its task
metadata, and the grading checks. The existing 0.5 mm and 30% thresholds are
unchanged. The existing prompt's three-second minimum is now checked against
actual elapsed trajectory time, not an absolute simulation timestamp.

Eleven meaningful saved-position counterexamples verify the published
contract: an isolated impact outlier cannot hide in a mean/RMSE; Euclidean
position distance is not a per-axis bound; improvement alone cannot satisfy
the absolute threshold; below/at/above boundary cases are checked; equal
trajectories at different recording times cannot interpolate their way into
evidence; agreeing sparse traces are rejected; and nonzero initial time does
not make a short trace three seconds long. These fixtures produce no new
physical steps. A fresh native Kimi trial must use the new source contract.
