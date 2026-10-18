# Independent consumer outputs and a distinct TF timestamp condition

The closed SDK adds two source-only configuration capture roles:
`rsk_tf_future_stamp` (actual AMCL transform_tolerance readback 30 seconds) and
`rsk_global_sensor_frame_unresolvable` (actual global obstacle-layer sensor_frame
readback). The first is a timestamp-window experiment, distinct from disabling
TF broadcast and distinct from merely observing mixed node clock domains. A
configured number alone is not native fault acceptance: actual TF stamps and
consumer failures must be inspected before any claim.

An optional `costmap_consumer_capture` role saves received original OccupancyGrid
messages in the owned SDK container. Its separate process has subscriptions only;
it creates no publisher, Runtime, driver, action, Session or Permit. Topics must
come from independently pinned requirements and its exact configuration bytes
must match the explicit operator SHA. Default SDK behavior is unchanged.

Each full original grid is saved exclusively with original header/receive times,
frame, dimensions, planar origin and data. A SHA-bound index records fixed frame
queries. Unknown frames are not transformed or guessed, unknown cells retain -1,
rotated origins and grid boundaries are handled explicitly. Capture is bounded by
65536 cells per grid, 16 query points, 1000 records and 20 MB total. Unexpected
process exit or budget exhaustion stops the owned capture; records never confer
L1, Body or execution authority.

Physical experiments must independently pin an actual source-world obstacle and
its relationship to the declared map frame, preregister query cells before any
run, preserve original full grids and compare true consumer output in independent
Before/After/Rollback Worlds. A grid publication or a successful navigation goal
is not proof of obstacle consumption. Operator-assigned condition labels never
enter the hypothesis engine. None of these source changes itself constitutes
new physical acceptance or model efficacy evidence.
