# Recovery prompt

If a round failed because of camera drop, serial fault, or gesture misread,
back off to a safe open pose, wait for temperatures to settle, and retry the
current round. Never retry more than the configured max_runtime_retries.
