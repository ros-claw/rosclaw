# Preserve explicit zero retries during onboarding

ROSClaw previously wrote `retry.maxRetries=1` during startup even when private
PI settings explicitly requested zero. PI 1.0.3 accepts zero, so this increased
the configured retry budget before the first model request. Observing successful
requests without retries does not prove that retries were disabled.

Onboarding now preserves integer `maxRetries=0` and `maxRetries=1` without
rewriting the settings file. Missing, malformed, negative or larger values
retain the existing default of one. Booleans and floating-point values are not
accepted as integer retry settings. The `enabled`, provider-specific settings
and other user keys are preserved.

This controls PI AgentSession retries. Provider SDK retries, normal tool
continuations, compaction requests and supervisor episode restarts are separate
mechanisms. A private `retry.enabled=false` remains a useful explicit disable
even with an older ROSClaw version that normalizes `maxRetries` to one.

Existing frozen episodes and global user settings are not rewritten by this
change. New onboarding calls honor explicit zero.
