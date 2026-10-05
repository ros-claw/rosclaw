# Prepared schema identity admission

SDK session creation may replace a provisional UUID while a delivery schema
still contains the earlier constant. Every file can have its correct hash and
the final artifact can nevertheless be impossible to register.

Before a clock, credentials, or worker, call
`validate_episode_schema_identity(schema, {"UUID": spec["new_UUID"], ...})`
using independently admitted spec values. The helper requires every supplied
identity to be a required, exact string constant. It reads without mutation,
grants no execution authority, and complements full JSON Schema validation.

The helper is opt-in. Existing runtime entry points do not change implicitly.
