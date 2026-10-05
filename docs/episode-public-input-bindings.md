# Bind public instructions to admitted runtime inputs

A ROS2 action trial had a current case and nonce in its schema, but an old
namespace in its public prompt. Startup qualification missed the endpoint and
the native client followed the old instructions. Keep that trial's failure.

`validate_episode_prompt_bindings` is an opt-in, read-only guard for a single
`ROSCLAW_EPISODE_INPUT_JSON=` line. Call it before the clock, credentials or
worker, with an independently admitted spec projection containing every public
runtime operand: case, nonce, namespace, action/topic/service name, domain,
command, and relevant goals or resource references. Exact typed comparison
rejects stale values, missing or extra fields, duplicate keys and nonfinite
numbers. Do not obtain expected values from the prompt itself.

Generate endpoint-bearing prose from the same spec projection or refer readers
only to the structured inputs. This guard does not validate arbitrary prose,
grant access, enforce tool counts, or turn a prepared trial into a runtime pass.

The PI capability probe also checks the public session extension-runner getter
and actual `ExtensionRunner.emitToolCall` method. An observational `subscribe`
method alone does not prove an execution-blocking tool hook exists. The probe
reports API availability; it does not install a tool policy automatically.
