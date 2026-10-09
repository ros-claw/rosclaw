
## Native launcher DDS port preflight

The future Native entry now evaluates its registered domain against the actual Linux host ephemeral range before source/merge checks or any process launch. It persists the bounded host preflight only after all admission checks pass. It does not assert container network readiness or diagnose the original B100828 timeout. A conflicting registered domain is refused before any command/process/output-directory creation. Source tests now register domain81; the frozen N02 evaluation protocol remains unchanged. Focused99 passed and full ROS1350 passed/10 integration deselected/one existing warning in29.00s. Actual Native transport/physics remains NOT_RUN.
