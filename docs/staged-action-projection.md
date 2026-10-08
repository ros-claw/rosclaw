# Staged offline action projection

`staged_action_projection` computes a bounded squashed action, applies the
change limit relative to the supplied previous action, then applies final
caller-supplied bounds. Order matters: moving final bounds can override the
earlier change limit. Replacing the ordered map with an intersection changes
the action law.

The helper owns its returned array, rejects non-finite or misaligned operands,
and assumes final bounds contain zero. It knows no robot, units, dimensions,
simulator, torque limits or permissions. It is offline mathematical data only,
not a safety validator or a motor command. Callers authenticate their actual
action law and bounds, and verify numerical agreement against recorded actions
before using it in any diagnostic or learning loss. Teacher-forced agreement
does not imply closed-loop policy improvement.
