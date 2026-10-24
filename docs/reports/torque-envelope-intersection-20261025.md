# Simulated torque projection: simultaneous constraints

The G1 neural torque projector previously clipped amplitude, parent deviation,
and previous-command delta sequentially. A later clip could undo an earlier
constraint. Power scaling towards zero could also leave the allowed box.

The pure numeric `control.vector_envelope.intersect_vector_bounds` helper now
intersects the boxes before clipping. Empty intersections explicitly select the
existing parent fallback. After power scaling, a command outside the intersection
also selects the parent fallback. Previous torque is bounded by the actual
parent execution cap (0.85), not the potentially smaller neural candidate cap.

Fallback semantics are unchanged: a parent fallback is not a certified learned
command and need not satisfy candidate-only slew, power, or deviation bounds.
Receipts classify the two new reasons as projection fallbacks. This fix provides
no hardware authority or physical-robot safety certification.

Three regression cases reproduce against the frozen old implementation:
disjoint parent/rate boxes, power scaling outside a nonzero box, and rewriting
actual previous torque to the smaller candidate amplitude cap. Existing model
rollouts using the defective projector must not be promoted based on claimed
simultaneous constraints; re-evaluate with pinned fixed source.
