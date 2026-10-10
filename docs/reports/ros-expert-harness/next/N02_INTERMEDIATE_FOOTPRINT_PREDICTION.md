# Intermediate repair footprint prediction candidate

Status: isolated implementation candidate, not selected for physical acceptance.
Parent73a686a; original Waffle109002 pair FAIL and Burger NOT_RUN remain frozen.

Closed Waffle109002 actually dispatched29 two-target requests. Independent GT
nearest-time phase extraction found28 intermediate targets never within25mm,
10 never within100mm, and11 nearest-point yaw errors above45 degrees. This
association does not establish a controller/localization root cause. One
selected two-target sequence was truncated by the original dispatch budget;
selection cardinality must not be reported as actual request cardinality.

Nav2 1.3.13 RemovePassedGoals removes only intermediate goals based on distance,
without checking their yaw, and retains the last goal:
https://github.com/ros-navigation/navigation2/blob/1.3.13/nav2_behavior_tree/plugins/action/remove_passed_goals_action.cpp

Implementation hypothesis: the explicitly selected tracking strategy should
not value an intermediate pose as though its requested yaw were attained.
Use a yaw-independent disk inscribed in the cleaning polygon, eroded by the
registered100mm map-frame pruning radius, only for intermediate reward. Keep
the existing final-pose oriented footprint and nine one-cell translation
scenarios. Reject sequences with no predicted intermediate gain. Legacy
strategies, BT bytes, final tolerance, budgets, velocity, Body, collision,
coverage denominator and independent measured verifier remain unchanged.

This is a conservative geometric prediction under its stated map-frame pose
assumption. It is not a calibrated localization bound, physical coverage
credit, a guarantee of actual arrival, or measured efficiency improvement.
Actual Nav2 endpoint offsets and GT/map separation remain unresolved. Do not
read formal holdout data to tune it. Any selected future physics requires a
new frozen source, preregistration and the established acceptance gates.

Candidate deduplication also retains distinct intermediate support when final
footprints match. A two-hole regression verifies that useful continuous
pairs remain selectable; the new model must not silently disable sequences.
The earlier af01 prototype and its31-snapshot single-target result are retained
separately. The initial positive-case fixture accidentally allowed one rotated
brush pose to cover both holes; that test failure is retained, and the revised
fixture separates the holes beyond every single-pose footprint diameter.
