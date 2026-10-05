# Zero-anchored offline effect prediction

`predict_zero_anchored_effect` evaluates the same sealed prediction-only MLP
at the actual features and at a reference with explicitly declared intervention
columns set to zero in original units. It subtracts the two outputs. Thus a
zero intervention produces exactly zero predicted effect, without retraining
or changing the model's normalization or weights.

This is domain-neutral: robot identity, actuator units, simulator, targets and
paired physical provenance remain the downstream adapter's responsibility.
Use it only for predictions of branch-minus-baseline effects with a meaningful
zero-intervention reference. An algebraic constraint does not establish
accuracy, causal validity, confidence, calibration or improved policy behavior.
Both underlying calls retain the source-bound SIM_ONLY prediction validation;
no policy, optimizer, runtime, hardware or promotion interface is added.
