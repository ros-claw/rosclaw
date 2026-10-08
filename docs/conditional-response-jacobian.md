# State-conditioned local response learning

`conditional_response_jacobian` learns `effect = J(state) @ intervention`
from adapter-authenticated paired intervention-minus-baseline effects. The
three-layer neural network predicts a local Jacobian, not motor commands.
Training and inference have exactly zero effect for a zero intervention.
State normalization is centered; intervention/effect normalization uses
training-only RMS without subtracting their means.

A global least-squares Jacobian fitted only on training contexts initializes
the final layer and remains an explicit evaluation baseline. Context-disjoint
evaluation also reports a zero-effect baseline. Optional Torch is used only
when fitting; NumPy inference validates source/dependency seals, dimensions,
finite values and the SIM_ONLY prediction boundary. Training restores Torch
RNG and thread settings.

Physical units, state observations, intervention bounds, measured paired
effects and provenance remain downstream responsibilities. Contact dynamics
can be nonsmooth: local linearity is a hypothesis, not a verified simulator or
guarantee outside the measured neighborhood. An improved prediction metric
does not demonstrate improved motion, policy growth, online RL or authorization.
