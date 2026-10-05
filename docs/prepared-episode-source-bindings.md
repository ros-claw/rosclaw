Call `validate_episode_source_bindings` with primary source paths and the complete
protected input map taken from independently pinned preparation. Do not derive
either expectation from the candidate's manifest. This prevents a copied gate
from requiring an old application's files and prevents a hash-consistent partial
input map from passing as a complete closure.

The helper verifies all declared source hashes, required source coverage, exact
protected input coverage and paths confined to the workspace. It writes nothing
and grants no runtime or physical authority. Existing launchers must explicitly
call it; correctness, real artifact registration and changes after admission
still require separate checks.
