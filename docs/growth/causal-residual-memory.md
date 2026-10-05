# Causal residual memory and bounded sequence imitation

This is a task-neutral numerical research component, not a deployed motor
policy, PPO, online RL, or evidence of football improvement. No simulator,
hardware transport, model-file loader, execution authority, or promotion is
provided.

`CausalResidualMemory` owns read-only GRU parameters, uses the reset-after
equations of the installed `torch.nn.GRU`, and exposes a strictly sequential
episode-relative index. Each state update consumes only the current context
and previous hidden state. Hidden state and outputs remain bounded by one;
the caller still owes physical gates, action caps, rate/torque limits, an
appropriate reset boundary, and independent physical validation. Creating
a new episode allocates independent zero state. A zero output head is exactly
zero regardless of hidden history. State inspection returns a copy.

`fit_recurrent_residual_imitation` fits complete, ordered context sequences.
The numerical prediction is frozen baseline plus at most `residual_cap` times
the supplied gate times the unit-bounded recurrent output. Teacher targets
enter only the offline supervised loss, not the GRU state. Input features
are supplied by the caller: a causal architecture does **not** certify that
those features contain no future information or that the rows describe
authenticated physical episodes.

The fit uses float64 Torch Adam, whole-episode mini-batches, causal unrolling,
gradient clipping, deterministic configuration, and an explicit CPU or CUDA
device. Torch is imported only for fitting. CPU RNG, the selected CUDA RNG,
thread count, and deterministic settings are restored. CUDA requires a
deterministic cuBLAS workspace before fitting. Signed-integer bounds are
checked after conversion to float64. Positive weights that underflow during
normalization are rejected rather than silently removed.

All original rows, including zero-weight rows, are checked for shape and
finite bounded values. Zero-weight episodes do not affect gradients and do
not teach avoidance. Episode state starts at zero for both fitting and full
training-set loss measurements. Mini-batch loss need not be monotonic. The
receipt binds the fitted parameters, training/memory source files, actual
update count, source episode/horizon counts, initial/final complete weighted
loss, and non-authorizing evidence flags. It does not authenticate a source
dataset, prove generalization, preserve old physical trajectories, or certify
a runtime controller.

Initial local validation: 13 memory tests and 13 fitter tests; Growth suite
466 passed. NumPy state/output matches the installed Torch GRUCell within
1e-14 on a finite test sequence, not as a universal bit-identity claim.
The synthetic temporal-recall fit reduces loss and restores RNG/threads.
No G1 recurrent policy has been trained or physically qualified at this stage.
