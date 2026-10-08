# Optional executed-action objective for sequence imitation

`fit_recurrent_residual_imitation(..., action_projection=None)` retains its
original latent-only loss by default. An explicit offline projection dataset can
instead train the same causal GRU with normalized executed-action MSE plus a
positive latent auxiliary loss. The auxiliary term retains a gradient where
the recorded slew or final box saturates the executed output.

The projection dataset supplies aligned previous actions, final bounds, actual
executed targets, cap, slew, and auxiliary weight. Every original row, including
zero-weight failures, must reconstruct its recorded teacher action from the
original latent target under the ordered NumPy reference before fitting.
Torch uses the same order in the differentiable loss. Projection labels never
enter recurrent inputs or hidden state. Previous actions in this loss are
teacher-forced records, not a new closed-loop policy rollout.

The returned objective receipt binds all projection arrays and scalar options,
the reference source, order, original row count, and no-authority flags. The
caller must separately authenticate physical data and its own action law. Loss
improvement does not prove physical retention or task improvement, and no
physics, actuator command, online RL, critic, PPO, or promotion is added.
Torch remains optional outside fitting. RNG, determinism, and thread state are
restored; caller arrays remain owned by the caller.

Default and projected regression tests cover actual fitting, initial loss
agreement, ownership/RNG restoration, malformed bounds, non-finite labels,
teacher/action mismatch even on zero-weight rows, and unchanged default receipt
mode. Downstream candidates require fresh source snapshots and independent
full-task exams; never rewrite an old source-bound candidate to change its loss.
