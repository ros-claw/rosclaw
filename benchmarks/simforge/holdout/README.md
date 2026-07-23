# Hidden holdout boundary

This directory intentionally contains no seeds, generated scenarios, or episode artifacts.
The holdout service receives a private seed-ledger secret at runtime and returns aggregate
metrics plus evidence commitments only. Raw holdout data must remain outside the source
checkout and inaccessible to candidate-generator processes.
