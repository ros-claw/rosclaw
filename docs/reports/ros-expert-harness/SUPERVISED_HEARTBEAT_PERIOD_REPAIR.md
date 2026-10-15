# Supervised renewal cadence repair

The first actual-main #680 matrix stopped at the lease-loss World after ten
independently accepted rows. Its original sink latched ACTION_LEASE_EXPIRED
before the separate canonical 10,000 ms Action Lease expired. The failure is
preserved and is not accepted as the intended canonical-lease experiment.

The last accepted SDK renewal expired at monotonic 510772677669059 ns; the
next original renewal request arrived at 510772693088358 ns. Supervision, lock
waits and RPC work were followed by a fixed extra 50 ms sleep. The SDK correctly
refused the late renewal; all old requests remained refused after stable stop.

Guarded renewal now schedules the next attempt from the previous iteration's
start. Work consumes the existing 50 ms period; overruns do not accumulate
catch-up renewals. Every attempt retains the original live supervision, actual
ACK and 150 ms RPC checks. SDK lease/proof deadlines, canonical Action Lease
10,000/3,000 ms defaults, all Body bounds and 500 ms source freshness are
unchanged. Legacy unsupervised operation retains its previous 200 ms delay.

The first new development lease-loss World passed independent original review:
531 healthy source reviews, 244 independently replayed source samples, 9,417
raw contact messages, zero unexpected contacts, canonical ACTION_LEASE_EXPIRED
and actual old-action/old-lease refusal. First independent physical stop after
canonical expiry was 0.170 seconds, followed by a 0.350 second stable-stop
window. The canonical terminal receipt itself does not claim verified physical
stop; independent physics supplies that evidence. This is development evidence,
not the required final-source/main full matrix. The 55 navigation executor
software contracts also pass.

The five-case G4 15-World matrix on preceding source 5b8e50d0 is preserved and
sealed separately. Final-source G3 and G4 matrices and original source/main CI
must be checked again before either gate closes. No actual model result or REAL
hardware acceptance follows from this repair.
