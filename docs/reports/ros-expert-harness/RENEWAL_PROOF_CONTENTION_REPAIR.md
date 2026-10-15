# Keep expensive proof production off the renewal critical section

Source 051972f2 independently passed its complete G3 14-World matrix. The next
G4 map-to-odom repaired World completed the position goal but its yaw action
latched the original SDK lease deadline: the next request arrived about 3.6 ms
late. This failed World is preserved. Retaining the real DDS client removed
repeated discovery; it did not remove serialized expensive proof production in
both the main execution loop and renewal thread.

The supervision interface now defaults check_renewal to the existing full check.
The isolated SDK override validates an immutable snapshot of the last durably
published AND audited proof, its unchanged original expiry, action identity,
private directory and current independently observed receiver nonterminal state.
It never waits on the proof producer's lock. If the old proof remains valid, is
older than 50 ms and the producer is idle, renewal may collect and durably publish
a fresh proof with the original 250 ms cap. It never refreshes an expired proof. Before
admission, dispatch and every main execution check, synchronous current-facts
collection and fresh HMAC proof production remain unchanged.

The receiver still independently checks current ROS/source/daemon facts, HMAC,
source cursor, original proof expiry and its old unexpired SDK lease for EVERY
actual renewal, command and watchdog check. A cached fact is never promoted to
live evidence. Reusing the exact authenticated proof inside its original at-most
250 ms lifetime is already supported by the receiver; expiry permanently rejects
renewal. No SDK/Action Lease/proof/source/RPC timeout or Body number changes.
An in-flight proof cannot be accepted as a durable publication after a producer
failure or close. Unknown receiver state is refused.

Software contracts explicitly hold the expensive producer lock while checking
renewal, forbid duplicate collection, assert unchanged proof bytes/deadline,
and reject expiry, revival and current receiver terminal state. Actual G4 repaired
Worlds plus final-source/main full matrices remain required before acceptance.

The first read-only renewal development World failed because main execution can
wait longer than a proof period for action events; that original failure is
preserved. Nonblocking opportunistic production while the previous proof is
still valid addresses that liveness gap without queued duplicate collection.
