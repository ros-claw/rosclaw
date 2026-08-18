#!/bin/bash
# DF-20 soak tiers (phase-II §29) — real machines only, never CI.
# Usage: soak.sh 1000 | 10000 | 7x24 [workdir]
set -u
TIER=${1:?usage: soak.sh 1000|10000|7x24 [workdir]}
WORK=${2:-/tmp/data_flywheel_soak_$TIER}
HERE="$(cd "$(dirname "$0")" && pwd)"
REPORT="$HERE/../reports/soak_${TIER}_$(date +%Y%m%dT%H%M%S).json"
mkdir -p "$(dirname "$REPORT")"

case "$TIER" in
  1000)   ARGS="--episodes 1000 --session-seconds 0.05" ;;
  10000)  ARGS="--episodes 10000 --session-seconds 0.05" ;;
  7x24)   ARGS="--episodes 100000000 --soak-duration-sec 604800 --session-seconds 0.1" ;;
  *) echo "unknown tier $TIER" >&2; exit 2 ;;
esac

echo "soak tier=$TIER workdir=$WORK report=$REPORT"
python3 "$HERE/run_live_acceptance.py" \
  $ARGS \
  --workdir "$WORK" \
  --report "$REPORT"
