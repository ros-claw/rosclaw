#!/usr/bin/env python3
"""Emit reproducible MuJoCo mobile deadman evidence as JSON."""

from __future__ import annotations

import json

from rosclaw.sandbox.mobile_deadman import run_mobile_deadman_scenario


def main() -> int:
    evidence = run_mobile_deadman_scenario()
    print(json.dumps(evidence.to_dict(), indent=2, sort_keys=True))
    return 0 if evidence.stopped else 1


if __name__ == "__main__":
    raise SystemExit(main())
