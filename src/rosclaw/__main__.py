#!/usr/bin/env python3
"""ROSClaw CLI entry point for `python -m rosclaw`."""

import sys

from rosclaw.cli import main

if __name__ == "__main__":
    sys.exit(main())
