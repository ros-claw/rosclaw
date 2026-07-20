#!/usr/bin/env python3
"""Export canonical product status as a provenance-bearing JSON snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosclaw.product.status import load_product_status, product_status_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = product_status_path()
    status = load_product_status(source)
    snapshot = {
        "generated": {
            "schema_version": "rosclaw.product_status_snapshot.v1",
            "source": "src/rosclaw/product/status.yaml",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "status": status,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
