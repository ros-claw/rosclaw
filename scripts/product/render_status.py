#!/usr/bin/env python3
"""Render or check README product-status blocks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rosclaw.product.readme import replace_readme_matrix  # noqa: E402
from rosclaw.product.status import load_product_status  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero instead of writing when a README is stale.",
    )
    args = parser.parse_args()

    status = load_product_status()
    stale: list[Path] = []
    for relative_path, language in (("README.md", "en"), ("README.zh.md", "zh")):
        path = REPOSITORY_ROOT / relative_path
        current = path.read_text(encoding="utf-8")
        rendered = replace_readme_matrix(current, status, language)
        if rendered == current:
            continue
        stale.append(path)
        if not args.check:
            path.write_text(rendered, encoding="utf-8")

    if args.check and stale:
        for path in stale:
            print(f"stale product status: {path.relative_to(REPOSITORY_ROOT)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
