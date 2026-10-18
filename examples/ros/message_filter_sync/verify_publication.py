"""Bounded exact archive verifier/extractor; never imports Native or runs tests."""

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked(blob, record):
    require(type(record["bytes"]) is int and len(blob) == record["bytes"], "size")
    require(hashlib.sha256(blob).hexdigest() == record["sha256"], "hash")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract-root", required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_bytes())
    require(manifest["schema"] == "ROSClaw.MF02.PublicationManifest.v1", "schema")
    require(
        {p.name for p in root.iterdir()} == set(manifest["files"]) | {"manifest.json"},
        "exact public file set",
    )
    blobs = {}
    for name, record in manifest["files"].items():
        require(
            type(name) is str and Path(name).name == name and name not in (".", ".."),
            "sibling name",
        )
        path = root / name
        require(
            path.is_file() and not path.is_symlink() and path.stat().st_size <= 131072,
            "bounded regular sibling",
        )
        blobs[name] = path.read_bytes()
        checked(blobs[name], record)
    sources = manifest["exact_native_source3"]
    require(
        set(sources)
        == {
            "mf02_dds_sync_scenario.py",
            "test_mf02_dds_sync_scenario.py",
            "MF02_SYNC_PARTIAL_REPAIR.md",
        },
        "source3",
    )
    for _name, record in sources.items():
        checked(blobs[record["readable_file"]], record)
    require(
        set(manifest["archives"]) == {"exact_native_source3.tar", "minimal_fixture.tar"},
        "archive set",
    )
    plans = {}
    for name, spec in manifest["archives"].items():
        expected = spec["members"]
        want = (
            set(sources)
            if name == "exact_native_source3.tar"
            else set(sources) - {"MF02_SYNC_PARTIAL_REPAIR.md"}
        )
        require(set(expected) == want, "archive allowlist")
        with tarfile.open(fileobj=io.BytesIO(blobs[name]), mode="r:") as tf:
            members = tf.getmembers()
            require(
                len(members) == len(want) and {m.name for m in members} == want,
                "members/duplicates",
            )
            total = 0
            contents = {}
            for m in members:
                require(
                    m.type == tarfile.REGTYPE and not m.linkname and not m.pax_headers,
                    "regular USTAR only",
                )
                require(m.name in want and Path(m.name).name == m.name, "no traversal")
                require(
                    m.mode == 0o400 and m.uid == m.gid == m.mtime == 0,
                    "canonical read-only archive metadata",
                )
                require(type(m.size) is int and 0 < m.size <= 65536, "member cap")
                total += m.size
                require(total <= 98304, "total cap")
                handle = tf.extractfile(m)
                require(handle is not None, "member stream")
                with handle:
                    blob = handle.read(65537)
                checked(blob, expected[m.name])
                require(blob == blobs[sources[m.name]["readable_file"]], "source/archive exact")
                contents[m.name] = blob
            plans[name.removesuffix(".tar")] = contents
    target = Path(args.extract_root)
    require(
        target.is_absolute() and not target.exists() and target.parent.is_dir(),
        "fresh absolute target",
    )
    require(not any(p.is_symlink() for p in [target, *target.parents]), "no symlink target")
    target.mkdir(mode=0o700)
    for name, contents in plans.items():
        folder = target / name
        folder.mkdir(mode=0o700)
        for name, blob in contents.items():
            fd = os.open(folder / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
            with os.fdopen(fd, "wb") as stream:
                stream.write(blob)
                stream.flush()
                os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": "MF02_PUBLIC_ARCHIVES_EXACT_AND_SAFELY_EXTRACTED_NOT_TESTED",
                "archives": 2,
                "SDK_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
