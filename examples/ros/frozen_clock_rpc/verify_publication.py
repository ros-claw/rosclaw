"""Mechanical file/archive integrity only; no Native import, ROS or test execution."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked(blob, record):
    require(type(record['bytes']) is int and len(blob) == record['bytes'], 'size')
    require(hashlib.sha256(blob).hexdigest() == record['sha256'], 'hash')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract-root', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    require(manifest['schema'] == 'ROSClaw.FrozenClockRPC.PublicationManifest.v1', 'schema')
    files = manifest['files']
    require(set(p.name for p in root.iterdir()) == set(files) | {'manifest.json'}, 'file set')
    blobs = {}
    for name, record in files.items():
        require(type(name) is str and Path(name).name == name and name not in ('.', '..'), 'basename')
        path = root / name
        require(path.is_file() and not path.is_symlink(), 'regular sibling')
        require(path.stat().st_size <= 131072, 'file cap')
        blobs[name] = path.read_bytes()
        checked(blobs[name], record)
    for name, digest in manifest['exact_native_source3_sha256'].items():
        require(name in blobs and hashlib.sha256(blobs[name]).hexdigest() == digest, 'Native pin')
    require(set(manifest['archives']) == {'minimal_fixture.tar', 'readable_source3.tar'}, 'archive set')
    planned = {}
    for archive, spec in manifest['archives'].items():
        expected = spec['members']
        want = set(manifest['exact_native_source3_sha256'])
        if archive == 'minimal_fixture.tar':
            want.remove('FROZEN_CLOCK_RPC_V1.md')
        require(set(expected) == want, 'archive allowlist')
        with tarfile.open(fileobj=io.BytesIO(blobs[archive]), mode='r:') as tf:
            members = tf.getmembers()
            require(len(members) == len(expected) and set(m.name for m in members) == want, 'member set/duplicates')
            total = 0
            contents = {}
            for member in members:
                require(member.type == tarfile.REGTYPE and not member.linkname and not member.pax_headers, 'regular USTAR member only')
                require(member.name in expected and Path(member.name).name == member.name, 'member path')
                require(type(member.size) is int and 0 <= member.size <= 65536, 'member cap')
                total += member.size
                require(total <= 98304, 'archive payload cap')
                handle = tf.extractfile(member)
                require(handle is not None, 'member stream')
                with handle:
                    blob = handle.read(65537)
                checked(blob, expected[member.name])
                require(blob == blobs[member.name], 'archive/readable byte equality')
                contents[member.name] = blob
            planned[archive.removesuffix('.tar')] = contents
    target = Path(args.extract_root)
    target.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name, contents in planned.items():
        folder = target / name
        folder.mkdir(mode=0o700)
        for member, blob in contents.items():
            with (folder / member).open('xb') as stream:
                stream.write(blob)
    print('Publication hashes and both bounded archives validated; tests and SDK not executed.')


if __name__ == '__main__':
    main()
