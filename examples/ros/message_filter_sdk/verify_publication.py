"""Verify published bytes and safely extract one finite pure fixture; no SDK import."""
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
    require(manifest['schema'] == 'ROSClaw.MessageFilterSDK.PublicationManifest.v1', 'schema')
    allowed = {'message_filter_scenario.py', 'test_message_filter_scenario.py',
               'MESSAGE_FILTER_SDK_V1.md', 'inputs/INPUT_CASES.json', 'minimal_fixture.tar',
               'README.md', 'runtime_verification_20261017.json', 'verify_publication.py'}
    files = manifest['files']
    require(set(files) == allowed, 'file allowlist')
    require(not (root / 'inputs').is_symlink(), 'input parent')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    require(actual == allowed | {'manifest.json'}, 'file set')
    blobs = {}
    for name, record in files.items():
        path = root / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1, 'regular sibling')
        require(path.stat().st_size <= 131072, 'file cap')
        blobs[name] = path.read_bytes()
        checked(blobs[name], record)
    require(set(manifest['exact_native_source3_sha256']) ==
            {'message_filter_scenario.py', 'test_message_filter_scenario.py', 'MESSAGE_FILTER_SDK_V1.md'}, 'Native set')
    for name, digest in manifest['exact_native_source3_sha256'].items():
        require(hashlib.sha256(blobs[name]).hexdigest() == digest, 'Native pin')
    expected = manifest['minimal_fixture_members']
    want = {'message_filter_scenario.py', 'test_message_filter_scenario.py', 'inputs/INPUT_CASES.json'}
    require(set(expected) == want, 'archive allowlist')
    with tarfile.open(fileobj=io.BytesIO(blobs['minimal_fixture.tar']), mode='r:') as tf:
        members = tf.getmembers()
        require(len(members) == 3 and {m.name for m in members} == want, 'member set/duplicates')
        contents = {}
        total = 0
        for member in members:
            require(member.type == tarfile.REGTYPE and not member.linkname and not member.pax_headers, 'regular USTAR only')
            require(member.name in want, 'member path')
            require(type(member.size) is int and 0 <= member.size <= 65536, 'member cap')
            total += member.size
            require(total <= 98304, 'archive payload cap')
            handle = tf.extractfile(member)
            require(handle is not None, 'member stream')
            with handle:
                blob = handle.read(65537)
            checked(blob, expected[member.name])
            require(blob == blobs[member.name], 'readable equality')
            contents[member.name] = blob
    target = Path(args.extract_root).absolute()
    require(not target.exists() and not target.is_symlink(), 'fresh extraction root')
    for parent in target.parents:
        require(not parent.is_symlink(), 'extraction parent')
    target.mkdir(mode=0o700, parents=False, exist_ok=False)
    folder = target / 'minimal_fixture'
    folder.mkdir(mode=0o700)
    (folder / 'inputs').mkdir(mode=0o700)
    for member, blob in contents.items():
        with (folder / member).open('xb') as stream:
            stream.write(blob)
    print('Published hashes and exact three-member fixture verified; SDK and tests not executed.')


if __name__ == '__main__':
    main()
