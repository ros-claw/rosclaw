"""Opt-in source preparation checks; these are not physical qualification."""
from __future__ import annotations

import hashlib
import json

import pytest

from rosclaw.task_kernel.verifier import verdict_for


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def packet(tmp_path):
    (tmp_path / 'driver.py').write_text('if __name__ == "__main__":\n    print("preflight")\n')
    (tmp_path / 'helper.py').write_text('print("helper only")\n')
    (tmp_path / 'READY.md').write_text('Candidate preparation; no physics claim\n')
    data = {
        'schema_version': 'rosclaw.source_packet.v1',
        'scope': 'source_preflight_only',
        'closure_kind': 'declared_files',
        'files': {name: {'sha256': _sha(tmp_path / name),
                         'size_bytes': (tmp_path / name).stat().st_size}
                  for name in ('driver.py', 'helper.py', 'READY.md')},
        'canonical_entry': {'path': 'driver.py', 'sha256': _sha(tmp_path / 'driver.py')},
        'preflight_argv': ['python3', 'driver.py'],
    }
    return tmp_path, data


def _acceptance(workspace, data, argv=None):
    path = workspace / 'packet.json'
    path.write_text(json.dumps(data))
    return {'source_packet': {'path': 'packet.json', 'sha256': _sha(path)},
            'run': {'argv': argv or data['preflight_argv'], 'timeout_sec': 3}}


def _verdict(workspace, acceptance):
    return verdict_for(artifacts=[], acceptance=acceptance, workspace=workspace,
                       summary='Preparation result')


def test_valid_packet_preflight_passes(packet):
    root, data = packet
    assert _verdict(root, _acceptance(root, data))['status'] == 'PASS'


def test_stale_nested_entry_rejected_before_execution(packet):
    root, data = packet
    data['canonical_entry']['sha256'] = '0' * 64
    assert _verdict(root, _acceptance(root, data))['status'] == 'REPAIR_REQUIRED'


def test_ready_modified_after_freeze_rejected(packet):
    root, data = packet
    acceptance = _acceptance(root, data)
    (root / 'READY.md').write_text('Edited after freeze')
    assert _verdict(root, acceptance)['status'] == 'REPAIR_REQUIRED'


def test_helper_cannot_replace_canonical_cli(packet):
    root, data = packet
    assert _verdict(root, _acceptance(root, data, ['python3', 'helper.py']))['status'] == 'REPAIR_REQUIRED'


def test_actual_cli_child_failure_is_not_helper_success(packet):
    root, data = packet
    driver = root / 'driver.py'
    driver.write_text('def helper(): return True\nif __name__ == "__main__": MissingOldClass()\n')
    data['files']['driver.py'] = {'sha256': _sha(driver), 'size_bytes': driver.stat().st_size}
    data['canonical_entry']['sha256'] = _sha(driver)
    result = _verdict(root, _acceptance(root, data))
    assert result['status'] == 'REPAIR_REQUIRED'
    assert any('rc=1' in f for f in result['failures'])


@pytest.mark.parametrize('mutation', ['unknown_schema', 'unknown_field', 'missing_scope',
                                      'bad_hash', 'wrong_size', 'no_entry_file',
                                      'traversal', 'absolute', 'symlink', 'unknown_closure'])
def test_invalid_closure_failclosed(packet, mutation):
    root, data = packet
    if mutation == 'unknown_schema':
        data['schema_version'] = 'rosclaw.source_packet.v99'
    elif mutation == 'unknown_field':
        data['physical_success'] = True
    elif mutation == 'missing_scope':
        del data['scope']
    elif mutation == 'bad_hash':
        data['files']['READY.md']['sha256'] = 'invalid'
    elif mutation == 'wrong_size':
        data['files']['READY.md']['size_bytes'] += 1
    elif mutation == 'no_entry_file':
        del data['files']['driver.py']
    elif mutation == 'traversal':
        data['files']['../outside'] = data['files']['READY.md']
    elif mutation == 'absolute':
        data['files'][str(root / 'READY.md')] = data['files']['READY.md']
    elif mutation == 'symlink':
        (root / 'READY.md').unlink()
        (root / 'READY.md').symlink_to(root / 'helper.py')
    elif mutation == 'unknown_closure':
        data['closure_kind'] = 'ambient_imports'
    assert _verdict(root, _acceptance(root, data))['status'] == 'REPAIR_REQUIRED'


@pytest.mark.parametrize('value', [None, {}, {'path': 'missing.json', 'sha256': '0' * 64},
                                  {'path': '../packet.json', 'sha256': '0' * 64}])
def test_missing_or_invalid_packet_ref_rejected(packet, value):
    root, _ = packet
    assert _verdict(root, {'source_packet': value})['status'] == 'REPAIR_REQUIRED'


def test_source_mutation_during_cli_cannot_pass(packet):
    root, data = packet
    driver = root / 'driver.py'
    driver.write_text('from pathlib import Path\nPath("READY.md").write_text("changed by CLI")\n')
    data['files']['driver.py'] = {'sha256': _sha(driver), 'size_bytes': driver.stat().st_size}
    data['canonical_entry']['sha256'] = _sha(driver)
    assert _verdict(root, _acceptance(root, data))['status'] == 'REPAIR_REQUIRED'


def test_packet_requires_actual_canonical_run(packet):
    root, data = packet
    acceptance = _acceptance(root, data)
    del acceptance['run']
    assert _verdict(root, acceptance)['status'] == 'REPAIR_REQUIRED'


def test_opaque_json_optout_is_unchanged(packet):
    root, data = packet
    data['canonical_entry']['sha256'] = '0' * 64
    _acceptance(root, data)
    artifact = {'path': str(root / 'packet.json'), 'sha256': _sha(root / 'packet.json')}
    result = verdict_for(artifacts=[artifact], acceptance={}, workspace=root, summary='Opaque data')
    assert result['status'] == 'PASS'


@pytest.mark.parametrize('mutation', ['packet_tamper', 'bad_packet_json', 'huge_packet',
                                      'file_count', 'file_size', 'total_size',
                                      'entry_not_python', 'helper_pinned', 'timeout_nan',
                                      'timeout_negative', 'timeout_large', 'symlink_parent'])
def test_bounds_and_pinned_packet_failclosed(packet, mutation):
    root, data = packet
    acceptance = _acceptance(root, data)
    if mutation == 'packet_tamper':
        (root / 'packet.json').write_text('{}')
    elif mutation == 'bad_packet_json':
        (root / 'packet.json').write_text('{invalid')
        acceptance['source_packet']['sha256'] = _sha(root / 'packet.json')
    elif mutation == 'huge_packet':
        (root / 'packet.json').write_text(' ' * (1024 * 1024 + 1))
        acceptance['source_packet']['sha256'] = _sha(root / 'packet.json')
    elif mutation == 'file_count':
        data['files'].update({f'f{i}': data['files']['READY.md'] for i in range(256)})
        acceptance = _acceptance(root, data)
    elif mutation == 'file_size':
        data['files']['READY.md']['size_bytes'] = 512 * 1024 * 1024 + 1
        acceptance = _acceptance(root, data)
    elif mutation == 'total_size':
        for record in data['files'].values():
            record['size_bytes'] = 512 * 1024 * 1024
        acceptance = _acceptance(root, data)
    elif mutation == 'entry_not_python':
        data['preflight_argv'][0] = 'pytest'
        acceptance = _acceptance(root, data)
    elif mutation == 'helper_pinned':
        data['preflight_argv'][1] = 'helper.py'
        acceptance = _acceptance(root, data)
    elif mutation == 'timeout_nan':
        acceptance['run']['timeout_sec'] = float('nan')
    elif mutation == 'timeout_negative':
        acceptance['run']['timeout_sec'] = -1
    elif mutation == 'timeout_large':
        acceptance['run']['timeout_sec'] = 601
    elif mutation == 'symlink_parent':
        (root / 'linked').symlink_to(root, target_is_directory=True)
        data['files']['linked/READY.md'] = data['files']['READY.md']
        acceptance = _acceptance(root, data)
    assert _verdict(root, acceptance)['status'] == 'REPAIR_REQUIRED'


def test_invalid_packet_never_executes_child(packet):
    root, data = packet
    driver = root / 'driver.py'
    driver.write_text('from pathlib import Path\nPath("CHILD_EXECUTED").write_text("yes")\n')
    data['files']['driver.py'] = {'sha256': _sha(driver), 'size_bytes': driver.stat().st_size}
    data['canonical_entry']['sha256'] = '0' * 64
    assert _verdict(root, _acceptance(root, data))['status'] == 'REPAIR_REQUIRED'
    assert not (root / 'CHILD_EXECUTED').exists()


def test_actual_cli_timeout_rejected(packet):
    root, data = packet
    driver = root / 'driver.py'
    driver.write_text('import time\ntime.sleep(10)\n')
    data['files']['driver.py'] = {'sha256': _sha(driver), 'size_bytes': driver.stat().st_size}
    data['canonical_entry']['sha256'] = _sha(driver)
    acceptance = _acceptance(root, data)
    acceptance['run']['timeout_sec'] = 0.05
    result = _verdict(root, acceptance)
    assert result['status'] == 'REPAIR_REQUIRED'
    assert any('超时' in f for f in result['failures'])


@pytest.mark.parametrize('changed', [False, True])
def test_kernel_frozen_acceptance_enforces_packet(packet, changed):
    import sqlite3

    from rosclaw.storage.migrations import MigrationRunner
    from rosclaw.task_kernel.service import TaskKernel

    root, data = packet
    with sqlite3.connect(':memory:') as conn:
        conn.row_factory = sqlite3.Row
        MigrationRunner().apply(conn, 'sqlite')
        kernel = TaskKernel(conn, root)
        bound = kernel.bind_message(mission_id='m', session_ref='s', backend_native_id='n',
                                    message_id='prep', text='Validate source preparation',
                                    cwd=str(root), workspace_root=str(root))
        kernel.set_acceptance(bound['task_id'], _acceptance(root, data))
        if changed:
            (root / 'READY.md').write_text('stale')
        result = kernel.finish_task(task_id=bound['task_id'], summary='Preparation only', artifact_ids=[])
        assert result['status'] == ('REPAIR_REQUIRED' if changed else 'SUCCEEDED')
    conn.close()


def test_duplicate_packet_json_key_rejected(packet):
    root, data = packet
    acceptance = _acceptance(root, data)
    raw = (root / 'packet.json').read_text()
    raw = raw.replace('"scope": "source_preflight_only"',
                      '"scope": "unknown", "scope": "source_preflight_only"')
    (root / 'packet.json').write_text(raw)
    acceptance['source_packet']['sha256'] = _sha(root / 'packet.json')
    assert _verdict(root, acceptance)['status'] == 'REPAIR_REQUIRED'
