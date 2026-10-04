"""Real private foreground processes must stop after manager recovery."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import sqlite3
import sys
from pathlib import Path

import pytest

from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.operation_manager import OperationManager


@pytest.mark.parametrize('field', ['sid', 'uid', 'boot_id', 'start_ticks'])
@pytest.mark.asyncio
async def test_same_session_member_proof_failure_is_unresolved(ledger, monkeypatch, field):
    from dataclasses import replace

    from rosclaw.task_kernel.operation_manager import OperationCancellationUnresolvedError
    from rosclaw.task_kernel.process_identity import ProcessIdentity

    conn, root = ledger
    manager = OperationManager(None, conn)
    marker = root / 'member.pid'
    child = ('import os,time; from pathlib import Path; '
             f'Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)')
    parent = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(60)'
    op = await manager.start(task_id='t', attempt_id='', kind='process',
                             argv=[sys.executable, '-c', parent], cwd=str(root))
    proc = manager._procs[op['operation_id']]
    try:
        for _ in range(200):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        pid = int(marker.read_text())
        capture = ProcessIdentity.capture
        original = capture(pid)
        leader = ProcessIdentity.parse(op['process_identity_json'])
        assert original is not None and leader is not None
        value = (leader.start_ticks - 1 if field == 'start_ticks' else
                 'wrong-boot' if field == 'boot_id' else getattr(original, field) + 1)

        def corrupted_capture(cls, candidate):
            result = capture(candidate)
            return replace(result, **{field: value}) if candidate == pid and result else result

        monkeypatch.setattr(ProcessIdentity, 'capture', classmethod(corrupted_capture))
        with pytest.raises(OperationCancellationUnresolvedError):
            await manager.cancel(op['operation_id'])
        assert manager._pid_alive(pid) and manager._pid_alive(proc.pid)
        assert manager.get(op['operation_id'])['state'] == 'CANCELING'
        assert not any(e['event_type'] == 'operation.process_stopped'
                       for e in manager.events_since('t', 0))
    finally:
        await _cleanup(proc, [manager])


@pytest.mark.parametrize('recovered', [False, True])
@pytest.mark.asyncio
async def test_gnu_timeout_different_pgid_same_owned_session_stops(ledger, monkeypatch, recovered):
    import rosclaw.task_kernel.operation_manager as module
    from rosclaw.task_kernel.process_identity import ProcessIdentity, signal_owned_members

    timeout = shutil.which('timeout')
    if timeout is None:
        pytest.skip('GNU timeout required')
    monkeypatch.setattr(module, '_CANCEL_GRACE_S', 0.1)
    conn, root = ledger
    first, second = OperationManager(None, conn), OperationManager(None, conn)
    marker = root / 'timeout-child.pid'
    child = ('import os,signal,time; from pathlib import Path; '
             'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
             f'Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)')
    op = await first.start(task_id='t', attempt_id='', kind='process',
                           argv=[timeout, '-k', '10s', '-s', 'TERM', '60s',
                                 sys.executable, '-c', child], cwd=str(root))
    proc = first._procs[op['operation_id']]
    unrelated = await asyncio.create_subprocess_exec(
        sys.executable, '-c', 'import time; time.sleep(60)', start_new_session=True,
    )
    unrelated_identity = ProcessIdentity.capture(unrelated.pid)
    captured = []
    try:
        for _ in range(200):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        assert marker.exists()
        child_identity = ProcessIdentity.capture(int(marker.read_text()))
        leader = ProcessIdentity.parse(op['process_identity_json'])
        assert child_identity is not None and leader is not None
        assert child_identity.sid == leader.sid and child_identity.pgid != leader.pgid
        # Keep exact private identities for fixture cleanup even against old code.
        for path in Path('/proc').iterdir():
            if path.name.isdecimal():
                identity = ProcessIdentity.capture(int(path.name))
                if identity is not None and identity.sid == leader.sid:
                    captured.append(identity)
        manager = second if recovered else first
        if recovered:
            driver = first._drivers[op['operation_id']]
            driver.cancel()
            await asyncio.gather(driver, return_exceptions=True)
            assert (await second.recover_on_boot())['reattached'] == 1
        await asyncio.wait_for(manager.cancel(op['operation_id']), 2)
        assert manager.get(op['operation_id'])['state'] == 'CANCELLED'
        assert not manager._pid_alive(child_identity.pid), 'CANCELLED left same-SID timeout child alive'
        assert all(not manager._pid_alive(identity.pid) for identity in captured)
        assert unrelated_identity is not None and unrelated_identity.matches()
    finally:
        signal_owned_members(captured, signal.SIGKILL)
        if unrelated_identity is not None:
            signal_owned_members([unrelated_identity], signal.SIGKILL)
        await asyncio.wait_for(unrelated.wait(), 5)
        await _cleanup(proc, [first, second])


@pytest.fixture
def ledger(tmp_path):
    conn = sqlite3.connect(tmp_path / 'private.db')
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, 'sqlite')
    conn.execute("INSERT INTO tasks (task_id,mission_id,root_goal,mode,state,active_revision,"
                 "workspace_path,created_at,updated_at) VALUES "
                 "('t','m','private lifecycle fixture','SIMULATION','RUNNING',1,?,'now','now')",
                 (str(tmp_path),))
    conn.commit()
    yield conn, tmp_path
    conn.close()


async def _cleanup(proc, managers):
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    await asyncio.wait_for(proc.wait(), 5)
    tasks = [task for mgr in managers for task in mgr._drivers.values()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_recovered_manager_cancel_actually_stops_process_group(ledger):
    conn, root = ledger
    first = OperationManager(None, conn)
    second = OperationManager(None, conn)
    op = await first.start(task_id='t', attempt_id='', kind='process',
                           argv=[sys.executable, '-c', 'import time; time.sleep(60)'], cwd=str(root))
    proc = first._procs[op['operation_id']]
    try:
        driver = first._drivers[op['operation_id']]
        driver.cancel()
        await asyncio.gather(driver, return_exceptions=True)
        await second.recover_on_boot()
        await second.cancel(op['operation_id'])
        assert not second._pid_alive(proc.pid), 'CANCELLED ledger did not stop recovered process'
        assert second.get(op['operation_id'])['state'] == 'CANCELLED'
        assert any(e['event_type'] == 'operation.process_stopped'
                   for e in second.events_since('t', 0))
    finally:
        await _cleanup(proc, [first, second])


@pytest.mark.parametrize('field', ['missing', 'start_ticks', 'boot_id', 'uid', 'pgid', 'sid',
                                  'cwd', 'command_sha256'])
@pytest.mark.asyncio
async def test_unverified_identity_never_signals_or_claims_stop(ledger, field):
    from rosclaw.task_kernel.operation_manager import OperationCancellationUnresolvedError

    conn, root = ledger
    first = OperationManager(None, conn)
    second = OperationManager(None, conn)
    op = await first.start(task_id='t', attempt_id='', kind='process',
                           argv=[sys.executable, '-c', 'import time; time.sleep(60)'], cwd=str(root))
    proc = first._procs[op['operation_id']]
    try:
        driver = first._drivers[op['operation_id']]
        driver.cancel()
        await asyncio.gather(driver, return_exceptions=True)
        data = json.loads(op['process_identity_json'])
        if field == 'missing':
            raw = ''
        else:
            data[field] = data[field] + 1 if type(data[field]) is int else 'wrong_identity'
            raw = json.dumps(data)
        conn.execute('UPDATE operations SET process_identity_json = ? WHERE operation_id = ?',
                     (raw, op['operation_id']))
        report = await second.recover_on_boot()
        assert report['reattached'] == 0 and report['unresolved'] == 1
        with pytest.raises(OperationCancellationUnresolvedError):
            await second.cancel(op['operation_id'])
        assert second._pid_alive(proc.pid)
        assert second.get(op['operation_id'])['state'] == 'CANCELING'
        types = [e['event_type'] for e in second.events_since('t', 0)]
        assert 'operation.cancel_unresolved' in types and 'operation.cancelled' not in types
    finally:
        await _cleanup(proc, [first, second])


@pytest.mark.asyncio
async def test_recovered_group_child_ignoring_term_is_stopped_and_zombie_not_alive(ledger, monkeypatch):
    import rosclaw.task_kernel.operation_manager as module

    monkeypatch.setattr(module, '_CANCEL_GRACE_S', 0.1)
    conn, root = ledger
    first = OperationManager(None, conn)
    second = OperationManager(None, conn)
    marker = root / 'child.json'
    child = ('import os,signal,time; from pathlib import Path; '
             'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
             f'Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)')
    parent = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(60)'
    op = await first.start(task_id='t', attempt_id='', kind='process',
                           argv=[sys.executable, '-c', parent], cwd=str(root))
    proc = first._procs[op['operation_id']]
    try:
        for _ in range(200):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        child_pid = int(marker.read_text())
        assert second._pid_alive(child_pid)
        driver = first._drivers[op['operation_id']]
        driver.cancel()
        await asyncio.gather(driver, return_exceptions=True)
        await second.recover_on_boot()
        await second.cancel(op['operation_id'])
        assert second.get(op['operation_id'])['state'] == 'CANCELLED'
        assert not second._pid_alive(proc.pid) and not second._pid_alive(child_pid)
    finally:
        await _cleanup(proc, [first, second])


@pytest.mark.asyncio
async def test_actual_owner_process_exit_new_connection_can_stop_owned_job(ledger):
    from rosclaw.task_kernel.process_identity import ProcessIdentity

    conn, root = ledger
    script = '''import asyncio,json,os,sqlite3,sys
from rosclaw.task_kernel.operation_manager import OperationManager
c=sqlite3.connect(sys.argv[1]);c.row_factory=sqlite3.Row
async def run():
 m=OperationManager(None,c)
 op=await m.start(task_id="t",attempt_id="",kind="process",argv=[sys.executable,"-c","import time;time.sleep(60)"],cwd=sys.argv[2])
 c.commit();print(json.dumps(op),flush=True);os._exit(0)
asyncio.run(run())
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / 'src'))
    owner = await asyncio.create_subprocess_exec(sys.executable, '-c', script,
                                                str(root / 'private.db'), str(root),
                                                env=env, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(owner.communicate(), 10)
    assert owner.returncode == 0, err.decode()
    op = json.loads(out)
    identity = ProcessIdentity.parse(op['process_identity_json'])
    second_conn = sqlite3.connect(root / 'private.db')
    second_conn.row_factory = sqlite3.Row
    manager = OperationManager(None, second_conn)
    try:
        assert identity is not None and identity.matches()
        assert (await manager.recover_on_boot())['reattached'] == 1
        await manager.cancel(op['operation_id'])
        assert manager.get(op['operation_id'])['state'] == 'CANCELLED'
        assert not manager._pid_alive(op['pid'])
    finally:
        if identity is not None and identity.matches():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(identity.pgid, signal.SIGKILL)
        for driver in manager._drivers.values():
            driver.cancel()
        await asyncio.gather(*manager._drivers.values(), return_exceptions=True)
        second_conn.close()


@pytest.mark.asyncio
async def test_numeric_pid_now_points_at_unrelated_owned_fixture_not_signaled(ledger):
    from rosclaw.task_kernel.operation_manager import OperationCancellationUnresolvedError

    conn, root = ledger
    first = OperationManager(None, conn)
    second = OperationManager(None, conn)
    one = await first.start(task_id='t', attempt_id='', kind='process',
                            argv=[sys.executable, '-c', 'import time;time.sleep(60)'], cwd=str(root))
    two = await first.start(task_id='t', attempt_id='', kind='process',
                            argv=[sys.executable, '-c', 'import time;time.sleep(60)'], cwd=str(root))
    procs = [first._procs[x['operation_id']] for x in (one, two)]
    try:
        for driver in first._drivers.values():
            driver.cancel()
        await asyncio.gather(*first._drivers.values(), return_exceptions=True)
        conn.execute('UPDATE operations SET pid = ? WHERE operation_id = ?',
                     (two['pid'], one['operation_id']))
        with pytest.raises(OperationCancellationUnresolvedError):
            await second.cancel(one['operation_id'])
        assert all(second._pid_alive(p.pid) for p in procs)
    finally:
        for proc in procs:
            await _cleanup(proc, [first, second])


@pytest.mark.asyncio
async def test_process_stop_and_cascade_report_typed_unresolved(ledger):
    from types import SimpleNamespace

    from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
    from tests.agentd.test_pi_tool_bridge import _request

    conn, root = ledger
    first = OperationManager(None, conn)
    second = OperationManager(None, conn)
    op = await first.start(task_id='t', attempt_id='', kind='process',
                           argv=[sys.executable, '-c', 'import time;time.sleep(60)'], cwd=str(root))
    proc = first._procs[op['operation_id']]
    try:
        driver = first._drivers[op['operation_id']]
        driver.cancel()
        await asyncio.gather(driver, return_exceptions=True)
        conn.execute("UPDATE operations SET process_identity_json = '' WHERE operation_id = ?",
                     (op['operation_id'],))
        service = SimpleNamespace(_store=SimpleNamespace(connection=conn), _operation_manager=second)
        result = await PiToolDispatcher(service)._process_stop(
            _request('rosclaw_process_stop', arguments={'operation_id': op['operation_id']}))
        assert result.ok is False and result.status == 'CANCELING'
        assert result.error_code == 'CANCEL_STOP_UNCONFIRMED'
        assert result.retryable and '停止未确认' in result.summary
        report = await second.cancel_many([op['operation_id']])
        assert report['ok'] is False and report['operations_cancelled'] == 0
        assert report['operations_unresolved'] == [op['operation_id']]
        assert second._pid_alive(proc.pid)
    finally:
        await _cleanup(proc, [first, second])


@pytest.mark.asyncio
async def test_persisted_cancel_request_resumes_only_proved_owned_group(ledger):
    conn, root = ledger
    first = OperationManager(None, conn)
    second = OperationManager(None, conn)
    op = await first.start(task_id='t', attempt_id='', kind='process',
                           argv=[sys.executable, '-c', 'import time;time.sleep(60)'], cwd=str(root))
    proc = first._procs[op['operation_id']]
    try:
        driver = first._drivers[op['operation_id']]
        driver.cancel()
        await asyncio.gather(driver, return_exceptions=True)
        conn.execute("UPDATE operations SET state='CANCELING',cancel_reason='persisted_user_stop' "
                     "WHERE operation_id=?", (op['operation_id'],))
        report = await second.recover_on_boot()
        assert report['terminated'] == 1
        assert second.get(op['operation_id'])['state'] == 'CANCELLED'
        assert not second._pid_alive(proc.pid)
    finally:
        await _cleanup(proc, [first, second])


@pytest.mark.parametrize('route', ['pi.op.cancel', 'pi.task.cancel', 'pi.session.interrupt', 'rest'])
@pytest.mark.asyncio
async def test_control_surfaces_return_unresolved_not_500(tmp_path, route):
    from rosclaw.agentd.pi_bridge.server import PiBridgeServer
    from rosclaw.agentd.service import create_app
    from tests.agentd.test_pi_tool_bridge import _setup

    service, mission = await _setup(tmp_path)
    bound = service._task_kernel.bind_message(
        mission_id=mission.mission_id, session_ref='pi_1', backend_native_id='pi_1',
        message_id='lifecycle_fixture', text='Private lifecycle fixture', cwd=str(tmp_path))
    manager = service._operation_manager
    op = await manager.start(task_id=bound['task_id'], attempt_id='', kind='process',
                              argv=[sys.executable, '-c', 'import time;time.sleep(60)'], cwd=str(tmp_path))
    proc = manager._procs[op['operation_id']]
    try:
        service._store.connection.execute(
            "UPDATE operations SET process_identity_json='' WHERE operation_id=?", (op['operation_id'],))
        if route == 'rest':
            # Invoke the actual FastAPI route in this process's existing loop,
            # avoiding TestClient's cross-loop subprocess transport ownership.
            endpoint = next(r.endpoint for r in create_app(service).routes
                            if getattr(r, 'path', '') == '/missions/{mission_id}/cancel')
            result = await endpoint(mission.mission_id)
            assert result['cancelled'] is False
        else:
            bridge = PiBridgeServer(service, tmp_path / 'run' / 'private.sock')
            result = await bridge._dispatch('user:local:1000', 1, route,
                                           {'token': service.control_token,
                                            'mission_id': mission.mission_id, 'session_ref': 'pi_1',
                                            'task_id': bound['task_id'], 'operation_id': op['operation_id']})
        assert result['ok'] is False and result['code'] == 'CANCEL_STOP_UNCONFIRMED'
        assert result['operations_cancelled'] == 0
        assert result['operations_unresolved'] == [op['operation_id']]
        assert manager.get(op['operation_id'])['state'] == 'CANCELING'
        assert manager._pid_alive(proc.pid)
    finally:
        await _cleanup(proc, [manager])
        await service.close()


@pytest.mark.parametrize('failure', [PermissionError, NotImplementedError])
@pytest.mark.asyncio
async def test_pidfd_failure_is_unresolved_not_cancelled(ledger, monkeypatch, failure):
    import rosclaw.task_kernel.process_identity as module
    from rosclaw.task_kernel.operation_manager import OperationCancellationUnresolvedError

    conn, root = ledger
    manager = OperationManager(None, conn)
    op = await manager.start(task_id='t', attempt_id='', kind='process',
                              argv=[sys.executable, '-c', 'import time;time.sleep(60)'], cwd=str(root))
    proc = manager._procs[op['operation_id']]
    try:
        def denied(pid):
            raise failure('private injected pidfd failure')
        monkeypatch.setattr(module, '_pidfd_open', denied)
        with pytest.raises(OperationCancellationUnresolvedError):
            await manager.cancel(op['operation_id'])
        assert manager._pid_alive(proc.pid)
        assert manager.get(op['operation_id'])['state'] == 'CANCELING'
    finally:
        await _cleanup(proc, [manager])


@pytest.mark.asyncio
async def test_persisted_identity_has_no_argv_or_environment_plaintext(ledger):
    conn, root = ledger
    manager = OperationManager(None, conn)
    fake_secret = 'PRIVATE_FIXTURE_NOT_A_CREDENTIAL'
    op = await manager.start(task_id='t', attempt_id='', kind='process',
                              argv=[sys.executable, '-c', f'import time; marker={fake_secret!r};time.sleep(60)'],
                              env=dict(os.environ, PRIVATE_FIXTURE_ENV=fake_secret), cwd=str(root))
    proc = manager._procs[op['operation_id']]
    try:
        proof = json.loads(op['process_identity_json'])
        assert proof['pid'] == proof['pgid'] == proof['sid'] == proc.pid
        assert proof['start_ticks'] > 0 and len(proof['command_sha256']) == 64
        assert fake_secret not in op['process_identity_json']
        assert 'argv' not in proof and 'env' not in proof
        await manager.cancel(op['operation_id'])
    finally:
        await _cleanup(proc, [manager])


@pytest.mark.asyncio
async def test_legacy_cancelled_row_does_not_claim_physical_stop_or_mutate_history(ledger):
    from types import SimpleNamespace

    from rosclaw.agentd.pi_bridge.tool_dispatch import PiToolDispatcher
    from tests.agentd.test_pi_tool_bridge import _request

    conn, root = ledger
    manager = OperationManager(None, conn)
    op = await manager.start(task_id='t', attempt_id='', kind='process',
                              argv=[sys.executable, '-c', 'import time;time.sleep(60)'], cwd=str(root))
    proc = manager._procs[op['operation_id']]
    try:
        driver = manager._drivers[op['operation_id']]
        driver.cancel()
        await asyncio.gather(driver, return_exceptions=True)
        conn.execute("UPDATE operations SET state='CANCELLED',process_identity_json='' WHERE operation_id=?",
                     (op['operation_id'],))
        before = manager.get(op['operation_id'])
        events = manager.events_since('t', 0)
        service = SimpleNamespace(_store=SimpleNamespace(connection=conn), _operation_manager=manager)
        result = await PiToolDispatcher(service)._process_stop(
            _request('rosclaw_process_stop', arguments={'operation_id': op['operation_id']}))
        assert result.ok is False and result.status == 'CANCELLED'
        assert result.error_code == 'CANCEL_STOP_UNCONFIRMED'
        report = await manager.cancel_many([op['operation_id']])
        assert report['ok'] is False and report['operations_cancelled'] == 0
        assert manager._pid_alive(proc.pid)
        assert manager.get(op['operation_id']) == before
        assert manager.events_since('t', 0) == events
    finally:
        await _cleanup(proc, [manager])


@pytest.mark.asyncio
async def test_db_reopen_identity_mismatch_preserves_pending_cancel_and_sweep_truth(ledger):
    from rosclaw.task_kernel.operation_manager import OperationCancellationUnresolvedError

    conn, root = ledger
    first = OperationManager(None, conn)
    op = await first.start(task_id='t', attempt_id='', kind='process',
                           argv=[sys.executable, '-c', 'import time;time.sleep(60)'], cwd=str(root))
    proc = first._procs[op['operation_id']]
    reopened = None
    second = None
    try:
        driver = first._drivers[op['operation_id']]
        driver.cancel()
        await asyncio.gather(driver, return_exceptions=True)
        proof = json.loads(op['process_identity_json'])
        proof['start_ticks'] += 1
        conn.execute('UPDATE operations SET process_identity_json=? WHERE operation_id=?',
                     (json.dumps(proof), op['operation_id']))
        with pytest.raises(OperationCancellationUnresolvedError):
            await first.cancel(op['operation_id'])
        conn.commit()
        reopened = sqlite3.connect(root / 'private.db')
        reopened.row_factory = sqlite3.Row
        second = OperationManager(None, reopened)
        assert second.get(op['operation_id'])['state'] == 'CANCELING'
        assert (await second.recover_on_boot())['unresolved'] == 1
        for threshold in (0, 86400):
            await second.sweep_liveness(stale_after_s=threshold)
            row = second.get(op['operation_id'])
            assert row['state'] == 'CANCELING'
            assert row['failure_code'] == 'CANCEL_STOP_UNCONFIRMED'
        assert second._pid_alive(proc.pid)
        types = [e['event_type'] for e in second.events_since('t', 0)]
        assert 'operation.recovery_unresolved' in types
        assert 'operation.resumed' not in types and 'operation.cancelled' not in types
    finally:
        await _cleanup(proc, [first] + ([second] if second else []))
        if reopened is not None:
            reopened.close()


@pytest.mark.asyncio
async def test_cancel_many_unknown_id_is_safe_no_success_count_or_events(ledger):
    conn, _ = ledger
    manager = OperationManager(None, conn)
    before = manager.events_since('t', 0)
    assert manager.get('not_a_real_operation') == {}
    report = await manager.cancel_many(['not_a_real_operation'])
    assert report['operations_cancelled'] == 0
    assert manager.events_since('t', 0) == before
