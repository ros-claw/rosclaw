"""Real OS children must be reaped before an explicitly closed journal.

No model or robot is used here; the separate Native/Pi exit trial validates
the actual product frontend and service wiring with a real model.
"""
import asyncio
import sqlite3
import sys

import pytest

from rosclaw.task_kernel.operation_manager import OperationManager
from tests.agentd.test_p1b1_operation_v2 import _conn, _events, _task
from tests.agentd.test_pi_tool_bridge import _setup


def manager(tmp_path):
    connection=_conn(tmp_path);_task(connection)
    return OperationManager(None,connection),connection


async def start(mgr,code='import time; time.sleep(30)'):
    return await mgr.start(task_id='task_1',attempt_id='',kind='process',argv=[sys.executable,'-c',code])


async def test_close_reaps_owned_process_and_is_idempotent(tmp_path):
    mgr,conn=manager(tmp_path)
    operation=await start(mgr);proc=mgr._procs[operation['operation_id']]
    await asyncio.gather(mgr.close(),mgr.close())
    assert proc.returncode is not None
    assert mgr.get(operation['operation_id'])['state']=='CANCELLED'
    assert mgr.get(operation['operation_id'])['cancel_reason']=='supervisor_shutdown'
    await mgr.close();conn.close()


async def test_close_waits_for_an_already_owned_cancel_and_sigkill_reap(tmp_path):
    mgr,conn=manager(tmp_path)
    operation=await start(mgr,'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("READY",flush=True); time.sleep(30)')
    proc=mgr._procs[operation['operation_id']]
    async with asyncio.timeout(5):
        while not any(e['type']=='operation.output' for e in _events(conn)):
            await asyncio.sleep(.01)
    cancellation=asyncio.create_task(mgr.cancel(operation['operation_id'],reason='original_cancel'))
    await asyncio.sleep(.02)
    await mgr.close();await cancellation
    assert proc.returncode is not None
    assert mgr.get(operation['operation_id'])['state']=='CANCELLED'
    assert mgr.get(operation['operation_id'])['cancel_reason']=='original_cancel'
    conn.close()


@pytest.mark.parametrize('disconnect_caller',[False,True])
async def test_shutdown_owns_a_spawn_in_flight_even_if_caller_disconnects(tmp_path,disconnect_caller):
    mgr,conn=manager(tmp_path)
    entered=asyncio.Event();release=asyncio.Event();spawn=mgr._spawn;children=[]
    async def delayed(*args):
        entered.set();await release.wait()
        process=await spawn(*args);children.append(process);return process
    mgr._spawn=delayed
    request=asyncio.create_task(start(mgr));await entered.wait()
    if disconnect_caller:
        request.cancel()
        with pytest.raises(asyncio.CancelledError):await request
    closing=asyncio.create_task(mgr.close());await asyncio.sleep(.01)
    assert not closing.done()
    release.set();await closing
    if not disconnect_caller:await request
    assert children and all(p.returncode is not None for p in children)
    assert all(row['state']=='CANCELLED' for row in conn.execute('SELECT state FROM operations'))
    conn.close()


async def test_start_after_shutdown_does_not_admit_or_spawn(tmp_path):
    mgr,conn=manager(tmp_path);await mgr.close()
    with pytest.raises(RuntimeError,match='closing'):await start(mgr)
    assert conn.execute('SELECT COUNT(*) FROM operations').fetchone()[0]==0
    conn.close()


async def test_actual_agent_service_closes_operations_before_its_event_store(tmp_path):
    service,mission=await _setup(tmp_path)
    task=service._task_kernel.ensure_task_for_effect(mission_id=mission.mission_id,
        session_ref='pi_1',backend_native_id='',cwd=str(tmp_path))
    operation=await service._operation_manager.start(task_id=task['task_id'],attempt_id='',
        kind='process',argv=[sys.executable,'-c','import time; time.sleep(30)'])
    proc=service._operation_manager._procs[operation['operation_id']]
    await service.close()
    assert proc.returncode is not None
    connection=sqlite3.connect('file:'+str(tmp_path/'agentd/missions.db')+'?mode=ro',uri=True)
    try:
        row=connection.execute('SELECT state FROM operations WHERE operation_id=?',
                               (operation['operation_id'],)).fetchone()
        assert row[0]=='CANCELLED'
    finally:connection.close()
