"""Finite real output producers must drain regardless of line length/UTF-8 cuts."""
from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

import pytest

from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.operation_manager import OperationManager
from rosclaw.task_kernel.service import TaskKernel


@pytest.mark.asyncio
@pytest.mark.parametrize('unicode_output', [False, True])
async def test_finite_long_line_drains_and_retains_all_output_and_exitcode(
    tmp_path: Path, unicode_output: bool,
) -> None:
    conn = sqlite3.connect(tmp_path / 'ledger.db')
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, 'sqlite')
    kernel = TaskKernel(conn, tmp_path)
    task = kernel.bind_message(mission_id='m', session_ref='s', backend_native_id='n',
                               message_id='msg', text='bounded output fixture',
                               cwd=str(tmp_path))['task_id']
    manager = OperationManager(kernel, conn)
    unit = '中🎾AB' if unicode_output else 'abcdef'
    # Single unterminated line > reader's default 64KiB and pipe capacity.
    expected = unit * 60000 + '\nSTDERR_END\nTAIL_NO_NEWLINE'
    program = ('import os,sys; '
               f'os.write(1,({unit!r}*60000).encode()); '
               'os.write(2,b"\\nSTDERR_END\\n"); '
               'os.write(1,b"TAIL_NO_NEWLINE"); sys.exit(7)')
    op = await manager.start(task_id=task, attempt_id='', kind='process',
                             argv=[sys.executable, '-c', program], cwd=str(tmp_path))
    op_id = op['operation_id']
    proc = manager._procs[op_id]
    driver = manager._drivers[op_id]
    try:
        final = await manager.wait(op_id, timeout=3)
        assert final['state'] == 'FAILED' and final['failure_code'] == 'exit_7'
        events = manager.events_since(task, 0)
        outputs = [e['payload']['text'] for e in events if e['event_type'] == 'operation.output']
        assert ''.join(outputs) == expected
        assert all(0 < len(text) <= 4000 for text in outputs)
        assert not any(e['event_type'] == 'operation.failed' and
                       'reader:' in e['payload'].get('error', '') for e in events)
        assert proc.returncode is not None
    finally:
        # Old code is stuck in proc.wait after readline fails. Drain its owned
        # finite pipe in cleanup so RED does not leave an orphan or DB task.
        if not driver.done():
            assert proc.stdout is not None
            await asyncio.wait_for(proc.stdout.read(), timeout=3)
            # Yield for the child-watcher notification after natural finite
            # exit; returncode may lag pipe EOF by one event-loop iteration.
            await asyncio.wait_for(proc.wait(), timeout=3)
            await asyncio.wait_for(asyncio.shield(driver), timeout=3)
        await proc.wait()
        conn.close()


@pytest.mark.asyncio
async def test_split_utf8_and_incomplete_final_sequence_decode_once(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / 'ledger.db')
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, 'sqlite')
    manager = OperationManager(None, conn)
    # Force distinct kernel writes; valid code point split must remain intact,
    # incomplete final byte uses the established errors='replace' policy.
    program = ('import os,time; os.write(1,b"\\xe4"); time.sleep(.03); '
               'os.write(1,b"\\xb8\\xad"); time.sleep(.03); '
               'os.write(1,b"\\xf0\\x9f\\x8e\\xbe\\xe4")')
    op = await manager.start(task_id='t', attempt_id='', kind='process',
                             argv=[sys.executable, '-c', program], cwd=str(tmp_path))
    try:
        final = await manager.wait(op['operation_id'], timeout=3)
        assert final['state'] == 'SUCCEEDED'
        assert ''.join(e['payload']['text'] for e in manager.events_since('t', 0)
                       if e['event_type'] == 'operation.output') == '中🎾�'
    finally:
        await manager.wait(op['operation_id'], timeout=3)
        conn.close()
