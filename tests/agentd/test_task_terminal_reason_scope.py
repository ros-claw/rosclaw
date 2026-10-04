"""Current task reasons must not masquerade as historical terminal evidence."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.service import TASK_ACTIVE, TaskKernel


@pytest.fixture
def task(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / 'private.db')
    conn.row_factory = sqlite3.Row
    MigrationRunner().apply(conn, 'sqlite')
    kernel = TaskKernel(conn, tmp_path)
    bound = kernel.bind_message(
        mission_id='m', session_ref='s', backend_native_id='n',
        message_id='first', text='Inspect a private fixture', cwd=str(tmp_path),
    )
    yield conn, kernel, bound['task_id'], tmp_path
    conn.close()


def _events(kernel, task_id):
    return [dict(r) for r in kernel._conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? ORDER BY seq", (task_id,))]


def _stale_reason(conn, task_id):
    # Model persisted pre-fix data; no production DB or history is rewritten.
    conn.execute('UPDATE tasks SET terminal_reason = ? WHERE task_id = ?',
                 ('verification_passed', task_id))


@pytest.mark.parametrize('state', sorted(TASK_ACTIVE))
@pytest.mark.parametrize('reason', ['', 'resume requested'])
def test_active_transition_clears_stale_terminal_reason(task, state, reason):
    conn, kernel, task_id, _ = task
    _stale_reason(conn, task_id)
    kernel.transition(task_id, state, reason=reason)
    assert kernel.get_task(task_id)['terminal_reason'] is None
    events = _events(kernel, task_id)
    event = next(e for e in reversed(events) if e['event_type'] == 'task.state_changed')
    assert json.loads(event['payload_json'])['reason'] == reason
    kernel.transition(task_id, 'RUNNING')
    assert kernel.get_task(task_id)['terminal_reason'] is None


@pytest.mark.parametrize('state', ['FAILED', 'BLOCKED', 'CANCELLED', 'SUCCEEDED'])
def test_empty_terminal_reason_never_inherits_unrelated_success(task, state):
    conn, kernel, task_id, _ = task
    _stale_reason(conn, task_id)
    kernel.transition(task_id, state)
    assert kernel.get_task(task_id)['terminal_reason'] is None


def test_blocked_reason_and_outcome_survive_rejected_resume(task):
    from rosclaw.task_kernel.coordinator import TaskCoordinator

    _, kernel, task_id, _ = task
    kernel.block_task(task_id=task_id, reason_code='MISSING_CAPABILITY',
                      detail='sensor unavailable', recovery=['install sensor'])
    outcome = TaskCoordinator(kernel).consider(task_id)
    assert any('MISSING_CAPABILITY' in x for x in outcome['repair_directive']['failures'])
    events_before = _events(kernel, task_id)
    kernel.transition(task_id, 'RUNNING')
    kernel.transition(task_id, 'BLOCKED')
    assert kernel.get_task(task_id)['state'] == 'BLOCKED'
    assert kernel.get_task(task_id)['terminal_reason'] == 'MISSING_CAPABILITY: sensor unavailable'
    assert _events(kernel, task_id) == events_before
    assert TaskCoordinator(kernel).consider(task_id) == outcome


def test_active_revision_clears_stale_reason_without_erasing_history(task):
    conn, kernel, task_id, tmp_path = task
    kernel.transition(task_id, 'WAITING_INPUT', reason='clarify target')
    _stale_reason(conn, task_id)
    events_before = _events(kernel, task_id)
    old_outcome = '{"revision":1,"verification":"FAIL","fixture":"historical"}'
    conn.execute('INSERT INTO task_outcomes VALUES (?, ?, ?, ?, ?)',
                 ('historical', task_id, 1, old_outcome, 'then'))
    bound = kernel.bind_message(
        mission_id='m', session_ref='s', backend_native_id='n',
        message_id='second', text='Use the clarified target', cwd=str(tmp_path),
    )
    assert bound['task_id'] == task_id and bound['revision'] == 2
    assert kernel.get_task(task_id)['state'] == 'WAITING_INPUT'
    assert kernel.get_task(task_id)['terminal_reason'] is None
    assert _events(kernel, task_id)[:len(events_before)] == events_before
    assert conn.execute('SELECT outcome_json FROM task_outcomes WHERE task_id = ?',
                        (task_id,)).fetchone()[0] == old_outcome
    replay = kernel.bind_message(
        mission_id='m', session_ref='s', backend_native_id='n',
        message_id='second', text='Use the clarified target', cwd=str(tmp_path),
    )
    assert replay['replayed'] and replay['revision'] == 2


def test_succeeded_reopen_retains_old_terminal_event_and_outcome(task):
    conn, kernel, task_id, tmp_path = task
    kernel.transition(task_id, 'SUCCEEDED', reason='verification_passed')
    before = _events(kernel, task_id)
    old_outcome = '{"revision":1,"verification":"PASS","fixture":"historical"}'
    conn.execute('INSERT INTO task_outcomes VALUES (?, ?, ?, ?, ?)',
                 ('historical', task_id, 1, old_outcome, 'then'))
    kernel.bind_message(mission_id='m', session_ref='s', backend_native_id='n',
                        message_id='reject', text='Correction required', cwd=str(tmp_path))
    current = kernel.get_task(task_id)
    assert current['state'] == 'RUNNING' and current['active_revision'] == 2
    assert current['terminal_reason'] is None
    assert _events(kernel, task_id)[:len(before)] == before
    assert conn.execute('SELECT outcome_json FROM task_outcomes WHERE task_id = ?',
                        (task_id,)).fetchone()[0] == old_outcome


def test_waiting_explanation_does_not_become_later_failure_reason(task):
    _, kernel, task_id, _ = task
    kernel.transition(task_id, 'WAITING_INPUT', reason='choose a destination')
    kernel.transition(task_id, 'RUNNING')
    kernel.transition(task_id, 'FAILED')
    assert kernel.get_task(task_id)['terminal_reason'] is None
    reasons = [json.loads(e['payload_json'])['reason']
               for e in _events(kernel, task_id) if e['event_type'] == 'task.state_changed']
    assert reasons == ['choose a destination', '', '']
