"""Pure oracle tests: fabricated transcripts are NOT actual DDS evidence.

No runtime entry, SDK, subprocess, socket, container or real file is used.
"""
import copy
import io
import json
import unittest
from unittest.mock import patch

import frozen_clock_rpc_scenario as scenario


OWNERS = dict(context='10', server='11', client_node='12', executor='13', client='14')


def closed_fixture():
    """Hand-authored synthetic timing, not runtime-generated expected output."""
    rows = []
    start = 100 * scenario.NS
    service = '/frozen_rpc_case_nonce'
    def emit(seconds, event, **data):
        rows.append(dict(schema=scenario.SCHEMA, case='case', nonce='nonce', domain=83,
                         seq=len(rows), mono_ns=start + int(seconds * scenario.NS),
                         event=event, data=data))
    def link(rid):
        return dict(request_id=rid, future_id={'FAST': '21', 'LATE': '22', 'RECOVERY': '23'}[rid],
                    owners=dict(OWNERS))
    def clock(seconds):
        emit(seconds, 'CLOCK_SAMPLE', owners=dict(OWNERS), ros_ns=[0, 0],
             active=[True, True], use_sim_time=[{'type': 1, 'value': True}] * 2)
    def submit(seconds, rid):
        t = start + int(seconds * scenario.NS)
        emit(seconds, 'REQUEST_SUBMITTED', **link(rid), submit_ns=t,
             deadline_ns=t + scenario.DURATIONS[rid], request={'names': ['nonce/' + rid]})
    def received(seconds, rid):
        emit(seconds, 'SERVER_REQUEST_RECEIVED', **link(rid), request={'names': ['nonce/' + rid]})
    def response(seconds, event, rid):
        emit(seconds, event, **link(rid), response={'values': [{'type': 4, 'string_value': 'nonce/' + rid}]})
    def terminal(seconds, rid):
        emit(seconds, 'REQUEST_TERMINAL', **link(rid),
             status='TIMED_OUT' if rid == 'LATE' else 'SUCCESS', future_pending=(rid == 'LATE'))
        terminal_row = rows[-1]
        # Independent fabricated completion samples, not runtime output.
        emit(seconds, 'TERMINAL_SYNC_COMPLETED', **link(rid),
             terminal_seq=terminal_row['seq'], clock='time.monotonic_ns',
             before_ns=terminal_row['mono_ns'] + 1, completed_ns=terminal_row['mono_ns'] + 2)
        rows[-1]['mono_ns'] = terminal_row['mono_ns'] + 3
    emit(0, 'BOOT', start_ns=start, work_end_ns=start + 60 * scenario.NS,
         whole_end_ns=start + 90 * scenario.NS)
    entities = {node: {category: [] for category in scenario.CATEGORIES}
                for node in ('server', 'client_node')}
    entities['server'].update(publishers=[{'id': '31', 'name': '/parameter_events'}],
                              subscriptions=[{'id': '32', 'name': '/clock'}],
                              services=[{'id': '33', 'name': service}])
    entities['client_node'].update(publishers=[{'id': '34', 'name': '/parameter_events'}],
                                   subscriptions=[{'id': '35', 'name': '/clock'}],
                                   clients=[{'id': '14', 'name': service}])
    emit(.01, 'ENTITY_INVENTORY', owners=dict(OWNERS), service=service, entities=entities)
    clock(.02)
    submit(.03, 'FAST')
    received(.04, 'FAST')
    response(.05, 'SERVER_RESPONSE_READY', 'FAST')
    response(.06, 'CLIENT_RESPONSE_OBSERVED', 'FAST')
    terminal(.07, 'FAST')
    clock(.08)
    submit(.09, 'LATE')
    received(.10, 'LATE')
    clock(.50)
    terminal(.60, 'LATE')
    clock(.61)
    submit(.62, 'RECOVERY')
    received(.63, 'RECOVERY')
    response(.64, 'SERVER_RESPONSE_READY', 'RECOVERY')
    response(.65, 'CLIENT_RESPONSE_OBSERVED', 'RECOVERY')
    terminal(.66, 'RECOVERY')
    clock(1.00)
    response(2.11, 'SERVER_RESPONSE_READY', 'LATE')
    response(2.12, 'CLIENT_RESPONSE_OBSERVED', 'LATE')
    response(2.13, 'LATE_RESPONSE_QUARANTINED', 'LATE')
    clock(2.14)
    emit(2.15, 'CLEANUP', owners=dict(OWNERS), executor_shutdown=True,
         released_futures=['21', '22', '23'], destroyed_nodes=['12', '11'],
         context_shutdown=True, entities_released=True)
    emit(2.16, 'FINAL', status='PASS', terminals={'FAST': 'SUCCESS', 'LATE': 'TIMED_OUT', 'RECOVERY': 'SUCCESS'}, errors=[])
    return rows


def find(rows, event, rid=None):
    return next(r for r in rows if r['event'] == event and
                (rid is None or r['data'].get('request_id') == rid))


def resequence(rows):
    for i, row in enumerate(rows):
        row['seq'] = i


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.rows = closed_fixture()

    def reject(self, change):
        rows = copy.deepcopy(self.rows)
        change(rows)
        with self.assertRaises((ValueError, TypeError)):
            scenario.validate_transcript(rows)

    def test_closed_synthetic_fixture(self):
        self.assertTrue(scenario.validate_transcript(self.rows))

    def test_partial_never_passes(self):
        self.assertFalse(scenario.validate_transcript(self.rows[:13], require_complete=False))
        rows = copy.deepcopy(self.rows)
        rows[-1]['data']['status'] = 'FAIL'
        self.assertFalse(scenario.validate_transcript(rows, require_complete=False))
        with self.assertRaises(ValueError):
            scenario.validate_transcript(self.rows, require_complete=False)

    def test_unfrozen_inactive_or_wrong_parameter(self):
        for field, value in [('ros_ns', [0, 1]), ('ros_ns', [False, 0]),
                             ('active', [False, False]), ('active', [1, 1]),
                             ('use_sim_time', [{'type': True, 'value': True}] * 2),
                             ('use_sim_time', [{'type': 1, 'value': False}] * 2)]:
            with self.subTest(field=field, value=value):
                self.reject(lambda rs: find(rs, 'CLOCK_SAMPLE')['data'].__setitem__(field, value))

    def test_timeout_early_and_late(self):
        for offset in (.49, 1.51):
            # Move the whole suffix so monotonic ordering is preserved; reject
            # the timeout window, not merely a backwards timestamp.
            def change(rs, offset=offset):
                terminal = find(rs, 'REQUEST_TERMINAL', 'LATE')
                delta = int((.09 + offset) * scenario.NS) + 100 * scenario.NS - terminal['mono_ns']
                for row in rs[terminal['seq']:]:
                    row['mono_ns'] += delta
                    if row['event'] == 'TERMINAL_SYNC_COMPLETED':
                        row['data']['before_ns'] += delta
                        row['data']['completed_ns'] += delta
            self.reject(change)

    def test_pending_timeout_required(self):
        self.reject(lambda rs: find(rs, 'REQUEST_TERMINAL', 'LATE')['data'].__setitem__('future_pending', False))

    def test_terminal_cannot_revive(self):
        self.reject(lambda rs: rs[-1]['data']['terminals'].__setitem__('LATE', 'SUCCESS'))
        def duplicate(rs):
            row = copy.deepcopy(find(rs, 'REQUEST_TERMINAL', 'LATE'))
            row['mono_ns'] = find(rs, 'LATE_RESPONSE_QUARANTINED')['mono_ns']
            row['data'].update(status='SUCCESS', future_pending=False)
            rs.insert(find(rs, 'LATE_RESPONSE_QUARANTINED')['seq'] + 1, row)
            resequence(rs)
        self.reject(duplicate)

    def test_nonce_future_and_typed_payload(self):
        for event, rid, field, value in [
            ('SERVER_REQUEST_RECEIVED', 'LATE', 'request', {'names': ['other/LATE']}),
            ('CLIENT_RESPONSE_OBSERVED', 'RECOVERY', 'future_id', '22'),
            ('CLIENT_RESPONSE_OBSERVED', 'RECOVERY', 'response', {'values': [{'type': 4, 'string_value': 'nonce/LATE'}]}),
            ('CLIENT_RESPONSE_OBSERVED', 'FAST', 'response', {'values': [{'type': 4.0, 'string_value': 'nonce/FAST'}]}),
            ('CLIENT_RESPONSE_OBSERVED', 'FAST', 'response', {'values': [{'type': True, 'string_value': 'nonce/FAST'}]}),
        ]:
            with self.subTest(event=event, field=field, value=value):
                self.reject(lambda rs: find(rs, event, rid)['data'].__setitem__(field, value))
        self.reject(lambda rs: rs[4].__setitem__('nonce', 'other'))

    def test_server_ready_is_not_client_evidence(self):
        def change(rs):
            rs.remove(find(rs, 'CLIENT_RESPONSE_OBSERVED', 'LATE'))
            resequence(rs)
        self.reject(change)

    def test_server_delay_not_shortened(self):
        self.reject(lambda rs: find(rs, 'SERVER_RESPONSE_READY', 'LATE').__setitem__('mono_ns', 102 * scenario.NS))

    def test_recovery_cannot_precede_timeout(self):
        def change(rs):
            terminal = find(rs, 'REQUEST_TERMINAL', 'LATE')
            terminal['event'] = 'ERROR'
            terminal['data'] = {'class': 'Synthetic', 'message': 'removed timeout'}
        self.reject(change)

    def test_recovery_must_precede_late_observation(self):
        def change(rs):
            recovery = find(rs, 'REQUEST_TERMINAL', 'RECOVERY')
            recovery['event'] = 'ERROR'
            recovery['data'] = {'class': 'Synthetic', 'message': 'no recovery terminal'}
        self.reject(change)

    def test_context_client_identity_not_recreated(self):
        for key in OWNERS:
            self.reject(lambda rs, key=key: find(rs, 'REQUEST_SUBMITTED', 'RECOVERY')['data']['owners'].__setitem__(key, '999'))

    def test_original_clock_cannot_reset(self):
        for field in ('start_ns', 'work_end_ns', 'whole_end_ns'):
            self.reject(lambda rs, field=field: rs[0]['data'].__setitem__(field, rs[0]['data'][field] + scenario.NS))
        self.reject(lambda rs: find(rs, 'REQUEST_SUBMITTED', 'RECOVERY')['data'].__setitem__('deadline_ns', 200 * scenario.NS))
        self.reject(lambda rs: rs[-1].__setitem__('mono_ns', 191 * scenario.NS))

    def test_cleanup_omissions(self):
        for field, value in [('executor_shutdown', False), ('context_shutdown', False),
                             ('entities_released', False), ('destroyed_nodes', ['11']),
                             ('released_futures', ['21', '23'])]:
            self.reject(lambda rs, field=field, value=value: find(rs, 'CLEANUP')['data'].__setitem__(field, value))
        def remove(rs):
            rs.remove(find(rs, 'CLEANUP'))
            resequence(rs)
        self.reject(remove)

    def test_final_clock_readback_required(self):
        def change(rs):
            rs.pop(find(rs, 'CLEANUP')['seq'] - 1)
            resequence(rs)
        self.reject(change)

    def test_duplicate_final_seq_and_closed_keys(self):
        self.reject(lambda rs: rs.append(copy.deepcopy(rs[-1])))
        self.reject(lambda rs: rs[3].__setitem__('seq', 2))
        self.reject(lambda rs: rs[3]['data'].__setitem__('unknown', 0))
        self.reject(lambda rs: rs[3].__setitem__('unexpected', 0))
        self.reject(lambda rs: rs[1]['data']['entities']['server']['timers'].append({'id': '90', 'name': 'unowned'}))

    def test_nonfinite_and_bool_numeric(self):
        for value in (True, float('nan'), float('inf'), 100.0):
            self.reject(lambda rs, value=value: rs[0]['data'].__setitem__('start_ns', value))
        self.reject(lambda rs: rs[0].__setitem__('domain', True))
        self.reject(lambda rs: rs[0].__setitem__('seq', False))
        self.reject(lambda rs: rs[0].__setitem__('mono_ns', True))

    def test_bounds(self):
        self.reject(lambda rs: rs.extend(copy.deepcopy(rs) * 10))
        self.reject(lambda rs: rs[0]['data'].__setitem__('start_ns', 'x' * scenario.LIMIT))

    def test_json_parser_without_disk(self):
        raw = ''.join(json.dumps(r) + '\n' for r in self.rows).encode()
        with patch('builtins.open', return_value=io.BytesIO(raw)):
            self.assertEqual(scenario.read_jsonl('unused'), self.rows)
        for raw in (b'{"seq":0,"seq":1}\n', b'{"n":NaN}\n', b'{"n":1e999}\n',
                    b'{}', b'\n', b'\xff\n', b'x' * (scenario.LIMIT + 1)):
            with self.subTest(raw=raw[:40]):
                with patch('builtins.open', return_value=io.BytesIO(raw)):
                    with self.assertRaises((ValueError, UnicodeError)):
                        scenario.read_jsonl('unused')

    def completion_at(self, rid, completed):
        """Keep the event legal and move only the receipt and following work.

        Later submits retain their original duration, not the target submit.
        This makes slow-sync rejection independent of backwards row order.
        """
        rows = copy.deepcopy(self.rows)
        receipt = find(rows, 'TERMINAL_SYNC_COMPLETED', rid)
        delta = completed + 1 - receipt['mono_ns']
        for row in rows[receipt['seq']:]:
            row['mono_ns'] += delta
            if row['event'] == 'REQUEST_SUBMITTED':
                row['data']['submit_ns'] += delta
                row['data']['deadline_ns'] += delta
            if row['event'] == 'TERMINAL_SYNC_COMPLETED' and row is not receipt:
                row['data']['before_ns'] += delta
                row['data']['completed_ns'] += delta
        receipt['data']['completed_ns'] = completed
        return rows

    def test_actual_completion_deadlines_all_three(self):
        for rid in scenario.REQUESTS:
            with self.subTest(rid=rid):
                sub = find(self.rows, 'REQUEST_SUBMITTED', rid)['data']
                upper = sub['submit_ns'] + 3 * scenario.NS // 2 if rid == 'LATE' else sub['deadline_ns']
                original = find(self.rows, 'REQUEST_TERMINAL', rid)
                self.assertLess(original['mono_ns'], upper)
                # Exact original bound accepted; just one ns over rejected,
                # while the original pre-write timestamp is unchanged/legal.
                on_time = self.completion_at(rid, upper)
                self.assertEqual(find(on_time, 'REQUEST_TERMINAL', rid), original)
                self.assertTrue(scenario.validate_transcript(on_time))
                slow = self.completion_at(rid, upper + 1)
                self.assertEqual(find(slow, 'REQUEST_TERMINAL', rid), original)
                with self.assertRaisesRegex(ValueError, 'actual fsync deadline'):
                    scenario.validate_transcript(slow)

    def test_completion_clock_order_domain_types_and_association(self):
        terminal = find(self.rows, 'REQUEST_TERMINAL', 'FAST')
        t = terminal['mono_ns']
        for field, value in [('clock', 'time.time_ns'), ('clock', 'ROS'),
                             ('before_ns', t - 1), ('completed_ns', t),
                             ('completed_ns', t + 4), ('completed_ns', True),
                             ('before_ns', float(t)), ('terminal_seq', True),
                             ('terminal_seq', terminal['seq'] - 1),
                             ('future_id', '22')]:
            with self.subTest(field=field, value=value):
                self.reject(lambda rs: find(rs, 'TERMINAL_SYNC_COMPLETED', 'FAST')['data'].__setitem__(field, value))
        # Old schema or a PASS field cannot substitute for sampled completion.
        self.reject(lambda rs: find(rs, 'TERMINAL_SYNC_COMPLETED', 'FAST')['data'].__setitem__('success', True))
        self.reject(lambda rs: rs[0].__setitem__('schema', 'ROSClaw.FrozenClockRPC.v1'))

    def test_completion_required_unique_and_before_progress(self):
        for rid in scenario.REQUESTS:
            def remove(rs, rid=rid):
                rs.remove(find(rs, 'TERMINAL_SYNC_COMPLETED', rid))
                resequence(rs)
            self.reject(remove)
        def duplicate(rs):
            receipt = find(rs, 'TERMINAL_SYNC_COMPLETED', 'FAST')
            rs.insert(receipt['seq'] + 1, copy.deepcopy(receipt))
            resequence(rs)
        self.reject(duplicate)
        def reorder(rs):
            receipt = find(rs, 'TERMINAL_SYNC_COMPLETED', 'FAST')
            i = receipt['seq']
            rs[i], rs[i + 1] = rs[i + 1], receipt
            receipt['mono_ns'] = rs[i]['mono_ns']
            resequence(rs)
        self.reject(reorder)
        # A failed prefix retains the terminal but cannot certify completion.
        terminal = find(self.rows, 'REQUEST_TERMINAL', 'FAST')
        prefix = self.rows[:terminal['seq'] + 1]
        self.assertFalse(scenario.validate_transcript(prefix, require_complete=False))
        with self.assertRaises(ValueError):
            scenario.validate_transcript(prefix)

    def test_sync_samples_after_actual_call_and_propagates_failures(self):
        calls = []
        class Stream:
            def fileno(self):
                calls.append('fileno')
                return 17
        samples = iter([101, 107])
        def clock():
            calls.append('clock')
            return next(samples)
        def sync(fd):
            calls.append(('fsync', fd))
        timing = scenario.terminal_sync(Stream(), clock, sync)
        self.assertEqual(calls, ['clock', 'fileno', ('fsync', 17), 'clock'])
        self.assertEqual(timing, dict(clock='time.monotonic_ns', before_ns=101, completed_ns=107))
        for failure_at in ('fsync', 'completion_clock', 'initial_clock'):
            calls.clear()
            def failing_clock():
                calls.append('clock')
                if failure_at == 'initial_clock' or (failure_at == 'completion_clock' and len(calls) > 1):
                    raise OSError('clock failure')
                return 101
            def failing_sync(fd):
                calls.append(('fsync', fd))
                if failure_at == 'fsync':
                    raise OSError('fsync failure')
            with self.subTest(failure_at=failure_at):
                with self.assertRaises(OSError):
                    scenario.terminal_sync(Stream(), failing_clock, failing_sync)
                self.assertEqual(calls.count('clock'), 2 if failure_at == 'completion_clock' else 1)

    def test_sync_slow_and_invalid_sample_controls(self):
        # Exercise the same producer and consumer used at runtime, with pure
        # injected clocks. No disk/SDK and no asserted success flag.
        class Stream:
            def fileno(self):
                return 19
        for rid in scenario.REQUESTS:
            terminal = find(self.rows, 'REQUEST_TERMINAL', rid)
            sub = find(self.rows, 'REQUEST_SUBMITTED', rid)['data']
            upper = sub['submit_ns'] + 3 * scenario.NS // 2 if rid == 'LATE' else sub['deadline_ns']
            before = terminal['mono_ns'] + 1
            for completed, valid in [(upper, True), (upper + 1, False),
                                     (before - 1, False), (True, False),
                                     (float(upper), False)]:
                with self.subTest(rid=rid, completed=completed):
                    samples = iter([before, completed])
                    sync_calls = []
                    timing = scenario.terminal_sync(Stream(), lambda: next(samples), sync_calls.append)
                    receipt = copy.deepcopy(find(self.rows, 'TERMINAL_SYNC_COMPLETED', rid))
                    receipt['data'].update(timing)
                    receipt['mono_ns'] = upper + 2
                    self.assertEqual(sync_calls, [19])
                    if valid:
                        scenario.check_terminal_completion(terminal, receipt, sub['submit_ns'],
                                                           sub['deadline_ns'], self.rows[0]['data']['work_end_ns'])
                    else:
                        with self.assertRaises(ValueError):
                            scenario.check_terminal_completion(terminal, receipt, sub['submit_ns'],
                                                               sub['deadline_ns'], self.rows[0]['data']['work_end_ns'])

    def test_invalid_runtime_arguments_never_enter_sdk_boundary(self):
        with patch.object(scenario, '_runtime', side_effect=AssertionError('must not enter')):
            for args in [('', 'n', 1, 'x'), ('c', '/', 1, 'x'), ('c', 'n', True, 'x'), ('c', 'n', 233, 'x')]:
                with self.assertRaises(ValueError):
                    scenario.run_frozen_clock_rpc_scenario(*args)


if __name__ == '__main__':
    unittest.main()
