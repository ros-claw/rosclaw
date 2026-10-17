"""Frozen ROS clock diagnostic RPC v1. Import and validators are stdlib-only.

Source-only until independently authorized and executed in the pinned image.
"""
import argparse
import json
import math
import os
import re
import threading
import time

SCHEMA = 'ROSClaw.FrozenClockRPC.v1.1'
NS = 1_000_000_000
WORK_NS = 60 * NS
WHOLE_NS = 90 * NS
LIMIT = 1024 * 1024
REQUESTS = ('FAST', 'LATE', 'RECOVERY')
DURATIONS = {'FAST': 2 * NS, 'LATE': NS // 2, 'RECOVERY': NS}
COMMON = {'schema', 'case', 'nonce', 'domain', 'seq', 'mono_ns', 'event', 'data'}
LINK = {'request_id', 'future_id', 'owners'}
KEYS = {
    'BOOT': {'start_ns', 'work_end_ns', 'whole_end_ns'},
    'ENTITY_INVENTORY': {'owners', 'service', 'entities'},
    'CLOCK_SAMPLE': {'owners', 'ros_ns', 'active', 'use_sim_time'},
    'REQUEST_SUBMITTED': LINK | {'submit_ns', 'deadline_ns', 'request'},
    'SERVER_REQUEST_RECEIVED': LINK | {'request'},
    'SERVER_RESPONSE_READY': LINK | {'response'},
    'CLIENT_RESPONSE_OBSERVED': LINK | {'response'},
    'REQUEST_TERMINAL': LINK | {'status', 'future_pending'},
    'TERMINAL_SYNC_COMPLETED': LINK | {'terminal_seq', 'clock', 'before_ns', 'completed_ns'},
    'LATE_RESPONSE_QUARANTINED': LINK | {'response'},
    'ERROR': {'class', 'message'},
    'CLEANUP': {'owners', 'executor_shutdown', 'released_futures',
                'destroyed_nodes', 'context_shutdown', 'entities_released'},
    'FINAL': {'status', 'terminals', 'errors'},
}
CATEGORIES = ('publishers', 'subscriptions', 'services', 'clients', 'timers', 'guards')


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def integer(value):
    return type(value) is int


def identifier(value):
    return type(value) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value) is not None


def wire_request(nonce, rid):
    return {'names': [nonce + '/' + rid]}


def wire_response(nonce, rid):
    return {'values': [{'type': 4, 'string_value': nonce + '/' + rid}]}


def _tree(value):
    """Reject bool/int confusion in numeric leaves separately at schema sites."""
    need(type(value) in (dict, list, str, int, float, bool, type(None)), 'JSON type')
    if type(value) is float:
        need(math.isfinite(value), 'nonfinite')
    if type(value) is dict:
        need(all(type(k) is str for k in value), 'object key')
        for item in value.values():
            _tree(item)
    if type(value) is list:
        for item in value:
            _tree(item)


def _payload(value, expected):
    # Canonical encoding distinguishes true from 1 and 4.0 from 4.
    need(json.dumps(value, sort_keys=True, allow_nan=False) ==
         json.dumps(expected, sort_keys=True), 'typed SDK projection')


def terminal_sync(stream, clock, sync):
    """Sample around the actual sync call, never derive completion from a row.

    Dependency injection is for pure tests; runtime supplies os.fsync and the
    very same time.monotonic_ns used by BOOT, submit and event timestamps.
    An exception produces no completion receipt and is not converted to success.
    """
    before = clock()
    sync(stream.fileno())
    completed = clock()
    return dict(clock='time.monotonic_ns', before_ns=before, completed_ns=completed)


def check_terminal_completion(terminal, receipt, submit_ns, deadline_ns, work_end):
    """Pure timing/association check, independent of any PASS/status claim."""
    d = receipt['data']
    rid = terminal['data']['request_id']
    upper = submit_ns + 3 * NS // 2 if rid == 'LATE' else deadline_ns
    need(d['clock'] == 'time.monotonic_ns', 'completion clock domain')
    need(integer(d['terminal_seq']) and d['terminal_seq'] == terminal['seq'], 'sync terminal correlation')
    need(all(d[k] == terminal['data'][k] for k in LINK), 'sync request/future/owners')
    before, completed = d['before_ns'], d['completed_ns']
    need(integer(before) and integer(completed) and
         terminal['mono_ns'] <= before <= completed <= receipt['mono_ns'], 'completion clock order/type')
    need(submit_ns <= terminal['mono_ns'] <= completed <= upper and
         completed < work_end, 'terminal actual fsync deadline')
    if rid == 'LATE':
        need(terminal['mono_ns'] >= deadline_ns, 'timeout before deadline')


def validate_transcript(rows, require_complete=True):
    """Closed-schema replay oracle. Only a complete valid PASS returns True.

    Failure/prefix mode validates structural/causal evidence but returns False.
    It never upgrades a failed run or treats fixtures as DDS evidence.
    """
    need(type(rows) is list and 0 < len(rows) <= 256, 'row count')
    _tree(rows)
    need(sum(len((json.dumps(r, ensure_ascii=False, allow_nan=False,
                             separators=(',', ':')) + '\n').encode('utf-8'))
             for r in rows) <= LIMIT, 'byte bound')
    identity = owners = inventory = boot = cleanup = final = None
    previous = -1
    submitted, received, ready, observed, terminals = {}, {}, {}, {}, {}
    future_ids, errors, samples = set(), [], []
    synced = set()
    quarantined = False
    for seq, row in enumerate(rows):
        need(type(row) is dict and set(row) == COMMON, 'row keys')
        need(row['schema'] == SCHEMA and integer(row['seq']) and row['seq'] == seq, 'schema/seq')
        need(identifier(row['case']) and identifier(row['nonce']), 'case/nonce')
        need(integer(row['domain']) and 0 <= row['domain'] <= 232, 'domain')
        current = (row['case'], row['nonce'], row['domain'])
        identity = identity or current
        need(current == identity, 'identity changed')
        t, event, d = row['mono_ns'], row['event'], row['data']
        need(integer(t) and t >= 0 and t >= previous, 'monotonic domain/order')
        prior_t = previous
        previous = t
        need(type(event) is str and event in KEYS and type(d) is dict and set(d) == KEYS[event], 'event schema')
        need(final is None, 'after FINAL')
        if seq == 0:
            need(event == 'BOOT', 'BOOT first')
        if event == 'BOOT':
            need(seq == 0, 'duplicate BOOT')
            need(all(integer(v) and v >= 0 for v in d.values()), 'clock types')
            need(d['start_ns'] <= t and d['work_end_ns'] == d['start_ns'] + WORK_NS and
                 d['whole_end_ns'] == d['start_ns'] + WHOLE_NS, 'original clock cutoffs')
            boot = d
        elif event == 'ENTITY_INVENTORY':
            need(seq == 1 and inventory is None, 'inventory position')
            owners = d['owners']
            need(type(owners) is dict and set(owners) == {'context', 'server', 'client_node', 'executor', 'client'}, 'owner keys')
            need(all(type(v) is str and v.isdecimal() and int(v) > 0 for v in owners.values()) and
                 len(set(owners.values())) == 5, 'owner identities')
            expected_service = '/frozen_rpc_' + identity[0] + '_' + identity[1]
            need(d['service'] == expected_service, 'unique diagnostic service')
            entities = d['entities']
            need(type(entities) is dict and set(entities) == {'server', 'client_node'}, 'inventory nodes')
            ids = set()
            totals = dict.fromkeys(CATEGORIES, 0)
            for node in entities.values():
                need(type(node) is dict and set(node) == set(CATEGORIES), 'entity categories')
                for category, items in node.items():
                    need(type(items) is list, 'entity list')
                    totals[category] += len(items)
                    for item in items:
                        need(type(item) is dict and set(item) == {'id', 'name'}, 'entity shape')
                        need(type(item['id']) is str and item['id'].isdecimal() and item['id'] not in ids and
                             type(item['name']) is str, 'entity identity')
                        ids.add(item['id'])
            need(totals == dict(publishers=2, subscriptions=2, services=1, clients=1, timers=0, guards=0), 'logical entity counts')
            need(entities['server']['services'][0]['name'] == expected_service and
                 entities['client_node']['clients'][0] == {'id': owners['client'], 'name': expected_service}, 'service/client inventory')
            need(all(n['subscriptions'][0]['name'] == '/clock' for n in entities.values()), 'clock subscriptions')
            inventory = d
        elif event == 'ERROR':
            need(all(type(v) is str and len(v) <= 2048 for v in d.values()), 'error text')
            errors.append(d)
        elif event == 'FINAL':
            need(d['status'] in ('PASS', 'FAIL') and type(d['terminals']) is dict and
                 set(d['terminals']) == set(REQUESTS), 'FINAL shape')
            need(all(v in ('SUCCESS', 'TIMED_OUT', 'UNKNOWN') for v in d['terminals'].values()), 'terminal values')
            need(d['terminals'] == {r: terminals.get(r, ('UNKNOWN',))[0] for r in REQUESTS}, 'FINAL terminal revival')
            need(d['errors'] == errors, 'FINAL errors')
            final = d
        else:
            need(owners is not None and d.get('owners') == owners, 'same owned context/entities')
            if event == 'CLOCK_SAMPLE':
                need(type(d['ros_ns']) is list and len(d['ros_ns']) == 2 and
                     all(integer(n) and n == 0 for n in d['ros_ns']), 'ROS clock not frozen')
                need(d['active'] == [True, True] and all(type(v) is bool for v in d['active']), 'ROS time inactive')
                _payload(d['use_sim_time'], [{'type': 1, 'value': True}] * 2)
                samples.append(seq)
            elif event == 'CLEANUP':
                need(cleanup is None and type(d['executor_shutdown']) is bool and
                     type(d['context_shutdown']) is bool and type(d['entities_released']) is bool, 'cleanup flags')
                need(type(d['released_futures']) is list and len(set(d['released_futures'])) == len(d['released_futures']) and
                     set(d['released_futures']) == future_ids, 'future cleanup')
                need(type(d['destroyed_nodes']) is list and len(set(d['destroyed_nodes'])) == len(d['destroyed_nodes']) and
                     set(d['destroyed_nodes']) <= {owners['server'], owners['client_node']}, 'node cleanup')
                cleanup = (d, seq)
            else:
                need(cleanup is None, 'RPC after cleanup')
                rid, fid = d['request_id'], d['future_id']
                need(type(rid) is str and rid in REQUESTS and type(fid) is str and fid.isdecimal(), 'request/future identity')
                if event == 'REQUEST_SUBMITTED':
                    need(rid == REQUESTS[len(submitted)] if len(submitted) < 3 else False, 'submission order/count')
                    need(fid not in future_ids, 'future alias')
                    need(len(submitted) - len(observed) < 2, 'pending bound')
                    need(samples and samples[-1] > max([0] + [v[1] for v in terminals.values()]), 'clock sample before submit')
                    need(integer(d['submit_ns']) and integer(d['deadline_ns']) and
                         prior_t <= d['submit_ns'] <= t and d['submit_ns'] >= boot['start_ns'] and
                         d['deadline_ns'] == d['submit_ns'] + DURATIONS[rid], 'request deadline')
                    if rid == 'LATE':
                        need(terminals.get('FAST', (None,))[0] == 'SUCCESS' and 'FAST' in synced, 'FAST before LATE')
                    if rid == 'RECOVERY':
                        need(terminals.get('LATE', (None,))[0] == 'TIMED_OUT' and 'LATE' in synced and 'LATE' not in observed, 'recovery after timeout while pending')
                    _payload(d['request'], wire_request(identity[1], rid))
                    submitted[rid] = (d, t, seq)
                    future_ids.add(fid)
                else:
                    need(rid in submitted and fid == submitted[rid][0]['future_id'], 'future correlation')
                    sub = submitted[rid][0]
                    if event == 'SERVER_REQUEST_RECEIVED':
                        need(rid not in received, 'duplicate server receive')
                        _payload(d['request'], sub['request'])
                        received[rid] = t
                    elif event == 'SERVER_RESPONSE_READY':
                        need(rid in received and rid not in ready, 'server response causality')
                        if rid == 'LATE':
                            need(t >= received[rid] + 2 * NS, 'server delay too short')
                        _payload(d['response'], wire_response(identity[1], rid))
                        ready[rid] = t
                    elif event == 'CLIENT_RESPONSE_OBSERVED':
                        need(rid in ready and rid not in observed, 'actual client response required')
                        _payload(d['response'], wire_response(identity[1], rid))
                        if rid == 'LATE':
                            need(terminals.get(rid, (None,))[0] == 'TIMED_OUT' and
                                 terminals.get('RECOVERY', (None,))[0] == 'SUCCESS' and 'RECOVERY' in synced and
                                 t <= sub['submit_ns'] + 5 * NS, 'late observation order/deadline')
                        else:
                            need(t <= sub['deadline_ns'], 'success observed too late')
                        observed[rid] = (t, seq)
                    elif event == 'REQUEST_TERMINAL':
                        need(rid not in terminals and type(d['future_pending']) is bool, 'terminal immutable')
                        if rid == 'LATE':
                            need(d['status'] == 'TIMED_OUT' and d['future_pending'] is True and
                                 rid in received and rid not in observed and
                                 sub['deadline_ns'] <= t <= sub['submit_ns'] + 3 * NS // 2, 'timeout window/pending')
                        else:
                            need(d['status'] == 'SUCCESS' and d['future_pending'] is False and
                                 rid in observed and t <= sub['deadline_ns'], 'success terminal')
                        terminals[rid] = (d['status'], seq)
                    elif event == 'TERMINAL_SYNC_COMPLETED':
                        need(rid in terminals and rid not in synced and
                             seq == terminals[rid][1] + 1, 'sync receipt order/uniqueness')
                        check_terminal_completion(rows[terminals[rid][1]], row,
                                                  sub['submit_ns'], sub['deadline_ns'], boot['work_end_ns'])
                        synced.add(rid)
                    elif event == 'LATE_RESPONSE_QUARANTINED':
                        need(rid == 'LATE' and rid in observed and not quarantined, 'late quarantine causality')
                        _payload(d['response'], wire_response(identity[1], rid))
                        quarantined = True
        if event not in ('CLEANUP', 'FINAL', 'ERROR'):
            need(t < boot['work_end_ns'], 'original work cutoff')
        need(t < boot['whole_end_ns'], 'original whole cutoff')
    if not require_complete:
        need(final is None or final['status'] != 'PASS', 'partial cannot claim PASS')
        return False
    need(final is not None and final['status'] == 'PASS' and not errors, 'missing/failed FINAL')
    need(len(submitted) == len(received) == len(ready) == len(observed) == len(terminals) == len(synced) == 3 and quarantined, 'incomplete phases')
    need(cleanup is not None and cleanup[0]['executor_shutdown'] and cleanup[0]['context_shutdown'] and
         cleanup[0]['entities_released'] and set(cleanup[0]['destroyed_nodes']) == {owners['server'], owners['client_node']}, 'cleanup incomplete')
    need(samples and samples[-1] > observed['LATE'][1] and samples[-1] < cleanup[1], 'final clock readback')
    return True


def read_jsonl(path):
    """Bounded UTF-8 parser; duplicate keys/nonfinite tokens fail closed."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite JSON: ' + value)
    with open(path, 'rb') as stream:
        raw = stream.read(LIMIT + 1)
    need(len(raw) <= LIMIT and raw.endswith(b'\n'), 'raw bound/final newline')
    lines = raw.decode('utf-8').splitlines()
    need(0 < len(lines) <= 256 and all(lines), 'line count/blank')
    rows = [json.loads(line, object_pairs_hook=unique, parse_constant=reject) for line in lines]
    _tree(rows)
    return rows


def run_frozen_clock_rpc_scenario(case_id, nonce, domain_id, evidence_path):
    """Four required arguments. The sole opt-in SDK boundary is _runtime."""
    need(identifier(case_id) and identifier(nonce), 'ASCII case/nonce')
    need(integer(domain_id) and 0 <= domain_id <= 232, 'domain 0..232')
    return _runtime(case_id, nonce, domain_id, evidence_path)


def _runtime(case_id, nonce, domain_id, evidence_path):
    start = time.monotonic_ns()
    work_end, whole_end = start + WORK_NS, start + WHOLE_NS
    stream = open(evidence_path, 'x', encoding='utf-8', newline='\n')
    lock = threading.RLock()
    rows, errors, nodes, pending = [], [], [], {}
    owners = {}
    terminals = dict.fromkeys(REQUESTS, 'UNKNOWN')
    context = executor = None
    initialized = False
    byte_count = 0
    callbacks_failed = []
    cleanup = dict(owners=owners, executor_shutdown=False, released_futures=[],
                   destroyed_nodes=[], context_shutdown=False, entities_released=False)

    def budget(end=work_end):
        remaining = end - time.monotonic_ns()
        if remaining <= 0:
            raise TimeoutError('original absolute cutoff')
        return remaining / NS

    def emit(event, data, durable=False):
        nonlocal byte_count
        with lock:
            row = dict(schema=SCHEMA, case=case_id, nonce=nonce, domain=domain_id,
                       seq=len(rows), mono_ns=time.monotonic_ns(), event=event, data=data)
            text = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
            size = len(text.encode('utf-8'))
            need(len(rows) < 256 and byte_count + size <= LIMIT, 'evidence budget')
            stream.write(text)
            stream.flush()
            # Retain the original written row even if fsync or its clock fails.
            rows.append(row)
            byte_count += size
            if event == 'REQUEST_TERMINAL':
                need(durable, 'terminal requires actual fsync')
                rid = data['request_id']
                terminals[rid] = data['status']
                timing = terminal_sync(stream, time.monotonic_ns, os.fsync)
                receipt = emit('TERMINAL_SYNC_COMPLETED', dict(
                    **{k: data[k] for k in LINK}, terminal_seq=row['seq'], **timing))
                item = pending[rid]
                check_terminal_completion(row, receipt, item['submit'], item['deadline'], work_end)
                budget()
            elif durable:
                os.fsync(stream.fileno())
            return row

    def error(exc):
        data = {'class': type(exc).__name__[:2048], 'message': str(exc)[:2048]}
        with lock:
            errors.append(data)
            emit('ERROR', data)

    def link(rid):
        return dict(request_id=rid, future_id=str(id(pending[rid]['future'])), owners=dict(owners))

    def clocks():
        budget()
        data = dict(owners=dict(owners), ros_ns=[n.get_clock().now().nanoseconds for n in nodes],
                    active=[n.get_clock().ros_time_is_active for n in nodes],
                    use_sim_time=[{'type': n.get_parameter('use_sim_time').type_.value,
                                   'value': n.get_parameter('use_sim_time').value} for n in nodes])
        emit('CLOCK_SAMPLE', data)
        need(data['ros_ns'] == [0, 0] and data['active'] == [True, True], 'ROS clock not frozen/active')
        _payload(data['use_sim_time'], [{'type': 1, 'value': True}] * 2)

    def project(response):
        return {'values': [{'type': v.type, 'string_value': v.string_value} for v in response.values]}

    try:
        emit('BOOT', dict(start_ns=start, work_end_ns=work_end, whole_end_ns=whole_end))
        # Lazy SDK imports; no import-time context, I/O, threads, or SDK access.
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rcl_interfaces.srv import GetParameters
        from rcl_interfaces.msg import ParameterValue

        budget()
        context = Context()
        context.init(args=[], initialize_logging=False, domain_id=domain_id)
        initialized = True
        options = dict(context=context, cli_args=[], use_global_arguments=False,
                       enable_rosout=False, enable_logger_service=False,
                       start_parameter_services=False,
                       automatically_declare_parameters_from_overrides=False,
                       parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True),
                                            Parameter('start_type_description_service', Parameter.Type.BOOL, False)])
        server = Node('frozen_server_' + nonce, **options)
        nodes.append(server)
        client_node = Node('frozen_client_' + nonce, **options)
        nodes.append(client_node)
        executor = MultiThreadedExecutor(num_threads=2, context=context)
        for node in nodes:
            need(executor.add_node(node), 'executor refused node')
        service_name = '/frozen_rpc_' + case_id + '_' + nonce
        group = ReentrantCallbackGroup()

        def callback(request, response):
            try:
                budget()
                actual = {'names': list(request.names)}
                with lock:
                    matches = [r for r in REQUESTS if actual == wire_request(nonce, r)]
                    need(len(matches) == 1 and matches[0] in pending, 'unexpected server request')
                    rid = matches[0]
                    event = emit('SERVER_REQUEST_RECEIVED', dict(**link(rid), request=actual))
                if rid == 'LATE':
                    target = event['mono_ns'] + 2 * NS
                    while True:
                        remaining = target - time.monotonic_ns()
                        if remaining <= 0:
                            break
                        time.sleep(min(0.01, budget(), remaining / NS))
                budget()
                response.values = [ParameterValue(type=4, string_value=actual['names'][0])]
                emit('SERVER_RESPONSE_READY', dict(**link(rid), response=project(response)))
                return response
            except Exception as exc:
                with lock:
                    callbacks_failed.append(exc)
                raise

        service = server.create_service(GetParameters, service_name, callback, callback_group=group)
        client = client_node.create_client(GetParameters, service_name)
        owners.update(context=str(id(context)), server=str(id(server)), client_node=str(id(client_node)),
                      executor=str(id(executor)), client=str(id(client)))
        entities = {}
        for label, node in zip(('server', 'client_node'), nodes):
            entities[label] = {}
            for category in CATEGORIES:
                items = list(getattr(node, category))
                entities[label][category] = [dict(id=str(id(item)), name=(
                    item.topic_name if category in ('publishers', 'subscriptions') else
                    item.service_name if category in ('services', 'clients') else '')) for item in items]
        emit('ENTITY_INVENTORY', dict(owners=dict(owners), service=client.service_name, entities=entities))
        clocks()
        need(client.wait_for_service(timeout_sec=min(5.0, budget())), 'diagnostic discovery timeout')
        budget()

        def submit(rid):
            clocks()
            request = GetParameters.Request()
            request.names = wire_request(nonce, rid)['names']
            with lock:
                budget()
                submitted = time.monotonic_ns()
                future = client.call_async(request)
                pending[rid] = dict(future=future, submit=submitted, deadline=submitted + DURATIONS[rid], observed=False)
                emit('REQUEST_SUBMITTED', dict(**link(rid), submit_ns=submitted,
                     deadline_ns=submitted + DURATIONS[rid], request={'names': list(request.names)}))

        def spin():
            budget()
            with lock:
                if callbacks_failed:
                    raise callbacks_failed[0]
            executor.spin_once(timeout_sec=min(0.01, budget()))
            budget()

        def observe(rid):
            item = pending[rid]
            if item['observed'] or not item['future'].done():
                return False
            budget()
            need(not item['future'].cancelled(), 'unexpected cancelled SDK future')
            response = project(item['future'].result())
            row = emit('CLIENT_RESPONSE_OBSERVED', dict(**link(rid), response=response))
            _payload(response, wire_response(nonce, rid))
            item['observed'] = True
            if rid == 'LATE':
                need(terminals[rid] == 'TIMED_OUT' and terminals['RECOVERY'] == 'SUCCESS' and
                     row['mono_ns'] <= item['submit'] + 5 * NS, 'late order/deadline')
                emit('LATE_RESPONSE_QUARANTINED', dict(**link(rid), response=response))
            else:
                need(row['mono_ns'] <= item['deadline'], 'response missed deadline')
                terminal = emit('REQUEST_TERMINAL', dict(**link(rid), status='SUCCESS', future_pending=False), True)
                need(terminal['mono_ns'] <= item['deadline'], 'terminal missed deadline')
                terminals[rid] = 'SUCCESS'
            return True

        submit('FAST')
        while not observe('FAST'):
            need(time.monotonic_ns() <= pending['FAST']['deadline'], 'FAST timeout')
            spin()
        submit('LATE')
        next_sample = time.monotonic_ns()
        while True:
            budget()
            need(not pending['LATE']['future'].done(), 'LATE completed before timeout observation')
            if time.monotonic_ns() >= pending['LATE']['deadline']:
                with lock:
                    need(any(r['event'] == 'SERVER_REQUEST_RECEIVED' and r['data']['request_id'] == 'LATE' for r in rows), 'LATE not received')
                    need(not pending['LATE']['future'].done(), 'LATE not pending')
                    row = emit('REQUEST_TERMINAL', dict(**link('LATE'), status='TIMED_OUT', future_pending=True), True)
                    need(row['mono_ns'] <= pending['LATE']['submit'] + 3 * NS // 2, 'late timeout observation')
                    terminals['LATE'] = 'TIMED_OUT'
                break
            if time.monotonic_ns() >= next_sample:
                clocks()
                next_sample = time.monotonic_ns() + NS // 10
            spin()
        submit('RECOVERY')
        while not observe('RECOVERY'):
            need(not pending['LATE']['future'].done(), 'LATE client completed before recovery')
            need(time.monotonic_ns() <= pending['RECOVERY']['deadline'], 'RECOVERY timeout')
            spin()
        while not observe('LATE'):
            need(time.monotonic_ns() <= pending['LATE']['submit'] + 5 * NS, 'late client observation timeout')
            if time.monotonic_ns() >= next_sample:
                clocks()
                next_sample = time.monotonic_ns() + NS // 10
            spin()
        clocks()
        budget()
    except Exception as exc:
        error(exc)
    finally:
        # Worker termination precedes node destruction. A failed shutdown does
        # not authorize unsafe destruction underneath a live callback.
        if executor is not None:
            try:
                cleanup['executor_shutdown'] = executor.shutdown(timeout_sec=max(0.0, min(30.0, (whole_end - time.monotonic_ns()) / NS))) is True
                need(cleanup['executor_shutdown'], 'executor shutdown incomplete')
            except Exception as exc:
                error(exc)
        safe = executor is None or cleanup['executor_shutdown']
        if safe:
            for rid, item in pending.items():
                try:
                    future = item['future']
                    if not future.done():
                        future.cancel()
                    need(future.done() or future.cancelled(), 'future release incomplete')
                    cleanup['released_futures'].append(str(id(future)))
                except Exception as exc:
                    error(exc)
            for node in reversed(nodes):
                try:
                    budget(whole_end)
                    node.destroy_node()
                    cleanup['destroyed_nodes'].append(str(id(node)))
                except Exception as exc:
                    error(exc)
            cleanup['entities_released'] = len(cleanup['destroyed_nodes']) == 2
            if initialized:
                try:
                    budget(whole_end)
                    context.shutdown()
                    cleanup['context_shutdown'] = True
                except Exception as exc:
                    error(exc)
        try:
            budget(whole_end)
        except Exception as exc:
            error(exc)
        emit('CLEANUP', {**cleanup, 'owners': dict(owners)})
        final = dict(status='FAIL', terminals=dict(terminals), errors=list(errors))
        if not errors:
            candidate = dict(schema=SCHEMA, case=case_id, nonce=nonce, domain=domain_id,
                             seq=len(rows), mono_ns=time.monotonic_ns(), event='FINAL',
                             data=dict(status='PASS', terminals=dict(terminals), errors=[]))
            try:
                validate_transcript(rows + [candidate])
                # Read back the actual persisted prefix before allowing PASS.
                need(read_jsonl(evidence_path) == rows, 'persisted prefix mismatch')
                budget(whole_end)
                final['status'] = 'PASS'
            except Exception as exc:
                error(exc)
                final['errors'] = list(errors)
        try:
            emit('FINAL', final, True)
        finally:
            stream.close()
    # Persistence/close failures propagate; an overrun cannot return success.
    budget(whole_end)
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--domain-id', required=True, type=int)
    parser.add_argument('--evidence', required=True)
    args = parser.parse_args()
    result = run_frozen_clock_rpc_scenario(args.case, args.nonce, args.domain_id, args.evidence)
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
