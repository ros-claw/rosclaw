"""Parameter Atomic12 source-owned protocol and oracle; import is stdlib-only.

Runtime is deliberately opt-in. Pure validation is not evidence of ROS execution.
"""
import argparse
import json
import os
import re
import time

SCHEMA = 'ROSClaw.ParameterAtomic12.Event.v1'
CASES = tuple('P' + str(i) for i in range(1, 7))
NAMES = ['target', 'limit', 'robot_id']
COUNTS = {'publishers': 2, 'subscriptions': 0, 'services': 6, 'clients': 3}
DATA_KEYS = {
    'BOOT': {'scope'},
    'ENTITY_INVENTORY': {*COUNTS, 'names'},
    'REQUEST_SUBMITTED': {'request_id', 'case_id', 'service', 'future_id', 'request'},
    'RESPONSE_RESOLVED': {'request_id', 'case_id', 'service', 'future_id', 'response'},
    'PARAMETER_VALIDATION': {'case_id', 'proposed', 'current_before', 'candidate', 'successful', 'reason'},
    'ABSENCE_PROBE': {'case_id', 'service', 'timeout_s', 'available', 'requests_sent'},
    'CASE_RESULT': {'case_id', 'status', 'reason'},
    'FINAL': {'status', 'condition_statuses', 'counters', 'cleanup', 'errors'},
}
COMMON_KEYS = {'schema', 'case', 'nonce', 'domain', 'seq', 'event', 'data'}
COUNTER_KEYS = {'service_requests', 'get_requests', 'atomic_requests',
                'missing_service_requests', 'parameter_event_publish_calls',
                'parameter_event_publish_observation'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int


def typed(value, code=2):
    return {'type': code, 'value': value}


def parameter(name, value, code=2):
    return {'name': name, **typed(value, code)}


def check_value(value, named=False):
    need(type(value) is dict and set(value) == ({'name', 'type', 'value'} if named else {'type', 'value'}), 'parameter shape')
    need(integer(value['type']) and value['type'] in (2, 4), 'parameter type code')
    need((value['type'] == 2 and integer(value['value'])) or
         (value['type'] == 4 and type(value['value']) is str), 'parameter value type')
    if named:
        need(type(value['name']) is str, 'parameter name')


def check_state(state):
    need(type(state) is dict and set(state) == set(NAMES), 'state names')
    for value in state.values():
        check_value(value)


def candidate_validation(current, proposed):
    """Pure custom semantic callback policy. Never changes either input.

    Descriptor enforcement belongs to the SDK, not this callback. This helper
    rejects invalid types defensively, but does not emulate a descriptor gate.
    Returns (typed candidate, successful bool, reason str).
    """
    check_state(current)
    need(type(proposed) is list, 'proposed list')
    candidate = {name: dict(value) for name, value in current.items()}
    seen = set()
    for item in proposed:
        check_value(item, named=True)
        need(item['name'] in candidate and item['name'] not in seen, 'unknown/duplicate proposed name')
        seen.add(item['name'])
        candidate[item['name']] = {'type': item['type'], 'value': item['value']}
    valid = (all(v['type'] == 2 for v in candidate.values()) and
             0 <= candidate['target']['value'] <= candidate['limit']['value'] <= 100)
    return candidate, bool(valid), '' if valid else 'candidate requires 0 <= target <= limit <= 100 and INTEGER values'


def request_plan():
    """Ordered public requests, not synthetic service replies."""
    plan = [('P1_GET_FORWARD', 'P1', 'get_parameters', {'names': list(NAMES)}),
            ('P1_GET_REVERSE', 'P1', 'get_parameters', {'names': list(reversed(NAMES))})]
    updates = {
        'P2': [parameter('target', 12), parameter('limit', 30)],
        'P3': [parameter('target', 50), parameter('limit', 30)],
        'P4': [parameter('robot_id', 43), parameter('target', 13)],
        'P5': [parameter('target', '12', 4), parameter('limit', 31)],
        'P6': [parameter('target', 13), parameter('limit', 31)],
    }
    for case, values in updates.items():
        plan.extend([(case + '_SET', case, 'set_parameters_atomically', {'parameters': values}),
                     (case + '_GET', case, 'get_parameters', {'names': list(NAMES)})])
    return plan


def _check_response(service, response):
    need(type(response) is dict, 'response object')
    if service == 'get_parameters':
        need(set(response) == {'values'} and type(response['values']) is list, 'get response shape')
        for value in response['values']:
            check_value(value)
    else:
        need(set(response) == {'successful', 'reason'}, 'atomic response shape')
        need(type(response['successful']) is bool and type(response['reason']) is str and len(response['reason']) <= 2048, 'atomic result types')
        need(response['successful'] or bool(response['reason']), 'rejection reason empty')


def validate_case(rows, case):
    """Require actual projected evidence for one case; no local readback substitute."""
    need(case in CASES, 'case id')
    local = [r['data'] for r in rows if r['event'] == 'RESPONSE_RESOLVED' and r['data']['case_id'] == case]
    expected = [p for p in request_plan() if p[1] == case]
    need(len(local) == len(expected), 'case response count')
    by_id = {d['request_id']: d for d in local}
    need(len(by_id) == len(local), 'duplicate response')
    for rid, _, service, request in expected:
        need(rid in by_id, 'missing case response')
        response = by_id[rid]['response']
        _check_response(service, response)
        if service == 'get_parameters':
            values = [10, 20, 42] if case == 'P1' else ([13, 31, 42] if case == 'P6' else [12, 30, 42])
            if rid == 'P1_GET_REVERSE':
                values.reverse()
            need(response['values'] == [typed(v) for v in values], 'ordered actual get mismatch')
        else:
            need(response['successful'] is (case in ('P2', 'P6')), 'atomic success mismatch')
    callbacks = [r['data'] for r in rows if r['event'] == 'PARAMETER_VALIDATION' and r['data']['case_id'] == case]
    if case in ('P2', 'P3', 'P6'):
        need(len(callbacks) == 1, 'custom callback missing/duplicate')
    else:
        need(not callbacks, 'unexpected custom callback after descriptor gate or during get')
    for cb in callbacks:
        before = dict(zip(NAMES, [typed(v) for v in ([10, 20, 42] if case == 'P2' else [12, 30, 42])]))
        update = expected[0][3]['parameters']
        candidate, ok, reason = candidate_validation(before, update)
        need(cb['proposed'] == update and cb['current_before'] == before and cb['candidate'] == candidate, 'callback paired candidate mismatch')
        need(type(cb['successful']) is bool and cb['successful'] is ok, 'callback decision mismatch')
        need(type(cb['reason']) is str and len(cb['reason']) <= 2048 and (ok or bool(cb['reason'])), 'callback reason')
    probes = [r['data'] for r in rows if r['event'] == 'ABSENCE_PROBE' and r['data']['case_id'] == case]
    need(len(probes) == (1 if case == 'P6' else 0), 'probe count')
    if probes:
        p = probes[0]
        need(type(p['timeout_s']) in (int, float) and p['timeout_s'] == 2 and
             p['available'] is False and integer(p['requests_sent']) and p['requests_sent'] == 0 and
             type(p['service']) is str and bool(p['service']), 'missing probe mismatch')
    return True


def validate_transcript(rows, require_complete=True):
    """Strict successful transcript oracle, or typed prefix check for partial failures.

    Partial validation returns False, never PASS. All matching uses local future
    identities; arrival order of the two P1 replies is deliberately unrestricted.
    """
    need(type(rows) is list and 0 < len(rows) <= 256, 'row bound')
    encoded = json.dumps(rows, allow_nan=False, ensure_ascii=False).encode('utf-8')
    need(len(encoded) + len(rows) <= 8388608, 'byte bound')
    identity = None
    submitted, resolved, future_ids = {}, set(), set()
    plan = request_plan()
    completed = []
    inventory = 0
    probes = 0
    final = None
    active = None
    pending_set = None
    for seq, row in enumerate(rows):
        need(type(row) is dict and set(row) == COMMON_KEYS, 'common exact keys')
        need(row['schema'] == SCHEMA and integer(row['seq']) and row['seq'] == seq, 'schema/sequence')
        need(type(row['case']) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', row['case']) and
             type(row['nonce']) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', row['nonce']), 'ASCII identity')
        need(integer(row['domain']) and 0 <= row['domain'] <= 232, 'domain')
        key = (row['case'], row['nonce'], row['domain'])
        identity = identity or key
        need(key == identity, 'identity changed')
        event, d = row['event'], row['data']
        need(type(event) is str and event in DATA_KEYS and type(d) is dict and set(d) == DATA_KEYS[event], 'event/data keys')
        need(final is None, 'row after FINAL')
        if seq == 0:
            need(event == 'BOOT', 'BOOT first')
        if event == 'BOOT':
            need(seq == 0 and d['scope'] == 'PARAMETER_ATOMIC12_V1', 'BOOT scope/duplicate')
        elif event == 'ENTITY_INVENTORY':
            need(seq == 1 and inventory == 0, 'inventory order')
            for category, count in COUNTS.items():
                need(integer(d[category]) and d[category] == count, 'logical inventory count')
            need(type(d['names']) is dict and set(d['names']) == set(COUNTS), 'inventory name categories')
            for category, count in COUNTS.items():
                names = d['names'][category]
                need(type(names) is list and len(names) == count and all(type(n) is str and n.startswith('/') for n in names), 'resolved entity names')
            inventory += 1
        elif event == 'REQUEST_SUBMITTED':
            need(inventory == 1 and len(submitted) < len(plan), 'request outside inventory/plan')
            rid, case, service, request = plan[len(submitted)]
            need(d['request_id'] == rid and d['case_id'] == case and d['service'] == service and d['request'] == request, 'submission order/projection')
            # Equality alone accepts bool == int: validate the nested types too.
            if service == 'get_parameters':
                need(set(d['request']) == {'names'} and type(d['request']['names']) is list, 'get request')
            else:
                need(set(d['request']) == {'parameters'} and type(d['request']['parameters']) is list, 'atomic request')
                for p in d['request']['parameters']:
                    check_value(p, named=True)
            need(type(d['future_id']) is str and bool(d['future_id']) and d['future_id'] not in future_ids, 'distinct future identity')
            need(len(submitted) - len(resolved) < 2, 'outstanding bound')
            if case != 'P1':
                need(all(p[0] in resolved for p in plan[:len(submitted)]), 'sequential transaction dependency')
                need(completed == list(CASES[:CASES.index(case)]), 'case boundary order')
            if case == 'P6':
                need(probes == 1, 'probe before healthy request')
            submitted[rid] = d
            future_ids.add(d['future_id'])
            active = case
            if service == 'set_parameters_atomically':
                pending_set = rid
        elif event == 'RESPONSE_RESOLVED':
            rid = d['request_id']
            need(type(rid) is str and rid in submitted and rid not in resolved, 'unknown/duplicate resolution')
            sub = submitted[rid]
            need(all(d[k] == sub[k] for k in ('case_id', 'service', 'future_id')), 'resolution identity swapped')
            if d['case_id'] == 'P1':
                need(len(submitted) >= 2, 'P1 must submit both before resolving')
            _check_response(d['service'], d['response'])
            resolved.add(rid)
            if rid == pending_set:
                pending_set = None
        elif event == 'PARAMETER_VALIDATION':
            need(d['case_id'] == active and pending_set is not None, 'callback outside pending atomic')
            need(type(d['proposed']) is list, 'callback proposed type')
            for p in d['proposed']:
                check_value(p, named=True)
            check_state(d['current_before'])
            check_state(d['candidate'])
            need(type(d['successful']) is bool and type(d['reason']) is str and len(d['reason']) <= 2048, 'callback result types')
        elif event == 'ABSENCE_PROBE':
            need(completed == list(CASES[:5]) and probes == 0 and d['case_id'] == 'P6', 'probe phase')
            need(d['available'] is False and integer(d['requests_sent']) and d['requests_sent'] == 0 and
                 type(d['timeout_s']) in (int, float) and d['timeout_s'] == 2 and type(d['service']) is str, 'probe fields')
            probes += 1
        elif event == 'CASE_RESULT':
            need(d['case_id'] in CASES and d['case_id'] not in completed and
                 d['case_id'] == CASES[len(completed)], 'case result order/duplicate')
            need(d['status'] in ('PASS', 'FAIL', 'UNKNOWN') and type(d['reason']) is str and len(d['reason']) <= 2048, 'case result types')
            if d['status'] == 'PASS':
                validate_case(rows[:seq], d['case_id'])
            if require_complete:
                need(d['status'] == 'PASS', 'case not PASS')
            completed.append(d['case_id'])
        elif event == 'FINAL':
            final = d
            need(d['status'] in ('PASS', 'FAIL', 'UNKNOWN'), 'final status')
            need(type(d['condition_statuses']) is dict and set(d['condition_statuses']) == set(CASES) and
                 all(v in ('PASS', 'FAIL', 'UNKNOWN') for v in d['condition_statuses'].values()), 'condition statuses')
            counters = d['counters']
            need(type(counters) is dict and set(counters) == COUNTER_KEYS, 'counter keys')
            actual = (len(submitted), sum(s['service'] == 'get_parameters' for s in submitted.values()),
                      sum(s['service'] == 'set_parameters_atomically' for s in submitted.values()), 0)
            for name, value in zip(('service_requests', 'get_requests', 'atomic_requests', 'missing_service_requests'), actual):
                need(integer(counters[name]) and counters[name] == value, 'actual counter mismatch')
            need(counters['parameter_event_publish_calls'] is None and counters['parameter_event_publish_observation'] == 'UNKNOWN_NOT_INSTRUMENTED', 'unobserved publish count')
            cleanup = d['cleanup']
            need(type(cleanup) is dict and set(cleanup) == {'executor_shutdown', 'nodes_destroyed', 'context_shutdown'}, 'cleanup keys')
            need(type(cleanup['executor_shutdown']) is bool and type(cleanup['context_shutdown']) is bool and
                 integer(cleanup['nodes_destroyed']) and 0 <= cleanup['nodes_destroyed'] <= 2, 'cleanup types')
            need(type(d['errors']) is list, 'errors list')
            for error in d['errors']:
                need(type(error) is dict and set(error) == {'class', 'message'} and all(type(v) is str and len(v) <= 2048 for v in error.values()), 'error shape')
    if not require_complete:
        need(final is None or final['status'] != 'PASS', 'partial cannot certify PASS')
        return False
    need(final is not None and final['status'] == 'PASS' and not final['errors'], 'missing/failed FINAL')
    need(inventory == 1 and len(submitted) == len(resolved) == 12 and completed == list(CASES), 'incomplete transcript')
    need(final['condition_statuses'] == dict.fromkeys(CASES, 'PASS'), 'final conditions')
    need(final['cleanup'] == {'executor_shutdown': True, 'nodes_destroyed': 2, 'context_shutdown': True}, 'cleanup incomplete')
    return True


def read_jsonl(path):
    """Bounded JSONL reader rejects duplicate object keys and nonfinite numbers."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite JSON: ' + value)
    with open(path, 'rb') as stream:
        raw = stream.read(8388609)
    need(len(raw) <= 8388608, 'raw bytes bound')
    lines = raw.decode('utf-8').splitlines()
    need(0 < len(lines) <= 256 and all(lines), 'raw rows bound/blank')
    return [json.loads(line, object_pairs_hook=unique, parse_constant=nonfinite) for line in lines]


def run_parameter_atomic_scenario(case_id, nonce, domain_id, evidence_path):
    """Real SDK entry. Four required args; never invoked by pure tests."""
    need(type(case_id) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', case_id), 'case identifier')
    need(type(nonce) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', nonce), 'nonce identifier')
    need(integer(domain_id) and 0 <= domain_id <= 232, 'domain integer 0..232')
    return _runtime(case_id, nonce, domain_id, evidence_path)


def _runtime(case_id, nonce, domain_id, evidence_path):
    start = time.monotonic()
    work_end, whole_end = start + 120, start + 180
    rows, errors, nodes, futures = [], [], [], []
    statuses = dict.fromkeys(CASES, 'UNKNOWN')
    cleanup = {'executor_shutdown': False, 'nodes_destroyed': 0, 'context_shutdown': False}
    counters = {'service_requests': 0, 'get_requests': 0, 'atomic_requests': 0,
                'missing_service_requests': 0, 'parameter_event_publish_calls': None,
                'parameter_event_publish_observation': 'UNKNOWN_NOT_INSTRUMENTED'}
    context = executor = None
    context_initialized = False
    active = None
    # Exclusive creation avoids overwriting previous evidence. This failure cannot
    # yield a durable FINAL and propagates to the process as a nonzero exit.
    stream = open(evidence_path, 'x', encoding='utf-8', newline='\n')
    byte_count = 0

    def emit(event, data):
        nonlocal byte_count
        need(len(rows) < 256, 'row budget exhausted')
        row = dict(schema=SCHEMA, case=case_id, nonce=nonce, domain=domain_id,
                   seq=len(rows), event=event, data=data)
        text = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
        size = len(text.encode('utf-8'))
        need(byte_count + size <= 8388608, 'byte budget exhausted')
        stream.write(text)
        stream.flush()
        rows.append(row)
        byte_count += size

    def record_error(exc):
        errors.append({'class': type(exc).__name__[:2048], 'message': str(exc)[:2048]})

    def budget(limit=work_end):
        remaining = limit - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('absolute runtime deadline')
        return remaining

    emit('BOOT', {'scope': 'PARAMETER_ATOMIC12_V1'})
    try:
        # All SDK imports are lazy, exclusively on the real runtime path.
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
        from rcl_interfaces.srv import GetParameters, SetParametersAtomically

        budget()
        context = Context()
        context.init(args=[], initialize_logging=False, domain_id=domain_id)
        context_initialized = True
        options = dict(context=context, cli_args=[], use_global_arguments=False,
                       enable_rosout=False, enable_logger_service=False,
                       automatically_declare_parameters_from_overrides=False,
                       parameter_overrides=[Parameter('start_type_description_service', Parameter.Type.BOOL, False),
                                            Parameter('use_sim_time', Parameter.Type.BOOL, False)])
        suffix = nonce[:40]
        server = Node('pa12_server_' + suffix, start_parameter_services=True, **options)
        nodes.append(server)
        client_node = Node('pa12_client_' + suffix, start_parameter_services=False, **options)
        nodes.append(client_node)
        executor = SingleThreadedExecutor(context=context)
        for node in nodes:
            need(executor.add_node(node), 'executor refused node')
        for name, value in zip(NAMES, (10, 20, 42)):
            server.declare_parameter(name, value, ParameterDescriptor(
                name=name, type=2, read_only=(name == 'robot_id'), dynamic_typing=False))

        def sdk_parameter(p):
            return parameter(p.name, p.value, parameter_type_code(p.type_))

        def validate_update(params):
            # No spin, service wait, or mutation inside this callback. Snapshot
            # reflects SDK current state; success/rollback proof is actual Get.
            before = {p.name: typed(p.value, parameter_type_code(p.type_)) for p in server.get_parameters(NAMES)}
            proposed = [sdk_parameter(p) for p in params]
            candidate, ok, reason = candidate_validation(before, proposed)
            emit('PARAMETER_VALIDATION', dict(case_id=active, proposed=proposed,
                 current_before=before, candidate=candidate, successful=ok, reason=reason))
            return SetParametersResult(successful=ok, reason=reason)

        server.add_on_set_parameters_callback(validate_update)
        base = '/' + server.get_name()
        getter = client_node.create_client(GetParameters, base + '/get_parameters')
        setter = client_node.create_client(SetParametersAtomically, base + '/set_parameters_atomically')
        missing = client_node.create_client(GetParameters, '/pa12_never_created_' + nonce + '/get_parameters')
        names = {k: [] for k in COUNTS}
        for node in nodes:
            names['publishers'].extend(p.topic_name for p in node.publishers)
            names['subscriptions'].extend(s.topic_name for s in node.subscriptions)
            names['services'].extend(s.service_name for s in node.services)
            names['clients'].extend(c.service_name for c in node.clients)
        observed_counts = {k: len(v) for k, v in names.items()}
        emit('ENTITY_INVENTORY', {**observed_counts, 'names': names})
        need(observed_counts == COUNTS, 'unexpected logical entities')
        for client in (getter, setter):
            need(client.wait_for_service(timeout_sec=min(5.0, budget())), 'known service startup timeout')
            budget()

        def submit(item):
            rid, case, service, projection = item
            budget()
            if service == 'get_parameters':
                request = GetParameters.Request()
                request.names = projection['names']
                client = getter
            else:
                request = SetParametersAtomically.Request()
                request.parameters = [Parameter(p['name'], Parameter.Type(p['type']), p['value']).to_parameter_msg()
                                      for p in projection['parameters']]
                client = setter
            future = client.call_async(request)
            # Retain all futures until shutdown: id() reuse cannot alias another request.
            futures.append(future)
            counters['service_requests'] += 1
            counters['get_requests' if service == 'get_parameters' else 'atomic_requests'] += 1
            emit('REQUEST_SUBMITTED', dict(request_id=rid, case_id=case, service=service,
                 future_id=str(id(future)), request=projection))
            return (item, future, min(work_end, time.monotonic() + 5.0))

        def project_value(value):
            code = int(value.type)
            need(code in (2, 4), 'unexpected actual ParameterValue type')
            return typed(value.integer_value if code == 2 else value.string_value, code)

        def wait(pending):
            remaining = list(pending)
            while remaining:
                budget()
                # Bound observation, not an invented SDK completion timestamp.
                # A result first noticed after its deadline is a timeout even if
                # the underlying future has already transitioned to FINISHED.
                for entry in remaining:
                    budget(entry[2])
                ready = [entry for entry in remaining if entry[1].done()]
                for entry in ready:
                    item, future, deadline = entry
                    rid, case, service, _ = item
                    response = future.result()  # propagates genuine errors, never PASS
                    if service == 'get_parameters':
                        projection = {'values': [project_value(v) for v in response.values]}
                    else:
                        projection = {'successful': response.result.successful, 'reason': response.result.reason}
                    _check_response(service, projection)
                    emit('RESPONSE_RESOLVED', dict(request_id=rid, case_id=case, service=service,
                         future_id=str(id(future)), response=projection))
                    remaining.remove(entry)
                if remaining:
                    nearest = min(entry[2] for entry in remaining)
                    executor.spin_once(timeout_sec=min(0.05, budget(nearest), budget()))

        plan = request_plan()
        for case in CASES:
            active = case
            if case == 'P6':
                budget()
                need(budget() >= 2.0, 'insufficient budget for missing probe')
                available = missing.wait_for_service(timeout_sec=2.0)
                emit('ABSENCE_PROBE', dict(case_id=case, service=missing.service_name,
                     timeout_s=2, available=available, requests_sent=0))
                need(available is False, 'unexpected missing service exists')
                budget()
            items = [p for p in plan if p[1] == case]
            if case == 'P1':
                # Crucially, both call_async calls occur before either wait.
                pending = [submit(item) for item in items]
                wait(pending)
            else:
                for item in items:
                    wait([submit(item)])
            validate_case(rows, case)
            statuses[case] = 'PASS'
            emit('CASE_RESULT', dict(case_id=case, status='PASS', reason='actual projected replies and callback policy verified'))
        budget()
    except Exception as exc:
        record_error(exc)
        if active is not None and statuses[active] != 'PASS':
            statuses[active] = 'FAIL'
            emit('CASE_RESULT', dict(case_id=active, status='FAIL', reason=str(exc)[:2048]))
    finally:
        # Cleanup is attempted independently: one failure cannot skip later releases.
        if executor is not None:
            try:
                cleanup['executor_shutdown'] = executor.shutdown(timeout_sec=max(0.0, min(60.0, whole_end - time.monotonic()))) is True
            except Exception as exc:
                record_error(exc)
        for node in reversed(nodes):
            try:
                node.destroy_node()
                cleanup['nodes_destroyed'] += 1
            except Exception as exc:
                record_error(exc)
        if context is not None and context_initialized:
            try:
                context.shutdown()
                cleanup['context_shutdown'] = True
            except Exception as exc:
                record_error(exc)
        if time.monotonic() > whole_end:
            record_error(TimeoutError('whole runtime exceeded; external supervisor required'))
        final = dict(status='FAIL', condition_statuses=dict(statuses), counters=dict(counters),
                     cleanup=dict(cleanup), errors=errors)
        if not errors and all(v == 'PASS' for v in statuses.values()):
            final['status'] = 'PASS'
            candidate_row = dict(schema=SCHEMA, case=case_id, nonce=nonce, domain=domain_id,
                                 seq=len(rows), event='FINAL', data=final)
            try:
                validate_transcript(rows + [candidate_row])
            except Exception as exc:
                record_error(exc)
                final['status'] = 'FAIL'
        try:
            emit('FINAL', final)
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            stream.close()
    # Return only after durable FINAL; fsync/close failures propagate, never exit 0.
    return final


def parameter_type_code(parameter_type):
    """Project the SDK's plain Enum through its official integer value.

    Do not coerce malformed values (including bool) into wire type codes.
    Missing value attributes fail closed with AttributeError.
    """
    code = parameter_type.value
    need(integer(code), 'parameter enum value must be a non-bool integer')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--domain-id', required=True, type=int)
    parser.add_argument('--evidence', required=True)
    args = parser.parse_args()
    result = run_parameter_atomic_scenario(args.case, args.nonce, args.domain_id, args.evidence)
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
