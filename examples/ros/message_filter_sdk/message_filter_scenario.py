"""Bounded generated local SDK experiment. Importing this module runs no SDK.

Only run_local_message_filter_scenario (or the guarded CLI) executes the SDK.
Metadata validation is NOT independent proof that SDK execution occurred.
"""
import argparse
import base64
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import signal
import sys
import threading
import time

SCHEMA = 'LOCAL_MESSAGE_FILTER_EVIDENCE_V1'  # Legacy metadata-only compatibility.
SEMANTIC_SCHEMA = 'LOCAL_MESSAGE_FILTER_EVIDENCE_V2'
EXPECTED = {
    'ordinary': [('O_L', 'O_R')], 'strict_boundary': [],
    'queue_eviction': [('Q_L3', 'Q_R3')],
    'out_of_order': [('A_L1', 'A_R1'), ('A_L2', 'A_R2')],
    'duplicate_stamp': [('D_L_NEW', 'D_R')],
    'epoch_isolation': [('E2_L', 'E2_R')],
}
# Public acceptance outcomes, not a matching implementation.
TRIGGERS = {'O_R', 'Q_R3', 'A_R1', 'A_R2', 'D_R', 'E2_L'}
MAX_BYTES = 1048576


class ScenarioError(Exception):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def require(condition, detail, code='INVALID_METADATA'):
    if not condition:
        raise ScenarioError(code, detail)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(doc):
    data = (json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    require(len(data) <= MAX_BYTES, 'raw evidence exceeds 1 MiB', 'RESOURCE_LIMIT')
    require(len(doc['rows']) <= 256, 'raw rows exceed 256', 'RESOURCE_LIMIT')
    return data


def validate_inputs(doc):
    require(doc['schema'] == 'PROPOSED_LOCAL_MESSAGE_FILTER_INPUTS_V1', 'input schema')
    require(doc['data_origin'] == 'GENERATED_NOT_SENSOR', 'origin')
    require(type(doc['queue_size']) is int and doc['queue_size'] == 2, 'queue size')
    require(type(doc['slop_ns']) is int and doc['slop_ns'] == 10000000, 'slop')
    require(doc['allow_headerless'] is False, 'headerless')
    require([c['id'] for c in doc['cases']] == list(EXPECTED), 'six case order')
    ids = set()
    for case in doc['cases']:
        epoch, resets = 1, 0
        require(0 < len(case['events']) <= 16, 'event bound')
        for event in case['events']:
            if event.get('kind') == 'RESET':
                require(case['id'] == 'epoch_isolation' and resets == 0, 'reset placement')
                require(event == {'kind': 'RESET', 'from_epoch': 1, 'to_epoch': 2}, 'reset schema')
                epoch, resets = 2, 1
                continue
            require(set(event) <= {'id', 'channel', 'stamp_ns', 'epoch'}, 'event fields')
            ident = event['id']
            require(isinstance(ident, str) and 0 < len(ident) <= 64 and ident not in ids, 'unique ID')
            ids.add(ident)
            require(event['channel'] in ('left', 'right'), 'channel')
            stamp = event['stamp_ns']
            require(type(stamp) is int and 0 <= stamp < 2147483648 * 1000000000, 'integer ROS stamp')
            require(type(event.get('epoch', 1)) is int and event.get('epoch', 1) == epoch, 'epoch')
        require(resets == (1 if case['id'] == 'epoch_isolation' else 0), 'reset count')
    return doc


def fields(msg):
    return {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec,
            'frame_id': msg.header.frame_id,
            'point': [msg.point.x, msg.point.y, msg.point.z]}


def expected_fields(event, index, epoch):
    sec, ns = divmod(event['stamp_ns'], 1000000000)
    return {'sec': sec, 'nanosec': ns, 'frame_id': event['id'],
            'point': [float(index), float(epoch), 0.0 if event['channel'] == 'left' else 1.0]}


def binding(case, epoch, event, index, feed_seq):
    """Exact typed source coordinates, not an authentication token."""
    return {'case': case, 'phase': case, 'epoch': epoch, 'id': event['id'],
            'channel': event['channel'], 'stamp_ns': event['stamp_ns'],
            'event_index': index, 'feed_seq': feed_seq}


def validate_capture(snap, expected_binding, capture):
    require(set(snap) == {'cdr_b64', 'cdr_sha256', 'decoded', 'binding', 'capture'}, 'snapshot keys')
    require(type(snap['binding']) is dict and set(snap['binding']) == set(expected_binding), 'binding keys')
    for key, value in expected_binding.items():
        require(type(snap['binding'][key]) is type(value) and snap['binding'][key] == value, 'typed source binding')
    require(type(snap['capture']) is dict and snap['capture'] == capture, 'capture provenance')
    require(all(type(snap['capture'][k]) is type(v) for k, v in capture.items()), 'capture types')
    require(type(snap['cdr_b64']) is str and type(snap['cdr_sha256']) is str, 'CDR metadata types')
    try:
        raw = base64.b64decode(snap['cdr_b64'], validate=True)
    except Exception as exc:
        raise ScenarioError('INVALID_METADATA', 'invalid capture base64') from exc
    require(bool(raw) and len(raw) <= MAX_BYTES and sha(raw) == snap['cdr_sha256'], 'capture hash')
    require(base64.b64encode(raw).decode('ascii') == snap['cdr_b64'], 'base64 encoding')
    return True


def validate_evidence(doc, inputs):
    """Pure structural/outcome checks; CDR base64/hash checks are NOT ROS decoding."""
    validate_inputs(inputs)
    semantic = doc['schema'] == SEMANTIC_SCHEMA
    require(doc['schema'] in (SCHEMA, SEMANTIC_SCHEMA) and doc['status'] == 'PASS', 'not completed')
    encode(doc)
    rows = doc['rows']
    require([r['seq'] for r in rows] == list(range(len(rows))), 'row order')
    cursor = 0
    counts = {}
    for case in inputs['cases']:
        cid, epoch, pairs, retained = case['id'], 1, [], {}
        def take(kind):
            nonlocal cursor
            require(cursor < len(rows), 'missing row')
            row = rows[cursor]
            cursor += 1
            require(row['kind'] == kind and row['case'] == cid, 'event order')
            return row
        take('start')
        for index, event in enumerate(case['events']):
            if event.get('kind') == 'RESET':
                row = take('reset')
                require(row['from_epoch'] == 1 and row['to_epoch'] == 2 and row['cleanup']['verified'], 'reset isolation')
                epoch = 2
                continue
            row = take('feed')
            require(row['id'] == event['id'] and row['channel'] == event['channel'] and row['epoch'] == epoch, 'feed identity')
            require(row['stamp_ns'] == event['stamp_ns'] and row['phase'] == cid, 'stamp/phase')
            require(row['decoded'] == expected_fields(event, index, epoch), 'payload')
            try:
                raw = base64.b64decode(row['cdr_b64'], validate=True)
            except Exception as exc:
                raise ScenarioError('INVALID_METADATA', 'invalid base64') from exc
            require(bool(raw) and sha(raw) == row['cdr_sha256'], 'CDR hash')
            if semantic:
                snap = {k: row[k] for k in ('cdr_b64', 'cdr_sha256', 'decoded', 'binding', 'capture')}
                validate_capture(snap, binding(cid, epoch, event, index, row['seq']),
                                 {'kind': 'feed', 'row_seq': row['seq'], 'side': event['channel']})
            retained[event['id']] = row
            if event['id'] in TRIGGERS:
                cb = take('callback')
                pair = tuple(cb['ids'])
                require(len(pairs) < len(EXPECTED[cid]) and pair == EXPECTED[cid][len(pairs)], 'wrong pair')
                require(cb['epoch'] == epoch and cb['trigger'] == event['id'], 'callback causality')
                for side, ident in enumerate(pair):
                    source = retained[ident]
                    require(source['epoch'] == epoch and source['channel'] == ('left', 'right')[side], 'callback epoch/channel')
                    if semantic:
                        require(len(cb['snapshots']) == 2, 'snapshot count')
                        snap = cb['snapshots'][side]
                        validate_capture(snap, source['binding'],
                                         {'kind': 'callback', 'row_seq': cb['seq'], 'side': ('left', 'right')[side], 'same_object': True})
                        require(snap['decoded'] == source['decoded'], 'callback payload/stamp')
                    else:
                        # Explicit legacy V1 contract, never emitted by the repaired runtime.
                        require(cb['snapshots'][side] == {k: source[k] for k in ('cdr_b64', 'cdr_sha256', 'decoded')}, 'callback bytes/payload')
                pairs.append(pair)
            returned = take('feed_return')
            require(returned['id'] == event['id'] and returned['callback_count'] == len(pairs), 'feed return count')
        end = take('end')
        require(end['callback_count'] == len(pairs) and pairs == EXPECTED[cid], 'missing/extra callback')
        require(end['cleanup']['verified'], 'cleanup')
        counts[cid] = len(pairs)
    require(cursor == len(rows) and doc['counts'] == counts, 'trailing rows/counts')
    require(doc['hashes_before'] == doc['hashes_after'] and bool(doc['hashes_before']), 'source/input/SDK changed')
    require(bool(doc['modules']), 'module identities missing')
    require(doc['work_seconds'] <= 20 and doc['whole_seconds'] <= 30, 'deadline')
    return True


def run_local_message_filter_scenario(case_id, nonce, input_path, evidence_path):
    """Execute only under separate runtime authorization; no retries, no ROS init."""
    started = time.monotonic()
    doc = {'schema': SEMANTIC_SCHEMA, 'case_id': case_id, 'nonce': nonce,
           'status': 'FAIL', 'rows': [], 'counts': {}, 'modules': {},
           'hashes_before': {}, 'hashes_after': {}, 'errors': []}
    output = Path(evidence_path).absolute()
    input_file = Path(input_path).resolve()
    source = Path(__file__).resolve()
    test_file = source.with_name('test_message_filter_scenario.py')
    notes = source.with_name('MESSAGE_FILTER_SDK_V1.md')
    graph = None
    alarm_set = False
    old_handler = None
    old_timer = None
    paths = set()

    def check_time():
        require(time.monotonic() - started < 20, 'SDK work deadline', 'WORK_TIMEOUT')

    def row(kind, cid, **data):
        require(len(doc['rows']) < 250, 'row budget', 'RESOURCE_LIMIT')
        doc['rows'].append(dict(seq=len(doc['rows']), kind=kind, case=cid, **data))

    def hashes():
        return {str(p): sha(p.read_bytes()) for p in sorted(paths)}

    def close(g):
        if g is None:
            return {'verified': True, 'queues_before': [], 'connections_remaining': 0}
        filters, sync, connection = g
        before = [{str(k): v.header.frame_id for k, v in q.items()} for q in sync.queues]
        for f, conn in zip(filters, sync.input_connections):
            f.unregisterCallback(conn)
        sync.unregisterCallback(connection)
        for q in sync.queues:
            q.clear()
        remaining = sum(len(f.callbacks) for f in filters) + len(sync.callbacks)
        verified = remaining == 0 and all(not q for q in sync.queues)
        require(verified, 'owned graph cleanup failed', 'CLEANUP_FAILED')
        return {'verified': verified, 'queues_before': before, 'connections_remaining': remaining}

    try:
        require(isinstance(case_id, str) and 0 < len(case_id) <= 128 and isinstance(nonce, str) and 0 < len(nonce) <= 128, 'case/nonce')
        require(not output.exists() and output.resolve() not in {input_file, source, test_file, notes}, 'evidence must be fresh distinct file')
        require(threading.current_thread() is threading.main_thread(), 'main thread required', 'UNSUPPORTED_API')
        require(hasattr(signal, 'SIGALRM'), 'POSIX alarm required', 'UNSUPPORTED_API')
        require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'existing alarm not owned', 'UNSUPPORTED_API')
        def timeout(signum, frame):
            raise ScenarioError('TIMEOUT', 'absolute child deadline')
        old_handler = signal.signal(signal.SIGALRM, timeout)
        old_timer = signal.setitimer(signal.ITIMER_REAL, max(0.001, 20 - (time.monotonic() - started)))
        alarm_set = True
        raw_input = input_file.read_bytes()
        require(len(raw_input) <= MAX_BYTES, 'input bound')
        inputs = validate_inputs(json.loads(raw_input))
        paths.update((source, test_file, notes, input_file))
        # Lazy imports: no SDK import in module loading or pure validators.
        try:
            mf = importlib.import_module('message_filters')
            simple = importlib.import_module('message_filters.simple_filter')
            geometry = importlib.import_module('geometry_msgs.msg._point_stamped')
            serialization = importlib.import_module('rclpy.serialization')
            builtin = importlib.import_module('builtin_interfaces.msg._time')
        except Exception as exc:
            raise ScenarioError('SDK_IMPORT_FAILED', str(exc)) from exc
        for module in (mf, simple, geometry, serialization, builtin):
            path = Path(inspect.getfile(module)).resolve()
            paths.add(path)
            doc['modules'][module.__name__] = {'file': str(path), 'name': module.__name__, 'spec_origin': module.__spec__.origin}
        package_xml = Path('/opt/ros/jazzy/share/message_filters/package.xml')
        paths.add(package_xml)
        doc['hashes_before'] = hashes()
        require(mf.SimpleFilter is simple.SimpleFilter, 'SimpleFilter identity', 'UNSUPPORTED_API')
        require(str(inspect.signature(mf.ApproximateTimeSynchronizer)) == '(fs, queue_size, slop, allow_headerless=False)', 'ATS signature', 'UNSUPPORTED_API')
        PointStamped = geometry.PointStamped
        serialize, deserialize = serialization.serialize_message, serialization.deserialize_message

        def snapshot(msg):
            raw = bytes(serialize(msg))
            decoded = deserialize(raw, PointStamped)
            require(fields(decoded) == fields(msg), 'roundtrip differs', 'ROUNDTRIP_FAILED')
            return {'cdr_b64': base64.b64encode(raw).decode('ascii'), 'cdr_sha256': sha(raw), 'decoded': fields(decoded)}

        for case in inputs['cases']:
            check_time()
            cid, epoch, count, trigger = case['id'], 1, 0, None
            retained = {}
            def callback(left, right):
                nonlocal count
                snaps = [snapshot(left), snapshot(right)]
                ids = [left.header.frame_id, right.header.frame_id]
                # Save both actual captures even when subsequent source checks fail.
                callback_seq = len(doc['rows'])
                for side, (msg, ident, snap) in enumerate(zip((left, right), ids, snaps)):
                    source_record = retained.get(ident)
                    snap['binding'] = dict(source_record['binding']) if source_record else None
                    snap['capture'] = {'kind': 'callback', 'row_seq': callback_seq,
                                       'side': ('left', 'right')[side]}
                    # The row is retained before enforcing this observed identity fact.
                    snap['capture']['same_object'] = source_record is not None and msg is source_record['message']
                row('callback', cid, epoch=epoch, trigger=trigger, ids=ids, snapshots=snaps)
                for side, (msg, ident, snap) in enumerate(zip((left, right), ids, snaps)):
                    source_record = retained.get(ident)
                    require(source_record is not None and msg is source_record['message'],
                            'callback object is not retained feed object', 'SDK_OUTCOME_FAILED')
                    bound = source_record['binding']
                    require(bound['case'] == cid and bound['phase'] == cid and bound['epoch'] == epoch
                            and bound['channel'] == ('left', 'right')[side],
                            'callback phase/epoch/channel mismatch', 'SDK_OUTCOME_FAILED')
                    require(snap['decoded'] == source_record['snapshot']['decoded'],
                            'callback fields/stamp changed', 'SDK_OUTCOME_FAILED')
                count += 1

            def new_graph():
                fs = [mf.SimpleFilter(), mf.SimpleFilter()]
                sync = mf.ApproximateTimeSynchronizer(fs, 2, 0.01, allow_headerless=False)
                conn = sync.registerCallback(callback)
                return fs, sync, conn

            graph = new_graph()
            row('start', cid, epoch=epoch)
            for index, event in enumerate(case['events']):
                check_time()
                if event.get('kind') == 'RESET':
                    cleanup = close(graph)
                    graph = None
                    epoch = 2
                    graph = new_graph()
                    row('reset', cid, from_epoch=1, to_epoch=2, cleanup=cleanup)
                    continue
                msg = PointStamped()
                expected = expected_fields(event, index, epoch)
                msg.header.stamp.sec = expected['sec']
                msg.header.stamp.nanosec = expected['nanosec']
                msg.header.frame_id = expected['frame_id']
                msg.point.x, msg.point.y, msg.point.z = expected['point']
                snap = snapshot(msg)
                require(snap['decoded'] == expected, 'generated payload differs', 'ROUNDTRIP_FAILED')
                bound = binding(cid, epoch, event, index, len(doc['rows']))
                snap['binding'] = dict(bound)
                snap['capture'] = {'kind': 'feed', 'row_seq': len(doc['rows']), 'side': event['channel']}
                # Strong references prevent address reuse; `is` rejects equal foreign objects.
                retained[event['id']] = {'message': msg, 'binding': bound, 'snapshot': snap}
                trigger = event['id']
                row('feed', cid, phase=cid, epoch=epoch, id=trigger, channel=event['channel'], stamp_ns=event['stamp_ns'], **snap)
                graph[0][0 if event['channel'] == 'left' else 1].signalMessage(msg)
                row('feed_return', cid, id=trigger, callback_count=count,
                    queues=[{str(k): v.header.frame_id for k, v in q.items()} for q in graph[1].queues])
            cleanup = close(graph)
            graph = None
            row('end', cid, callback_count=count, cleanup=cleanup)
            doc['counts'][cid] = count
        check_time()
        doc['work_seconds'] = time.monotonic() - started
        signal.setitimer(signal.ITIMER_REAL, max(0.001, 30 - doc['work_seconds']))
        doc['hashes_after'] = hashes()
        doc['whole_seconds'] = time.monotonic() - started
        doc['status'] = 'PASS'
        validate_evidence(doc, inputs)
    except Exception as exc:
        doc['status'] = 'FAIL'
        doc['errors'].append({'code': getattr(exc, 'code', 'SDK_OR_API_FAILURE'), 'detail': str(exc)[:2048], 'type': type(exc).__name__})
    finally:
        try:
            if graph is not None:
                doc['failure_cleanup'] = close(graph)
            if paths:
                doc['hashes_after'] = hashes()
        except Exception as exc:
            doc['status'] = 'FAIL'
            doc['errors'].append({'code': 'CLEANUP_FAILED', 'detail': str(exc)[:2048]})
        doc['whole_seconds'] = time.monotonic() - started
        if doc['whole_seconds'] > 30:
            doc['status'] = 'FAIL'
            doc['errors'].append({'code': 'WHOLE_TIMEOUT', 'detail': '30 seconds exceeded'})
        try:
            # Exclusive creation: never overwrite an input or existing evidence.
            with output.open('xb') as stream:
                stream.write(encode(doc))
        finally:
            if alarm_set:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
    return doc


def main():
    parser = argparse.ArgumentParser(description='Generated local SDK only; requires separate runtime approval')
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--input-path', required=True)
    parser.add_argument('--evidence-path', required=True)
    args = parser.parse_args()
    try:
        result = run_local_message_filter_scenario(args.case_id, args.nonce, args.input_path, args.evidence_path)
        summary = {'status': result['status'], 'counts': result['counts'], 'errors': result['errors']}
        print(json.dumps(summary, separators=(',', ':'))[:60000])
        return 0 if result['status'] == 'PASS' else 1
    except Exception as exc:
        print(json.dumps({'status': 'FAIL', 'code': getattr(exc, 'code', 'EVIDENCE_WRITE_FAILED'), 'detail': str(exc)[:2048]}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
