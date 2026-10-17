"""Pure adversarial tests. No SDK imports, Context, subprocess, or runtime calls.

Fixtures are explicitly synthetic protocol examples, never execution evidence.
"""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

# Load this exact sibling product independent of test runner's sys.path.
_spec = importlib.util.spec_from_file_location('parameter_atomic_source', Path(__file__).with_name('parameter_atomic_scenario.py'))
source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(source)


def v(value, code=2):
    return {'type': code, 'value': value}


def p(name, value, code=2):
    return {'name': name, 'type': code, 'value': value}


def state(target, limit):
    return {'target': v(target), 'limit': v(limit), 'robot_id': v(42)}


def fixture(reverse_completion=False):
    """Independent, explicit example transcript, not the production request plan."""
    rows = []

    def add(event, **data):
        rows.append({'schema': 'ROSClaw.ParameterAtomic12.Event.v1', 'case': 'synthetic',
                     'nonce': 'not_runtime_evidence', 'domain': 71, 'seq': len(rows),
                     'event': event, 'data': data})

    def submit(rid, case, service, request):
        fid = 'synthetic_future_' + rid
        add('REQUEST_SUBMITTED', request_id=rid, case_id=case, service=service,
            future_id=fid, request=request)
        return (rid, case, service, fid)

    def resolve(ref, response):
        rid, case, service, fid = ref
        add('RESPONSE_RESOLVED', request_id=rid, case_id=case, service=service,
            future_id=fid, response=response)

    def result(case):
        add('CASE_RESULT', case_id=case, status='PASS', reason='synthetic fixture')

    add('BOOT', scope='PARAMETER_ATOMIC12_V1')
    # Duplicate publisher names are two entities, intentionally not deduplicated.
    add('ENTITY_INVENTORY', publishers=2, subscriptions=0, services=6, clients=3,
        names={'publishers': ['/parameter_events', '/parameter_events'], 'subscriptions': [],
               'services': ['/s/' + name for name in ('describe_parameters', 'get_parameter_types',
                            'get_parameters', 'list_parameters', 'set_parameters', 'set_parameters_atomically')],
               'clients': ['/s/get_parameters', '/s/set_parameters_atomically', '/never/get_parameters']})
    forward = submit('P1_GET_FORWARD', 'P1', 'get_parameters', {'names': ['target', 'limit', 'robot_id']})
    reverse = submit('P1_GET_REVERSE', 'P1', 'get_parameters', {'names': ['robot_id', 'limit', 'target']})
    replies = [(forward, {'values': [v(10), v(20), v(42)]}),
               (reverse, {'values': [v(42), v(20), v(10)]})]
    for ref, response in reversed(replies) if reverse_completion else replies:
        resolve(ref, response)
    result('P1')
    examples = [
        ('P2', [p('target', 12), p('limit', 30)], True, state(10, 20), state(12, 30), [12, 30, 42]),
        ('P3', [p('target', 50), p('limit', 30)], False, state(12, 30), state(50, 30), [12, 30, 42]),
        ('P4', [p('robot_id', 43), p('target', 13)], False, None, None, [12, 30, 42]),
        ('P5', [p('target', '12', 4), p('limit', 31)], False, None, None, [12, 30, 42]),
        ('P6', [p('target', 13), p('limit', 31)], True, state(12, 30), state(13, 31), [13, 31, 42]),
    ]
    for case, update, ok, before, candidate, readback in examples:
        if case == 'P6':
            add('ABSENCE_PROBE', case_id=case, service='/never/get_parameters', timeout_s=2,
                available=False, requests_sent=0)
        ref = submit(case + '_SET', case, 'set_parameters_atomically', {'parameters': update})
        if before is not None:
            add('PARAMETER_VALIDATION', case_id=case, proposed=copy.deepcopy(update),
                current_before=before, candidate=candidate, successful=ok,
                reason='' if ok else 'semantic candidate violation')
        resolve(ref, {'successful': ok, 'reason': '' if ok else 'rejected by policy or descriptor'})
        ref = submit(case + '_GET', case, 'get_parameters', {'names': ['target', 'limit', 'robot_id']})
        resolve(ref, {'values': [v(value) for value in readback]})
        result(case)
    add('FINAL', status='PASS', condition_statuses={case: 'PASS' for case in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6')},
        counters={'service_requests': 12, 'get_requests': 7, 'atomic_requests': 5,
                  'missing_service_requests': 0, 'parameter_event_publish_calls': None,
                  'parameter_event_publish_observation': 'UNKNOWN_NOT_INSTRUMENTED'},
        cleanup={'executor_shutdown': True, 'nodes_destroyed': 2, 'context_shutdown': True}, errors=[])
    return rows


def select(rows, event, rid=None, case=None):
    return next(row for row in rows if row['event'] == event and
                (rid is None or row['data'].get('request_id') == rid) and
                (case is None or row['data'].get('case_id') == case))


def resequence(rows):
    for index, row in enumerate(rows):
        row['seq'] = index
    return rows


class CandidateTests(unittest.TestCase):
    def test_paired_success_uses_both_and_preserves_inputs(self):
        current = state(10, 20)
        update = [p('target', 25), p('limit', 30)]
        original = copy.deepcopy((current, update))
        candidate, ok, reason = source.candidate_validation(current, update)
        self.assertTrue(ok)
        self.assertEqual(candidate, state(25, 30))
        self.assertEqual(reason, '')
        self.assertEqual((current, update), original)
        candidate['robot_id']['value'] = 99
        self.assertEqual(current, state(10, 20))

    def test_semantic_rejection_does_not_commit(self):
        current = state(12, 30)
        update = [p('target', 50), p('limit', 30)]
        snapshot = copy.deepcopy((current, update))
        candidate, ok, reason = source.candidate_validation(current, update)
        self.assertFalse(ok)
        self.assertEqual(candidate, state(50, 30))
        self.assertTrue(reason)
        self.assertEqual((current, update), snapshot)

    def test_boundaries_and_wrong_type(self):
        for target, limit, expected in [(0, 0, True), (100, 100, True), (-1, 20, False),
                                         (20, 19, False), (10, 101, False)]:
            with self.subTest(target=target, limit=limit):
                _, ok, _ = source.candidate_validation(state(10, 20), [p('target', target), p('limit', limit)])
                self.assertIs(ok, expected)
        self.assertFalse(source.candidate_validation(state(12, 30), [p('target', '12', 4)])[1])

    def test_bool_duplicate_and_unknown_rejected(self):
        for update in ([p('target', True)], [p('target', 12, True)],
                       [p('target', 12), p('target', 13)], [p('unknown', 1)]):
            with self.subTest(update=update), self.assertRaises(ValueError):
                source.candidate_validation(state(10, 20), update)


class TranscriptTests(unittest.TestCase):
    def assert_invalid(self, rows):
        with self.assertRaises(ValueError):
            source.validate_transcript(resequence(rows))

    def test_complete_and_asynchronous_reverse_completion(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                self.assertTrue(source.validate_transcript(fixture(reverse)))

    def test_missing_extra_duplicate_response(self):
        rows = fixture()
        rows.remove(select(rows, 'RESPONSE_RESOLVED', 'P3_GET'))
        self.assert_invalid(rows)
        rows = fixture()
        row = copy.deepcopy(select(rows, 'RESPONSE_RESOLVED', 'P1_GET_FORWARD'))
        rows.insert(6, row)
        self.assert_invalid(rows)
        rows = fixture()
        row = copy.deepcopy(select(rows, 'REQUEST_SUBMITTED', 'P6_GET'))
        row['data']['request_id'] = 'EXTRA_GET'
        rows.insert(-1, row)
        self.assert_invalid(rows)

    def test_swapped_or_reused_future_identity(self):
        rows = fixture()
        a = select(rows, 'RESPONSE_RESOLVED', 'P1_GET_FORWARD')['data']
        b = select(rows, 'RESPONSE_RESOLVED', 'P1_GET_REVERSE')['data']
        a['future_id'], b['future_id'] = b['future_id'], a['future_id']
        self.assert_invalid(rows)
        rows = fixture()
        select(rows, 'REQUEST_SUBMITTED', 'P1_GET_REVERSE')['data']['future_id'] = select(rows, 'REQUEST_SUBMITTED', 'P1_GET_FORWARD')['data']['future_id']
        self.assert_invalid(rows)

    def test_p1_both_submit_before_any_resolution(self):
        rows = fixture()
        row = select(rows, 'RESPONSE_RESOLVED', 'P1_GET_FORWARD')
        rows.remove(row)
        rows.insert(3, row)
        self.assert_invalid(rows)

    def test_ordered_names_and_values(self):
        for event, key in [('REQUEST_SUBMITTED', 'request'), ('RESPONSE_RESOLVED', 'response')]:
            rows = fixture()
            d = select(rows, event, 'P1_GET_REVERSE')['data'][key]
            field = 'names' if key == 'request' else 'values'
            d[field].reverse()
            self.assert_invalid(rows)

    def test_rollback_and_no_coercion(self):
        for case, index, bad in [('P3', 0, v(50)), ('P4', 2, v(43)),
                                 ('P4', 0, v(13)), ('P5', 0, v('12', 4)), ('P5', 1, v(31))]:
            rows = fixture()
            select(rows, 'RESPONSE_RESOLVED', case + '_GET')['data']['response']['values'][index] = bad
            with self.subTest(case=case, bad=bad):
                self.assert_invalid(rows)

    def test_type_bool_and_status_adversaries(self):
        mutations = [
            lambda r: select(r, 'RESPONSE_RESOLVED', 'P2_SET')['data']['response'].__setitem__('successful', 1),
            lambda r: select(r, 'RESPONSE_RESOLVED', 'P3_SET')['data']['response'].__setitem__('reason', ''),
            lambda r: select(r, 'RESPONSE_RESOLVED', 'P1_GET_FORWARD')['data']['response']['values'][0].__setitem__('type', True),
            lambda r: select(r, 'REQUEST_SUBMITTED', 'P2_SET')['data']['request']['parameters'][0].__setitem__('value', True),
            lambda r: r[0].__setitem__('domain', True),
            lambda r: r[-1]['data']['counters'].__setitem__('missing_service_requests', False),
            lambda r: r[1]['data'].__setitem__('publishers', True),
            lambda r: r[-1]['data']['cleanup'].__setitem__('nodes_destroyed', True),
            lambda r: select(r, 'ABSENCE_PROBE')['data'].__setitem__('available', 0),
        ]
        for index, mutate in enumerate(mutations):
            rows = fixture()
            mutate(rows)
            with self.subTest(index=index):
                self.assert_invalid(rows)

    def test_callback_evidence_is_paired_and_not_fabricated(self):
        rows = fixture()
        rows.remove(select(rows, 'PARAMETER_VALIDATION', case='P3'))
        self.assert_invalid(rows)
        rows = fixture()
        select(rows, 'PARAMETER_VALIDATION', case='P2')['data']['candidate']['limit'] = v(20)
        self.assert_invalid(rows)
        rows = fixture()
        fake = copy.deepcopy(select(rows, 'PARAMETER_VALIDATION', case='P3'))
        fake['data']['case_id'] = 'P4'
        rows.insert(rows.index(select(rows, 'RESPONSE_RESOLVED', 'P4_SET')), fake)
        self.assert_invalid(rows)

    def test_cleanup_and_final_are_mandatory(self):
        for key, value in [('executor_shutdown', False), ('nodes_destroyed', 1), ('context_shutdown', False)]:
            rows = fixture()
            rows[-1]['data']['cleanup'][key] = value
            self.assert_invalid(rows)
        self.assert_invalid(fixture()[:-1])
        rows = fixture()
        rows.append(copy.deepcopy(rows[0]))
        self.assert_invalid(rows)

    def test_partial_record_never_certifies_success(self):
        rows = fixture()[:3]  # one submitted request, no result
        self.assertFalse(source.validate_transcript(rows, require_complete=False))
        self.assert_invalid(rows)
        rows = fixture()
        rows[-1]['data']['status'] = 'FAIL'
        rows[-1]['data']['errors'] = [{'class': 'TimeoutError', 'message': 'synthetic failure'}]
        self.assertFalse(source.validate_transcript(rows, require_complete=False))
        self.assert_invalid(rows)

    def test_unmeasured_publish_calls_and_exact_keys(self):
        rows = fixture()
        rows[-1]['data']['counters']['parameter_event_publish_calls'] = 9
        self.assert_invalid(rows)
        rows = fixture()
        rows[0]['data']['extra'] = 'not allowed'
        self.assert_invalid(rows)
        rows = fixture()
        rows[1]['data']['names']['publishers'] = ['/parameter_events']
        self.assert_invalid(rows)

    def test_json_reader_duplicate_nonfinite_and_partial(self):
        # Temporary local data only; no ROS/process activity.
        for payload in ('{"a":1,"a":2}\n', '{"a":NaN}\n', '{"a":\n'):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'synthetic.jsonl'
                path.write_text(payload, encoding='utf-8')
                with self.assertRaises(ValueError):
                    source.read_jsonl(path)

    def test_sequence_and_identity(self):
        rows = fixture()
        rows[3]['seq'] = 99
        with self.assertRaises(ValueError):
            source.validate_transcript(rows)
        rows = fixture()
        rows[4]['nonce'] = 'different'
        self.assert_invalid(rows)


from enum import Enum


class PlainEnumProjectionTests(unittest.TestCase):
    class ParameterType(Enum):
        INTEGER = 2
        STRING = 4

    def test_integer_and_string_projections(self):
        class FakeParameter:
            def __init__(self, name, value, type_):
                self.name, self.value, self.type_ = name, value, type_

        for member, value, code in ((self.ParameterType.INTEGER, 12, 2),
                                    (self.ParameterType.STRING, '12', 4)):
            with self.subTest(member=member):
                item = FakeParameter('target', value, member)
                projected = source.parameter(
                    item.name, item.value, source.parameter_type_code(item.type_))
                snapshot = {item.name: source.typed(
                    item.value, source.parameter_type_code(item.type_))}
                self.assertEqual(projected, {'name': 'target', 'type': code, 'value': value})
                self.assertEqual(snapshot, {'target': {'type': code, 'value': value}})
                self.assertIs(type(projected['type']), int)
                self.assertIs(type(snapshot['target']['type']), int)
                source.check_value(projected, named=True)
                source.check_value(snapshot['target'])
                self.assertIs(item.type_, member)
                self.assertEqual(member.value, code)

    def test_old_integer_conversion_rejects_genuine_plain_enum(self):
        self.assertEqual(self.ParameterType.__bases__, (Enum,))
        for member in self.ParameterType:
            with self.subTest(member=member):
                with self.assertRaises(TypeError):
                    int(member)
                self.assertEqual(source.parameter_type_code(member), member.value)

    def test_invalid_values_are_not_coerced(self):
        class InvalidType(Enum):
            TEXT = '2'
            FLOAT = 2.0
            TRUE = True
            FALSE = False
            NONE = None

        for member in InvalidType:
            with self.subTest(member=member), self.assertRaises(ValueError):
                source.parameter_type_code(member)
        for missing_value in (object(), 2, '2', True, None):
            with self.subTest(missing_value=missing_value), self.assertRaises(AttributeError):
                source.parameter_type_code(missing_value)


if __name__ == '__main__':
    unittest.main()
