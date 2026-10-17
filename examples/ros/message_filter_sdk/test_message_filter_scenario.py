"""Synthetic metadata regression suite. No SDK imports, runtime or real CDR claims.

Running this suite is a separate approval; it was not run during authoring.
"""
import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import message_filter_scenario as scenario


def synthetic_fixture():
    """Build deliberately NON-CDR bytes for structural validator tests only."""
    inputs = json.loads((Path(__file__).parent / 'inputs' / 'INPUT_CASES.json').read_text())
    doc = {'schema': scenario.SCHEMA, 'status': 'PASS', 'case_id': 'SYNTHETIC',
           'nonce': 'SYNTHETIC', 'data_origin': 'SYNTHETIC_NOT_SDK', 'rows': [],
           'counts': {}, 'modules': {'SYNTHETIC': {'file': 'not-installed'}},
           'hashes_before': {'SYNTHETIC': '0' * 64},
           'hashes_after': {'SYNTHETIC': '0' * 64},
           'work_seconds': 0.0, 'whole_seconds': 0.0, 'errors': []}
    def emit(kind, cid, **data):
        doc['rows'].append(dict(seq=len(doc['rows']), kind=kind, case=cid, **data))
    cleanup = {'verified': True, 'queues_before': [], 'connections_remaining': 0}
    for case in inputs['cases']:
        cid, epoch, count = case['id'], 1, 0
        retained = {}
        emit('start', cid, epoch=epoch)
        for index, event in enumerate(case['events']):
            if event.get('kind') == 'RESET':
                emit('reset', cid, from_epoch=1, to_epoch=2, cleanup=copy.deepcopy(cleanup))
                epoch = 2
                continue
            raw = ('SYNTHETIC_NOT_CDR:' + event['id']).encode()
            snap = {'cdr_b64': base64.b64encode(raw).decode(),
                    'cdr_sha256': scenario.sha(raw),
                    'decoded': scenario.expected_fields(event, index, epoch)}
            retained[event['id']] = snap
            emit('feed', cid, phase=cid, epoch=epoch, id=event['id'],
                 channel=event['channel'], stamp_ns=event['stamp_ns'], **copy.deepcopy(snap))
            if event['id'] in scenario.TRIGGERS:
                pair = scenario.EXPECTED[cid][count]
                emit('callback', cid, epoch=epoch, trigger=event['id'], ids=list(pair),
                     snapshots=[copy.deepcopy(retained[x]) for x in pair])
                count += 1
            emit('feed_return', cid, id=event['id'], callback_count=count, queues=[])
        emit('end', cid, callback_count=count, cleanup=copy.deepcopy(cleanup))
        doc['counts'][cid] = count
    return inputs, doc


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.inputs, self.doc = synthetic_fixture()

    def row(self, kind, ident=None, cid=None):
        return next(r for r in self.doc['rows'] if r['kind'] == kind
                    and (ident is None or r.get('id') == ident)
                    and (cid is None or r['case'] == cid))

    def reject(self):
        with self.assertRaises((scenario.ScenarioError, KeyError, TypeError, ValueError)):
            scenario.validate_evidence(self.doc, self.inputs)

    def test_synthetic_structure_only(self):
        # Passing does not establish SDK provenance or that these bytes are CDR.
        with patch.object(scenario.importlib, 'import_module', side_effect=AssertionError('SDK forbidden')):
            self.assertTrue(scenario.validate_evidence(self.doc, self.inputs))
        self.assertEqual(sum(self.doc['counts'].values()), 6)
        self.assertLess(len(self.doc['rows']), 256)

    def test_wrong_pair(self):
        self.row('callback', cid='ordinary')['ids'].reverse()
        self.reject()

    def test_payload_mutation(self):
        self.row('feed', 'D_L_NEW')['decoded']['point'][0] = -1.0
        self.reject()

    def test_epoch_contamination(self):
        self.row('callback', cid='epoch_isolation')['epoch'] = 1
        self.reject()

    def test_extra_callback(self):
        self.doc['rows'].insert(1, copy.deepcopy(self.row('callback', cid='ordinary')))
        for index, row in enumerate(self.doc['rows']):
            row['seq'] = index
        self.reject()

    def test_missing_callback(self):
        self.doc['rows'].remove(self.row('callback', cid='ordinary'))
        for index, row in enumerate(self.doc['rows']):
            row['seq'] = index
        self.reject()

    def test_invalid_base64(self):
        self.row('feed', 'O_L')['cdr_b64'] = 'not base64!'
        self.reject()

    def test_stale_cdr_hash(self):
        self.row('feed', 'O_L')['cdr_sha256'] = '0' * 64
        self.reject()

    def test_callback_byte_substitution(self):
        self.row('callback', cid='duplicate_stamp')['snapshots'][0]['cdr_b64'] = 'WA=='
        self.reject()

    def test_changed_timestamp(self):
        self.row('feed', 'B_R')['stamp_ns'] -= 1
        self.reject()

    def test_changed_source_hash(self):
        self.doc['hashes_after']['SYNTHETIC'] = '1' * 64
        self.reject()

    def test_missing_feed(self):
        self.doc['rows'].remove(self.row('feed', 'O_L'))
        for index, row in enumerate(self.doc['rows']):
            row['seq'] = index
        self.reject()

    def test_feed_return_causality(self):
        self.row('feed_return', 'E2_R')['callback_count'] = 1
        self.reject()

    def test_cleanup_failure(self):
        self.row('reset', cid='epoch_isolation')['cleanup']['verified'] = False
        self.reject()

    def test_failure_not_empty_pass(self):
        self.doc['status'] = 'FAIL'
        self.reject()

    def test_time_limit(self):
        self.doc['work_seconds'] = 20.01
        self.reject()

    def test_row_and_byte_caps(self):
        expanded = copy.deepcopy(self.doc)
        expanded['rows'] = [{}] * 257
        with self.assertRaises(scenario.ScenarioError):
            scenario.encode(expanded)
        expanded['rows'] = []
        expanded['padding'] = 'x' * scenario.MAX_BYTES
        with self.assertRaises(scenario.ScenarioError):
            scenario.encode(expanded)

    def test_integer_stamp_not_float_or_bool(self):
        for stamp in (True, 1.0, -1, 2147483648 * 1000000000):
            altered = copy.deepcopy(self.inputs)
            altered['cases'][0]['events'][0]['stamp_ns'] = stamp
            with self.assertRaises(scenario.ScenarioError):
                scenario.validate_inputs(altered)

    def test_duplicate_input_id(self):
        self.inputs['cases'][0]['events'][1]['id'] = 'O_L'
        with self.assertRaises(scenario.ScenarioError):
            scenario.validate_inputs(self.inputs)

    def test_reset_is_explicit(self):
        del self.inputs['cases'][-1]['events'][1]
        with self.assertRaises(scenario.ScenarioError):
            scenario.validate_inputs(self.inputs)

    def test_synthetic_filesystem_roundtrip(self):
        # Fixture-only write under an automatically removed temporary directory.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.json'
            path.write_bytes(scenario.encode(self.doc))
            restored = json.loads(path.read_bytes())
            self.assertEqual(restored['data_origin'], 'SYNTHETIC_NOT_SDK')
            self.assertTrue(scenario.validate_evidence(restored, self.inputs))

    def test_public_signature_has_four_mandatory_args(self):
        import inspect
        parameters = inspect.signature(scenario.run_local_message_filter_scenario).parameters
        self.assertEqual(list(parameters), ['case_id', 'nonce', 'input_path', 'evidence_path'])
        self.assertTrue(all(p.default is inspect.Parameter.empty for p in parameters.values()))


if __name__ == '__main__':
    unittest.main()


# V2 additions: pytest/import collection reaches these after the legacy main guard.
# All bytes below are intentionally NON-CDR; no padding or SDK validity is asserted.
def semantic_fixture():
    inputs, doc = synthetic_fixture()
    doc['schema'] = scenario.SEMANTIC_SCHEMA
    feeds = {}
    for case in inputs['cases']:
        epoch = 1
        for index, event in enumerate(case['events']):
            if event.get('kind') == 'RESET':
                epoch = 2
                continue
            row = next(r for r in doc['rows'] if r['kind'] == 'feed' and r['id'] == event['id'])
            row['binding'] = scenario.binding(case['id'], epoch, event, index, row['seq'])
            row['capture'] = {'kind': 'feed', 'row_seq': row['seq'], 'side': event['channel']}
            feeds[event['id']] = row
    for row in doc['rows']:
        if row['kind'] != 'callback':
            continue
        for side, ident in enumerate(row['ids']):
            raw = ('INDEPENDENT_SYNTHETIC_NON_CDR_CALLBACK:' + ident).encode()
            row['snapshots'][side] = {
                'cdr_b64': base64.b64encode(raw).decode(), 'cdr_sha256': scenario.sha(raw),
                'decoded': copy.deepcopy(feeds[ident]['decoded']),
                'binding': copy.deepcopy(feeds[ident]['binding']),
                'capture': {'kind': 'callback', 'row_seq': row['seq'],
                            'side': ('left', 'right')[side], 'same_object': True}}
    return inputs, doc


class SemanticMetadataTests(unittest.TestCase):
    def setUp(self):
        self.inputs, self.doc = semantic_fixture()
        self.cb = next(r for r in self.doc['rows'] if r['kind'] == 'callback')
        self.snap = self.cb['snapshots'][0]

    def reject(self):
        with self.assertRaises((scenario.ScenarioError, KeyError, TypeError, ValueError)):
            scenario.validate_evidence(self.doc, self.inputs)

    def test_distinct_retained_captures(self):
        original = copy.deepcopy(self.doc)
        with patch.object(scenario.importlib, 'import_module', side_effect=AssertionError('SDK forbidden')):
            self.assertTrue(scenario.validate_evidence(self.doc, self.inputs))
        self.assertEqual(original, self.doc)
        feed = next(r for r in self.doc['rows'] if r.get('id') == 'O_L')
        self.assertNotEqual(feed['cdr_b64'], self.snap['cdr_b64'])
        self.assertNotEqual(feed['cdr_sha256'], self.snap['cdr_sha256'])

    def test_wrong_payload_and_stamp(self):
        for key, value in [('sec', 9), ('nanosec', 9), ('frame_id', 'FOREIGN'), ('point', [-1., 1., 0.])]:
            with self.subTest(key=key):
                saved = copy.deepcopy(self.snap['decoded'])
                self.snap['decoded'][key] = value
                self.reject()
                self.snap['decoded'] = saved

    def test_wrong_typed_source_coordinates(self):
        for key, value in [('id', 'FOREIGN'), ('phase', 'strict_boundary'), ('case', 'strict_boundary'),
                           ('epoch', 2), ('epoch', True), ('channel', 'right'),
                           ('stamp_ns', 7), ('event_index', 7), ('feed_seq', 7)]:
            with self.subTest(key=key, value=value):
                saved = copy.deepcopy(self.snap['binding'])
                self.snap['binding'][key] = value
                self.reject()
                self.snap['binding'] = saved

    def test_foreign_object_observation(self):
        self.snap['capture']['same_object'] = False
        self.reject()

    def test_swapped_captures(self):
        self.cb['snapshots'].reverse()
        self.reject()

    def test_copied_feed_snapshot(self):
        feed = next(r for r in self.doc['rows'] if r.get('id') == 'O_L')
        self.cb['snapshots'][0] = {k: copy.deepcopy(feed[k]) for k in self.snap}
        self.reject()

    def test_reused_callback_capture(self):
        self.snap['capture']['row_seq'] += 1
        self.reject()

    def test_stale_callback_hash(self):
        self.snap['cdr_sha256'] = '0' * 64
        self.reject()

    def test_malformed_capture_metadata(self):
        for key, value in [('cdr_b64', '!'), ('cdr_b64', ''), ('cdr_sha256', 1), ('binding', {})]:
            with self.subTest(key=key, value=value):
                saved = copy.deepcopy(self.snap[key])
                self.snap[key] = value
                self.reject()
                self.snap[key] = saved

    def test_identical_bytes_are_not_forbidden(self):
        # Legitimately equal representations are possible; capture context stays distinct.
        feed = next(r for r in self.doc['rows'] if r.get('id') == 'O_L')
        for key in ('cdr_b64', 'cdr_sha256'):
            self.snap[key] = feed[key]
        self.assertTrue(scenario.validate_evidence(self.doc, self.inputs))
        # This is metadata acceptance only: a malicious writer that forges all
        # provenance cannot be detected without independent observer/decoder evidence.
