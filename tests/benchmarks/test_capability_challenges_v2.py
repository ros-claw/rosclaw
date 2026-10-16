"""Authored, NOT RUN. Root must qualify these tests and independent references."""
import json
import unittest
from copy import deepcopy

import jsonschema

from benchmarks.harnessbench.capability_challenge_judge_v2 import (
    Invalid,
    _solve,
    equal,
    judge_bytes,
    parse,
    reference_result,
)
from benchmarks.harnessbench.capability_challenges_v2 import (
    CONTRACTS,
    ERRORS,
    RESULT_SCHEMAS,
    RESULT_VALUE_SCHEMAS,
    VERSION,
    build_registry,
    encode,
    make_instance,
    sha,
)


def answer(i, result=None):
    return encode({"scope": "FIXTURE_ONLY", "result": reference_result(i) if result is None else result,
                   "input_sha256": sha(i.staged()["input.json"]), "reason": "Synthetic fixture only."})


class ChallengeTests(unittest.TestCase):
    def test_plan_and_all_invalid(self):
        plan = build_registry()
        self.assertEqual(len(plan), 30)
        self.assertEqual(plan, build_registry())
        for i in plan.values():
            with self.subTest(task=i.task_id):
                self.assertEqual(i.manifest()["input_hashes"]["input.json"], sha(i.staged()["input.json"]))
                result = reference_result(i)
                self.assertEqual(result["status"], "INVALID" if i.role == "invalid" else "OK")
                if i.role == "invalid":
                    self.assertEqual(result, {"status": "INVALID", "code": ERRORS[i.family], "values": None})
                self.assertTrue(judge_bytes(i.task_id, i.staged(), answer(i))["task_success"])
                changed = i.staged()
                changed["input.json"] += b" "
                self.assertFalse(judge_bytes(i.task_id, changed, answer(i))["task_success"])

    def test_hand_calculated_boundaries(self):
        def v(f):
            return reference_result(make_instance(f, "success"))["values"]
        self.assertEqual([x["compatible"] for x in v("H01")["links"]], [False, True])
        self.assertTrue(v("H02")["fresh"])
        self.assertEqual(v("H02")["distance"], 0)
        self.assertEqual(v("H03")["matches"], [[0, 0], [2, 1], [3, 2]])
        self.assertEqual(v("H03")["lost"], 2)
        self.assertEqual(v("H04")["valid_depth"], [True, False])
        self.assertIn([-1, 0], v("H05")["voxels"])
        self.assertEqual(len(v("H05")["blocked"]), 9)
        self.assertEqual(v("H06")["path"], ["a", "b", "d"])
        self.assertTrue(v("H06")["braking_safe"])
        self.assertTrue(v("H07")["reachable"])
        self.assertTrue(v("H07")["graspable"])
        self.assertEqual(v("H08")["accepted"], [0, 1, 3])
        self.assertEqual(v("H09")["joints"][0]["delta"], -8)
        self.assertEqual(v("H09")["joints"][0]["force"], -2)
        self.assertEqual(v("H10")["history"], ["result.json"])

    def test_metamorphic_all_families(self):
        for family in CONTRACTS:
            a = reference_result(make_instance(family, "success"))["values"]
            b = deepcopy(reference_result(make_instance(family, "companion"))["values"])
            if family == "H01":
                b["topic"] = "/lab/reader/data"
                for link in b["links"]:
                    link["id"] = link["id"][2:]
            elif family == "H02":
                b["frame"] = "map"
            elif family == "H06":
                b["path"] = [n[2:] for n in b["path"]]
            elif family == "H08":
                for key in ("ma", "ema", "posterior"):
                    b[key] = [x-7 for x in b[key]]
            elif family == "H10":
                b["target"] = b["target"][2:]
            self.assertTrue(equal(a, b), family)

    def test_strict_answer_mutations(self):
        i = make_instance("H06", "success")
        a = parse(answer(i))
        mutations = []
        for key, value in (("scope", "REAL"), ("input_sha256", "0"*64), ("reason", " ")):
            changed = deepcopy(a)
            changed[key] = value
            mutations.append(encode(changed))
        changed = deepcopy(a)
        changed["result"]["values"]["braking_safe"] = 1
        mutations.append(encode(changed))
        mutations.extend([b'{"scope":1,"scope":2}', b'{"result":NaN}', b'[]'])
        for raw in mutations:
            self.assertFalse(judge_bytes(i.task_id, i.staged(), raw)["task_success"])
        self.assertFalse(equal(True, 1))
        self.assertFalse(equal(float("nan"), 1))

    def test_recovery_and_safety_mutations(self):
        i = make_instance("H10", "success")
        p = parse(i.staged()["input.json"])["data"]
        p["budget"] = 0
        blocked = _solve("H10", p, i.staged())
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertEqual(blocked["values"]["history"], [])
        p["budget"] = 3
        p["landmarks"].append({"id": "z", "label": "door"})
        self.assertEqual(_solve("H10", p, i.staged())["decision"], "AMBIGUOUS")
        aux = i.staged()
        cp = parse(aux["checkpoint.json"])
        cp["revision"] += 1
        aux["checkpoint.json"] = encode(cp)
        with self.assertRaisesRegex(Invalid, "STALE_CHECKPOINT"):
            _solve("H10", p, aux)
        g = parse(make_instance("H06", "success").staged()["input.json"])["data"]
        g["forbidden"] = [["b", "d"], ["c", "d"]]
        self.assertFalse(_solve("H06", g, {})["reachable"])
        t = parse(make_instance("H02", "success").staged()["input.json"])["data"]
        t["stamp"] = t["now"]+1
        self.assertFalse(_solve("H02", t, {})["fresh"])

    def test_public_schema_all_ten_family_keys(self):
        expected = {
            "H01": {"links", "topic", "state"},
            "H02": {"point", "roundtrip", "gravity_free", "distance", "torque", "frame", "fresh"},
            "H03": {"matches", "rewinds", "lost", "fresh"},
            "H04": {"rgb", "valid_depth", "xyz"},
            "H05": {"voxels", "blocked"},
            "H06": {"reachable", "cost", "path", "braking_safe", "time_valid"},
            "H07": {"reachable", "graspable", "odometry"},
            "H08": {"ma", "ema", "posterior", "variance", "accepted"},
            "H09": {"joints"},
            "H10": {"decision", "target", "history", "required_calls"},
        }
        self.assertEqual(set(RESULT_VALUE_SCHEMAS), set(expected))
        self.assertEqual(set(RESULT_SCHEMAS), set(expected))
        for family, keys in expected.items():
            schema = RESULT_VALUE_SCHEMAS[family]
            jsonschema.Draft202012Validator.check_schema(schema)
            jsonschema.Draft202012Validator.check_schema(RESULT_SCHEMAS[family])
            self.assertEqual(set(schema["properties"]), keys)
            self.assertEqual(set(schema["required"]), keys)
            self.assertIs(schema["additionalProperties"], False)
            invalid = RESULT_SCHEMAS[family]
            self.assertEqual(invalid["properties"]["values"], {"type": "null"})
            self.assertIn(ERRORS[family], invalid["properties"]["code"]["enum"])
        self.assertEqual(set(RESULT_VALUE_SCHEMAS["H01"]["properties"]["links"]["items"]["required"]),
                         {"id", "compatible"})
        self.assertEqual(set(RESULT_VALUE_SCHEMAS["H09"]["properties"]["joints"]["items"]["required"]),
                         {"name", "address", "delta", "command", "force", "action"})

    def test_public_prompt_schema_all_existing_thirty(self):
        before = deepcopy(RESULT_VALUE_SCHEMAS)
        invalid_before = deepcopy(RESULT_SCHEMAS)
        for i in build_registry().values():
            with self.subTest(task=i.task_id):
                prompt = i.prompt()
                raw = prompt.split("Public answer schema: ", 1)[1].split("\nPublic provenance: ", 1)[0]
                schema = json.loads(raw)
                jsonschema.Draft202012Validator.check_schema(schema)
                jsonschema.Draft202012Validator(schema).validate(parse(answer(i)))
                self.assertEqual(set(schema["required"]), {"scope", "result", "input_sha256", "reason"})
                self.assertEqual(len(schema["properties"]["result"]["oneOf"]), 3 if i.family == "H10" else 2)
                self.assertIn(CONTRACTS[i.family], prompt)
                self.assertIn(VERSION, schema["title"])
        self.assertEqual(RESULT_VALUE_SCHEMAS, before)
        self.assertEqual(RESULT_SCHEMAS, invalid_before)

    def test_public_schema_missing_extra_and_alias_rejection(self):
        aliases = {"H01": ("state", "lifecycle"), "H02": ("distance", "plane_distance"),
                   "H09": ("joints", "rows")}
        for family in CONTRACTS:
            i = make_instance(family, "success")
            original = reference_result(i)["values"]
            validator = jsonschema.Draft202012Validator(RESULT_VALUE_SCHEMAS[family])
            for key in original:
                missing = deepcopy(original)
                del missing[key]
                self.assertFalse(validator.is_valid(missing), (family, key))
                extra = deepcopy(original)
                extra["undeclared"] = None
                self.assertFalse(validator.is_valid(extra), family)
            if family in aliases:
                old, alias = aliases[family]
                renamed = deepcopy(original)
                renamed[alias] = renamed.pop(old)
                self.assertFalse(validator.is_valid(renamed), family)
                self.assertFalse(judge_bytes(i.task_id, i.staged(), answer(i, {
                    "status": "OK", "code": "OK", "values": renamed}))["task_success"])
        for family, field in (("H01", "links"), ("H09", "joints")):
            original = reference_result(make_instance(family, "success"))["values"]
            for mutation in ("missing", "extra"):
                changed = deepcopy(original)
                if mutation == "missing":
                    del changed[field][0]["id" if family == "H01" else "name"]
                else:
                    changed[field][0]["undeclared"] = 0
                self.assertFalse(jsonschema.Draft202012Validator(RESULT_VALUE_SCHEMAS[family]).is_valid(changed))

    def test_public_strict_numeric_boolean_integer_types(self):
        # JSON Schema normally admits 1.0 as an integer; the public token rule
        # intentionally does not. This test's checker enforces the stated rule.
        checker = jsonschema.Draft202012Validator.TYPE_CHECKER.redefine(
            "integer", lambda checker, value: type(value) is int).redefine(
            "number", lambda checker, value: type(value) in (int, float)
            and value == value and value not in (float("inf"), float("-inf")))
        strict = jsonschema.validators.extend(jsonschema.Draft202012Validator, type_checker=checker)
        cases = [("H02", "distance", [True, "1", None, float("nan"), float("inf")]),
                 ("H03", "lost", [False, "1", 1.0, None]),
                 ("H06", "braking_safe", [0, 1, "true", None]),
                 ("H07", "reachable", [1, "false", None]),
                 ("H10", "required_calls", [True, "1", 1.0, -1])]
        for family, key, bad_values in cases:
            original = reference_result(make_instance(family, "success"))["values"]
            validator = strict(RESULT_VALUE_SCHEMAS[family])
            self.assertTrue(validator.is_valid(original))
            for value in bad_values:
                changed = deepcopy(original)
                changed[key] = value
                self.assertFalse(validator.is_valid(changed), (family, key, value))
        self.assertFalse(strict({"type": "integer"}).is_valid(True))
        self.assertFalse(strict({"type": "number"}).is_valid(False))
        self.assertTrue(strict({"type": "number"}).is_valid(1.0))

    def test_public_arrays_cardinality_and_h08_per_sample(self):
        for family in CONTRACTS:
            i = make_instance(family, "success")
            schema = json.loads(i.prompt().split("Public answer schema: ", 1)[1].split("\nPublic provenance: ", 1)[0])
            validator = jsonschema.Draft202012Validator(schema)
            original = parse(answer(i))
            for key, value in original["result"]["values"].items():
                if type(value) is list:
                    changed = deepcopy(original)
                    changed["result"]["values"][key] = 0
                    self.assertFalse(validator.is_valid(changed), (family, key))
        i = make_instance("H08", "success")
        data = parse(i.staged()["input.json"])["data"]
        schema = json.loads(i.prompt().split("Public answer schema: ", 1)[1].split("\nPublic provenance: ", 1)[0])
        validator = jsonschema.Draft202012Validator(schema)
        original = parse(answer(i))
        for key in ("ma", "ema", "posterior", "variance"):
            self.assertEqual(len(original["result"]["values"][key]), len(data["samples"]))
            for bad in (original["result"]["values"][key][-1], [],
                        original["result"]["values"][key][:-1],
                        [0] + original["result"]["values"][key]):
                changed = deepcopy(original)
                changed["result"]["values"][key] = bad
                self.assertFalse(validator.is_valid(changed), key)
                self.assertFalse(judge_bytes(i.task_id, i.staged(), encode(changed))["task_success"])
        self.assertIn("ORIGINAL sample order", CONTRACTS["H08"])
        self.assertIn("NOT final scalars", CONTRACTS["H08"])

    def test_public_nullable_and_h10_status_variants(self):
        i = make_instance("H10", "success")
        schema = json.loads(i.prompt().split("Public answer schema: ", 1)[1].split("\nPublic provenance: ", 1)[0])
        validator = jsonschema.Draft202012Validator(schema["properties"]["result"])
        p = parse(i.staged()["input.json"])["data"]
        p["budget"] = 0
        blocked = _solve("H10", p, i.staged())
        self.assertTrue(validator.is_valid(blocked))
        self.assertEqual(set(blocked["values"]), {"history", "required_calls"})
        self.assertEqual(blocked["values"]["history"], parse(i.staged()["checkpoint.json"])["history"])
        mutations = []
        for key, value in (("decision", "WAIT"), ("target", None)):
            changed = deepcopy(blocked)
            changed["values"][key] = value
            mutations.append(changed)
        changed = deepcopy(blocked)
        changed["status"] = "OK"
        mutations.append(changed)
        changed = deepcopy(blocked)
        changed["code"] = "OK"
        mutations.append(changed)
        for changed in mutations:
            self.assertFalse(validator.is_valid(changed))
        for code in ("INVALID_JSON", "STALE_CHECKPOINT"):
            invalid = {"status": "INVALID", "code": code, "values": None}
            self.assertTrue(validator.is_valid(invalid))
            invalid["values"] = {}
            self.assertFalse(validator.is_valid(invalid))
        for decision in ("ACT", "WAIT", "AMBIGUOUS", "NOT_FOUND"):
            values = {"decision": decision, "target": "symbol" if decision in ("ACT", "WAIT") else None,
                      "history": [], "required_calls": 0}
            v = jsonschema.Draft202012Validator(RESULT_VALUE_SCHEMAS["H10"])
            self.assertTrue(v.is_valid(values))
            values["target"] = None if values["target"] is not None else "symbol"
            self.assertFalse(v.is_valid(values))
        nav = {"reachable": False, "cost": None, "path": [], "braking_safe": True, "time_valid": True}
        v = jsonschema.Draft202012Validator(RESULT_VALUE_SCHEMAS["H06"])
        self.assertTrue(v.is_valid(nav))
        nav["cost"] = 0
        self.assertFalse(v.is_valid(nav))

    def test_public_version_manifest_and_existing_plan(self):
        self.assertEqual(VERSION, "rosclaw.capability_challenges.v2.2")
        plan = build_registry()
        self.assertEqual(len(plan), 30)
        self.assertEqual(set(plan), {f"H{n:02d}-{suffix}" for n in range(1, 11) for suffix in ("S", "I", "M")})
        for i in plan.values():
            files = i.staged()
            envelope = parse(files["input.json"])
            manifest = i.manifest()
            self.assertEqual(envelope["version"], VERSION)
            self.assertEqual(manifest["version"], VERSION)
            self.assertEqual(i.seed, 2026101600 + int(i.family[1:]))
            self.assertEqual(set(files), {"input.json", "batch.json", "checkpoint.json"}
                             if i.family == "H10" else {"input.json"})
            self.assertEqual(manifest["input_hashes"], {k: sha(v) for k, v in files.items()})
            self.assertEqual(manifest["contract_sha256"], sha(CONTRACTS[i.family].encode()))
            self.assertEqual(envelope["aux_sha256"], {k: sha(v) for k, v in files.items() if k != "input.json"})
            provenance = json.loads(i.prompt().split("Public provenance: ", 1)[1])
            self.assertEqual(provenance, manifest)
            # Only the envelope version differs from prior-version encoding;
            # do not bless an old answer/hash/grade as evidence for this version.
            old = deepcopy(envelope)
            old["version"] = "rosclaw.capability_challenges.v2.1"
            self.assertNotEqual(sha(encode(old)), sha(files["input.json"]))
