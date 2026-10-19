"""Synthetic contracts for the pinned, readonly action_mapping library.

Future unittest discovery is limited to the separately reviewed isolated real
namespace seam: genuine action_mapping package, no eager parent initializers.
Ordinary full production bootstrap: NOT_RUN. Author execution: NOT_RUN.
Default source overlay cb7d34f2de03d2fc307b44921f0e3ae4eb12260c;
checkout equality is comparison metadata, not source adoption.

Strict safety contracts intentionally expose suspected readonly defects:
extra duplicate policy names and inconsistent declared shape are not checked;
a nonfinite later-chunk value appears to index beyond body_joint_names in
validate_mapped_action. Those are potential failures/errors, not approved
acceptances, skips, or expected failures. No implementation is replaced.
Only synthetic vector adaptation is covered: no learned model, grounding,
robot operation, dynamics simulation, or CAP80 equivalence claim.
"""

import math
import unittest

from native import action_mapping_fixture_data as data
from rosclaw.body.action_mapping import (
    ActionSpace,
    BodyActionSpace,
    generate_action_mapping,
    map_action_to_body,
    validate_mapped_action,
)
from rosclaw.body.action_mapping.schema import MappingCompatibility


def canonical_body():
    return BodyActionSpace(
        body_id="synthetic_canonical29_only",
        representation="joint_position",
        joint_names=list(data.CANONICAL29),
        units=["rad"] * 29,
        joint_limits=data.SYNTHETIC_LIMITS,
        reference_frame="synthetic_fixture",
    )


def permuted_policy():
    return ActionSpace(
        representation="joint_position",
        names=list(data.PERMUTED29),
        units=["rad"] * 29,
        shape=[29],
    )


class ActionMappingContract(unittest.TestCase):
    def assert_values_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for index, (observed, truth) in enumerate(zip(actual, expected)):  # noqa: B905 - length equality is asserted above
            with self.subTest(index=index):
                self.assertAlmostEqual(observed, truth, delta=1e-12)

    def assert_rejected(self, mapping, values):
        mapped = map_action_to_body(values, mapping)
        report = validate_mapped_action(mapped)
        self.assertTrue(mapping.is_blocked())
        self.assertFalse(mapping.allow_partial)
        self.assertTrue(mapped.blocked)
        self.assertEqual(mapped.body_action_values, [])
        self.assertTrue(report["blocked"])
        self.assertTrue(mapped.block_reasons)

    def test_permuted_named29_exact(self):
        mapping = generate_action_mapping(permuted_policy(), canonical_body())
        self.assertEqual(mapping.compatibility, MappingCompatibility.EXACT)
        self.assertFalse(mapping.is_blocked())
        mapped = map_action_to_body(list(data.POLICY_ROW_A), mapping)
        self.assertFalse(mapped.blocked)
        self.assertEqual(mapped.body_joint_names, data.CANONICAL29)
        self.assertEqual(mapped.body_action_values, data.EXPECTED_ROW_A)
        self.assertEqual(mapped.reference_frame, "synthetic_fixture")
        report = validate_mapped_action(mapped, joint_limits=data.SYNTHETIC_LIMITS)
        self.assertFalse(report["blocked"])
        self.assertEqual(report["issues"], [])

    def test_permuted_named29_two_chunks(self):
        policy = permuted_policy()
        policy.is_chunked = True
        policy.chunk_size = 2
        policy.shape = [2, 29]
        mapping = generate_action_mapping(policy, canonical_body())
        mapped = map_action_to_body(
            data.POLICY_ROW_A + data.POLICY_ROW_B,
            mapping,
            chunk_size=2,
        )
        self.assertFalse(mapped.blocked)
        self.assertEqual(mapped.chunk_size, 2)
        self.assertEqual(mapped.body_joint_names, data.CANONICAL29)
        self.assertEqual(
            mapped.body_action_values,
            data.EXPECTED_ROW_A + data.EXPECTED_ROW_B,
        )
        report = validate_mapped_action(mapped, joint_limits=data.SYNTHETIC_LIMITS)
        self.assertFalse(report["blocked"])
        self.assertEqual(report["issues"], [])

    def test_degrees_to_radians_permuted_two_chunks(self):
        policy = ActionSpace(
            representation="joint_position",
            names=list(data.CONVERSION_POLICY_NAMES),
            units=["deg"] * 3,
            shape=[2, 3],
            is_chunked=True,
            chunk_size=2,
        )
        body = BodyActionSpace(
            body_id="synthetic_conversion",
            joint_names=list(data.CONVERSION_BODY_NAMES),
            units=["rad"] * 3,
        )
        mapping = generate_action_mapping(policy, body)
        self.assertEqual(mapping.compatibility, MappingCompatibility.CONVERTIBLE)
        mapped = map_action_to_body(list(data.DEGREE_POLICY_CHUNKS), mapping, chunk_size=2)
        self.assertFalse(mapped.blocked)
        self.assertEqual(mapped.body_joint_names, data.CONVERSION_BODY_NAMES)
        self.assert_values_close(mapped.body_action_values, data.EXPECTED_RADIAN_CHUNKS)
        self.assertFalse(validate_mapped_action(mapped)["blocked"])

    def test_radians_to_degrees_permuted_two_chunks(self):
        policy = ActionSpace(
            representation="joint_position",
            names=list(data.CONVERSION_POLICY_NAMES),
            units=["rad"] * 3,
            shape=[2, 3],
            is_chunked=True,
            chunk_size=2,
        )
        body = BodyActionSpace(
            body_id="synthetic_conversion",
            joint_names=list(data.CONVERSION_BODY_NAMES),
            units=["deg"] * 3,
        )
        mapping = generate_action_mapping(policy, body)
        self.assertEqual(mapping.compatibility, MappingCompatibility.CONVERTIBLE)
        mapped = map_action_to_body(list(data.RADIAN_POLICY_CHUNKS), mapping, chunk_size=2)
        self.assertFalse(mapped.blocked)
        self.assertEqual(mapped.body_joint_names, data.CONVERSION_BODY_NAMES)
        self.assert_values_close(mapped.body_action_values, data.EXPECTED_DEGREE_CHUNKS)
        self.assertFalse(validate_mapped_action(mapped)["blocked"])

    def test_27_outputs_missing_two_joints_block_by_default(self):
        policy = ActionSpace(
            representation="joint_position",
            names=list(data.POLICY27_NAMES),
            units=["rad"] * 27,
            shape=[27],
        )
        mapping = generate_action_mapping(policy, canonical_body())
        self.assertEqual(mapping.compatibility, MappingCompatibility.PARTIAL)
        missing = []
        for joint in mapping.joints:
            if joint.policy_index < 0:
                missing.append(joint.body_name)
        self.assertEqual(missing, data.MISSING27_NAMES)
        self.assertIn("Partial mapping is not allowed by default", mapping.block_reasons)
        self.assert_rejected(mapping, list(data.POLICY27_VALUES))

    def test_wrong_units_block(self):
        policy = permuted_policy()
        policy.units[0] = "N"
        mapping = generate_action_mapping(policy, canonical_body())
        self.assertEqual(mapping.compatibility, MappingCompatibility.INCOMPATIBLE)
        self.assert_rejected(mapping, list(data.POLICY_ROW_A))

    def test_wrong_representation_blocks(self):
        for representation, compatibility in (
            ("joint_velocity", MappingCompatibility.INCOMPATIBLE),
            ("cartesian_pose", MappingCompatibility.UNKNOWN),
        ):
            with self.subTest(representation=representation):
                policy = permuted_policy()
                policy.representation = representation
                mapping = generate_action_mapping(policy, canonical_body())
                self.assertEqual(mapping.compatibility, compatibility)
                self.assert_rejected(mapping, list(data.POLICY_ROW_A))

    def test_malformed_flat_lengths_block(self):
        mapping = generate_action_mapping(permuted_policy(), canonical_body())
        for values, chunks in (
            (data.POLICY_ROW_A[:-1], None),
            (data.POLICY_ROW_A + [0.3], None),
            (data.POLICY_ROW_A + data.POLICY_ROW_B[:-1], 2),
        ):
            with self.subTest(length=len(values), chunks=chunks):
                mapped = map_action_to_body(values, mapping, chunk_size=chunks)
                self.assertTrue(mapped.blocked)
                self.assertEqual(mapped.body_action_values, [])
                self.assertIn("Action length mismatch", mapped.block_reasons[0])
                self.assertTrue(validate_mapped_action(mapped)["blocked"])

    def test_inconsistent_declared_shape_must_reject(self):
        # Known static gap: generate/map ignore shape metadata. Do not normalize
        # this invalid declaration or weaken the rejection contract to pass.
        policy = permuted_policy()
        policy.shape = [28]
        mapping = generate_action_mapping(policy, canonical_body())
        mapped = map_action_to_body(list(data.POLICY_ROW_A), mapping)
        report = validate_mapped_action(mapped)
        self.assertTrue(report["blocked"], "29 names/values with declared shape [28] must reject")

    def test_duplicate_policy_name_must_reject(self):
        # All 29 body joints still match; the extra duplicate must not silently
        # overwrite an earlier index. Current dict construction suggests failure.
        policy = permuted_policy()
        policy.names.append("right_wrist_yaw_joint")
        policy.units.append("rad")
        policy.shape = [30]
        mapping = generate_action_mapping(policy, canonical_body())
        mapped = map_action_to_body(data.POLICY_ROW_A + [0.73], mapping)
        report = validate_mapped_action(mapped)
        self.assertTrue(report["blocked"], "duplicate policy names must reject")

    def test_foreign_policy_name_blocks(self):
        policy = permuted_policy()
        policy.names[0] = "foreign_synthetic_joint"
        mapping = generate_action_mapping(policy, canonical_body())
        self.assertEqual(mapping.compatibility, MappingCompatibility.PARTIAL)
        self.assert_rejected(mapping, list(data.POLICY_ROW_A))

    def test_finite_out_of_range_in_later_chunk_blocks_validation(self):
        values = data.POLICY_ROW_A + data.POLICY_ROW_B
        values[29 + data.OUT_OF_RANGE_POLICY_INDEX] = data.OUT_OF_RANGE_VALUE
        self.assertTrue(math.isfinite(data.OUT_OF_RANGE_VALUE))
        mapping = generate_action_mapping(permuted_policy(), canonical_body())
        mapped = map_action_to_body(values, mapping, chunk_size=2)
        self.assertFalse(mapped.blocked)  # mapping does not itself enforce limits
        self.assertEqual(
            mapped.body_action_values[data.OUT_OF_RANGE_BODY_INDEX_LATER_CHUNK],
            data.OUT_OF_RANGE_VALUE,
        )
        report = validate_mapped_action(mapped, joint_limits=data.SYNTHETIC_LIMITS)
        self.assertTrue(report["blocked"])
        self.assertEqual(len(report["issues"]), 1)
        self.assertIn("left_hip_pitch_joint chunk 1 above upper limit", report["issues"][0])

    def test_nonfinite_first_chunk_blocks_validation(self):
        mapping = generate_action_mapping(permuted_policy(), canonical_body())
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid):
                values = list(data.POLICY_ROW_A)
                values[28] = invalid
                mapped = map_action_to_body(values, mapping)
                report = validate_mapped_action(mapped)
                self.assertTrue(report["blocked"])
                self.assertIn("Invalid value at index 0", report["issues"][0])

    def test_nonfinite_later_chunk_must_return_blocked_report(self):
        # Intentionally uncaught: a validator IndexError is an implementation
        # failure, not an honest blocked report and not a successful rejection.
        mapping = generate_action_mapping(permuted_policy(), canonical_body())
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid):
                values = data.POLICY_ROW_A + data.POLICY_ROW_B
                values[57] = invalid  # second row, left_hip_pitch -> body index 29
                mapped = map_action_to_body(values, mapping, chunk_size=2)
                self.assertFalse(mapped.blocked)
                self.assertFalse(math.isfinite(mapped.body_action_values[29]))
                report = validate_mapped_action(mapped, joint_limits=data.SYNTHETIC_LIMITS)
                self.assertTrue(report["blocked"])
                self.assertTrue(report["issues"])
                self.assertIn("Invalid value at index 29", report["issues"][0])
