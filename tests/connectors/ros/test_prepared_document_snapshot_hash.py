"""Static literal encoding preserves every original canonical hash byte."""

from copy import deepcopy

import pytest

from rosclaw.connectors.ros.intelligence.system_model import (
    RosSystemModel,
    _encoded_prepared_document,
    _snapshot_content_hash,
)
from rosclaw.contracts.common import content_hash
from tests.connectors.ros.test_navigation_body_workspace import source


@pytest.mark.parametrize(
    "document",
    ['{"original":"bytes"}', 'quote" slash\\ newline\n汉字', "\x00\t\r", "a" * 300000, ""],
)
def test_streamed_hash_matches_existing_canonical_encoding_and_all_mutations(document):
    payload = {
        "observations": {
            "prepared_collision_assets": {
                "document": document,
                "sha256": "a" * 64,
                "extra": {"document": "different", "unicode": "汉字"},
            },
            "other": [True, None, 1.5],
        },
        "clock": "current",
        "graph": {"nested": ["}", "document", 0]},
    }
    assert _snapshot_content_hash(payload) == content_hash("rossnap", payload)
    for path, value in [
        (("clock",), "later"),
        (("observations", "other"), [False]),
        (("observations", "prepared_collision_assets", "sha256"), "b" * 64),
        (("observations", "prepared_collision_assets", "document"), document + "changed"),
    ]:
        changed = deepcopy(payload)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert _snapshot_content_hash(changed) == content_hash("rossnap", changed)
        assert _snapshot_content_hash(changed) != _snapshot_content_hash(payload)


def test_model_resealing_and_old_hash_validation_are_unchanged():
    model, *_ = source()
    model.observations["prepared_collision_assets"] = {
        "schema_version": "rosclaw.prepared_collision_asset_observation.v1",
        "document": '{"literal":"quote\\n汉字"}',
        "source": "/assets/collision-asset-originals.json",
        "sha256": "a" * 64,
    }
    model.seal()
    payload = model.to_dict()
    payload.pop("snapshot_hash")
    payload.pop("snapshot_id")
    assert model.snapshot_hash == content_hash("rossnap", payload)
    assert RosSystemModel.from_dict(model.to_dict()).snapshot_hash == model.snapshot_hash
    changed = model.to_dict()
    changed["observations"]["prepared_collision_assets"]["document"] += "changed"
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        RosSystemModel.from_dict(changed)


def test_literal_encoding_cache_is_bounded_and_does_not_cache_live_fields():
    _encoded_prepared_document.cache_clear()
    for i in range(4):
        payload = {"observations": {"prepared_collision_assets": {"document": str(i)}}}
        assert _snapshot_content_hash(payload) == content_hash("rossnap", payload)
    assert _encoded_prepared_document.cache_info().currsize == 2
    _encoded_prepared_document.cache_clear()
