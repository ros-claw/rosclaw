"""Body action mapping for LeRobot policy proposals."""

from rosclaw.body.action_mapping.mapper import (
    generate_action_mapping,
    map_action_proposal_to_body,
    map_action_to_body,
)
from rosclaw.body.action_mapping.report import (
    format_mapped_action_report,
    format_mapping_report,
    format_mapping_text,
)
from rosclaw.body.action_mapping.resolver import resolve_body_action_space
from rosclaw.body.action_mapping.schema import (
    ActionMapping,
    ActionSpace,
    BodyActionSpace,
    JointMapping,
    MappedAction,
    MappingCompatibility,
)
from rosclaw.body.action_mapping.validator import (
    validate_action_mapping,
    validate_mapped_action,
)

__all__ = [
    "ActionMapping",
    "ActionSpace",
    "BodyActionSpace",
    "JointMapping",
    "MappedAction",
    "MappingCompatibility",
    "generate_action_mapping",
    "map_action_to_body",
    "map_action_proposal_to_body",
    "resolve_body_action_space",
    "validate_action_mapping",
    "validate_mapped_action",
    "format_mapping_report",
    "format_mapping_text",
    "format_mapped_action_report",
]
