"""
ROSClaw e-URDF Parser - Physical DNA Registry

Parses extended URDF files with semantic annotations for LLM grounding.
"""

from rosclaw.e_urdf.parser import EURDFParser, JointSpec, LinkSpec, RobotModel

# Backward-compatible alias for documentation
EUrdfParser = EURDFParser

__all__ = ["EURDFParser", "EUrdfParser", "RobotModel", "JointSpec", "LinkSpec"]
