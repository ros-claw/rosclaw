"""Compact ROS context for existing ContextCompiler source protocols."""

from .compiler import RosCapabilitySourceAdapter, RosSelfAugmentingSource, compile_agent_summary

__all__ = ["RosCapabilitySourceAdapter", "RosSelfAugmentingSource", "compile_agent_summary"]
