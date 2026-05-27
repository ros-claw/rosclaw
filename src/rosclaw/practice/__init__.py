"""
ROSClaw Practice - Timeline Grounding Engine

MCAP-based black box recording and replay.
Captures robot execution traces for analysis and learning.
"""

from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.practice.timeline import UnifiedTimeline, TimelineChannel, TimelineEntry

__all__ = ["PracticeRecorder", "UnifiedTimeline", "TimelineChannel", "TimelineEntry"]
