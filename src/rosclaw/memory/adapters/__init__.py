"""Task distillation adapters (数据库优化v3 §4).

The generic extractors in ``distill.py`` understand generic
``failure_event``/verified-gesture shapes.  Task adapters add the
task-specific semantics the generic layer cannot know — e.g. the RH56
RPS stress protocol where ``rps.stress.round.resolved result=invalid``
IS a failure even when no explicit failure event exists, where episode
quality is a verified-rate distribution rather than a flat SUCCESS, and
where temperature rise is an OBSERVED correlation, never a limit.
"""

from .base import TaskDistillationAdapter
from .registry import adapter_for, register_adapter

__all__ = ["TaskDistillationAdapter", "adapter_for", "register_adapter"]
