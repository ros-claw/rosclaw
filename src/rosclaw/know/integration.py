"""Know integration into Runtime decision flow."""
from rosclaw.know.knowledge_base import KnowledgeBase


class KnowIntegration:
    def query_before_decision(self, robot_id, task):
        kb = KnowledgeBase()
        result = kb.query(robot=robot_id, task=task)
        return result
