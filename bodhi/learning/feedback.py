"""
🧘 Feedback System for Agent Learning
====================================

Provides feedback collection and processing for agent learning and adaptation.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class FeedbackEntry:
    """Single feedback entry for agent learning"""
    agent_id: str
    task_id: str
    feedback_type: str  # 'performance', 'quality', 'user', etc.
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'agent_id': self.agent_id,
            'task_id': self.task_id,
            'feedback_type': self.feedback_type,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }


class FeedbackCollector:
    """Collects and processes feedback for agent learning"""
    
    def __init__(self):
        self.feedback_entries: List[FeedbackEntry] = []
    
    def add_feedback(self, entry: FeedbackEntry):
        """Add feedback entry"""
        self.feedback_entries.append(entry)
    
    def get_feedback_for_agent(self, agent_id: str) -> List[FeedbackEntry]:
        """Get all feedback for specific agent"""
        return [entry for entry in self.feedback_entries if entry.agent_id == agent_id] 