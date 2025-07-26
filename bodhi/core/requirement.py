"""
🧘 Requirement Primitive - Task Requirements
===========================================

Defines requirement structures for tasks.
"""

from typing import Set, Dict, Any, Optional
from dataclasses import dataclass, field
from .capability import CapabilityType


@dataclass(frozen=True)
class Requirement:
    """Represents a capability requirement for task execution"""
    capability_type: CapabilityType
    domain: Optional[str] = None
    min_confidence: float = 0.7
    required: bool = True
    alternatives: Set[CapabilityType] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("Minimum confidence must be between 0.0 and 1.0") 