"""
🧘 Capability Primitive - Agent Capabilities
===========================================

Defines the capability types and structures used throughout the Bodhi framework.
This is imported by agent.py but defined here for better organization.
"""

from enum import Enum
from typing import Set, Dict, Any
from dataclasses import dataclass, field


class CapabilityType(Enum):
    """Types of capabilities an agent can possess"""
    NLP_PROCESSING = "nlp_processing"
    DATABASE_CONNECTOR = "database_connector"
    API_CONNECTOR = "api_connector"
    DATA_ANALYSIS = "data_analysis"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    PLANNING = "planning"
    LEARNING = "learning"
    COMMUNICATION = "communication"
    TOOL_CREATION = "tool_creation"


@dataclass(eq=True)
class Capability:
    """Represents a specific capability an agent possesses"""
    type: CapabilityType
    domain: str
    confidence: float = 1.0
    prerequisites: Set[str] = field(default_factory=set, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def __hash__(self):
        return hash((self.type, self.domain, self.confidence)) 