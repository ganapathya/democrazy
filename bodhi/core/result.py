"""
🧘 Result Primitive - Task Execution Results
===========================================

The Result classes represent the outcome of task execution by agents.
They provide structured data about success/failure, output data, and metadata.
"""

import uuid
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class ResultStatus(Enum):
    """Result status enumeration"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of task execution by an agent"""
    
    # Core result data
    success: bool
    agent_id: str
    task_id: str
    
    # Output data
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    
    # Execution metadata
    execution_time: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Quality metrics
    confidence: float = 1.0
    quality_score: Optional[float] = None
    validation_passed: bool = True
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)  # Files, outputs, etc.
    
    @property
    def status(self) -> ResultStatus:
        """Get result status based on success and other factors"""
        if self.success:
            return ResultStatus.SUCCESS
        elif self.error and "timeout" in self.error.lower():
            return ResultStatus.TIMEOUT
        elif self.error and "cancel" in self.error.lower():
            return ResultStatus.CANCELLED
        elif self.data:  # Has some data despite failure
            return ResultStatus.PARTIAL
        else:
            return ResultStatus.FAILURE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            'result_id': self.result_id,
            'success': self.success,
            'status': self.status.value,
            'agent_id': self.agent_id,
            'task_id': self.task_id,
            'data': self.data,
            'message': self.message,
            'error': self.error,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'quality_score': self.quality_score,
            'validation_passed': self.validation_passed,
            'metadata': self.metadata,
            'artifacts': self.artifacts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """Create result from dictionary"""
        return cls(
            result_id=data.get('result_id', str(uuid.uuid4())),
            success=data['success'],
            agent_id=data['agent_id'],
            task_id=data['task_id'],
            data=data.get('data'),
            message=data.get('message'),
            error=data.get('error'),
            execution_time=data.get('execution_time'),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else datetime.now(timezone.utc),
            confidence=data.get('confidence', 1.0),
            quality_score=data.get('quality_score'),
            validation_passed=data.get('validation_passed', True),
            metadata=data.get('metadata', {}),
            artifacts=data.get('artifacts', {})
        )


# Type alias for backward compatibility
Result = ExecutionResult 