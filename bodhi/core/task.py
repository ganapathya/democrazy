"""
🧘 Task Primitive - Unit of Work
===============================

The Task represents a unit of work to be executed by agents in the Bodhi framework.
Tasks specify what needs to be done, required capabilities, context, and execution parameters.

Key Features:
- Intent-based task specification
- Capability requirements
- Rich context and metadata
- Priority and urgency handling
- Decomposition and composition support
- Execution tracking and validation
"""

import uuid
import json
from typing import Dict, List, Set, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import logging

from .capability import CapabilityType
from ..utils.exceptions import BodhiError, TaskValidationError

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status states"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DECOMPOSED = "decomposed"  # Task was broken into subtasks


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class TaskType(Enum):
    """Types of tasks"""
    QUERY = "query"  # Information retrieval
    COMMAND = "command"  # Action execution
    ANALYSIS = "analysis"  # Data analysis
    GENERATION = "generation"  # Content creation
    TRANSFORMATION = "transformation"  # Data transformation
    COMMUNICATION = "communication"  # Agent-to-agent communication
    LEARNING = "learning"  # Learning and adaptation
    ORCHESTRATION = "orchestration"  # Complex multi-step process


@dataclass(eq=True)
class TaskRequirement:
    """Represents a specific requirement for task execution"""
    capability_type: CapabilityType
    domain: Optional[str] = None
    min_confidence: float = 0.7
    required: bool = True
    alternatives: Set[CapabilityType] = field(default_factory=set, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def __post_init__(self):
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("Minimum confidence must be between 0.0 and 1.0")
    
    def __hash__(self):
        return hash((self.capability_type, self.domain, self.min_confidence, self.required))


@dataclass
class TaskContext:
    """Rich context information for task execution"""
    # Input data and parameters
    input_data: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Environmental context
    environment: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Execution preferences
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Security and access control
    permissions: Set[str] = field(default_factory=set)
    sandbox_config: Dict[str, Any] = field(default_factory=dict)
    
    # Communication context
    communication_channels: List[str] = field(default_factory=list)
    collaboration_agents: Set[str] = field(default_factory=set)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            'input_data': self.input_data,
            'parameters': self.parameters,
            'environment': self.environment,
            'constraints': self.constraints,
            'preferences': self.preferences,
            'permissions': list(self.permissions),
            'sandbox_config': self.sandbox_config,
            'communication_channels': self.communication_channels,
            'collaboration_agents': list(self.collaboration_agents),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskContext':
        """Create context from dictionary"""
        return cls(
            input_data=data.get('input_data', {}),
            parameters=data.get('parameters', {}),
            environment=data.get('environment', {}),
            constraints=data.get('constraints', {}),
            preferences=data.get('preferences', {}),
            permissions=set(data.get('permissions', [])),
            sandbox_config=data.get('sandbox_config', {}),
            communication_channels=data.get('communication_channels', []),
            collaboration_agents=set(data.get('collaboration_agents', [])),
            metadata=data.get('metadata', {})
        )


class Task:
    """
    Core Task class - represents a unit of work to be executed by agents
    
    A Task encapsulates:
    - What needs to be done (intent)
    - How it should be done (requirements, context)
    - When it should be done (priority, deadlines)
    - Who can do it (capability requirements)
    - Success criteria and validation
    """
    
    def __init__(
        self,
        intent: str,
        task_type: TaskType = TaskType.QUERY,
        context: Optional[TaskContext] = None,
        requirements: Optional[Set[TaskRequirement]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_execution_time: Optional[int] = None,
        expected_output_format: Optional[str] = None,
        success_criteria: Optional[Dict[str, Any]] = None,
        parent_task_id: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        self.id = task_id or str(uuid.uuid4())
        self.intent = intent
        self.task_type = task_type
        self.context = context or TaskContext()
        self.requirements = requirements or set()
        self.priority = priority
        self.max_execution_time = max_execution_time or 300  # Default 5 minutes
        self.expected_output_format = expected_output_format
        self.success_criteria = success_criteria or {}
        self.parent_task_id = parent_task_id
        
        # Runtime state
        self.status = TaskStatus.PENDING
        self.assigned_agent_id: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        
        # Execution tracking
        self.execution_attempts: List[Dict[str, Any]] = []
        self.subtasks: List[str] = []  # IDs of decomposed subtasks
        self.dependencies: Set[str] = set()  # Task IDs this task depends on
        
        # Results and feedback
        self.results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.feedback: List[Dict[str, Any]] = []
        
        # Validation
        self._validate_task()
        
        logger.info(f"📝 Task created: {self.intent} (ID: {self.id})")
    
    @property
    def required_capabilities(self) -> Set[CapabilityType]:
        """Get set of required capability types"""
        return {req.capability_type for req in self.requirements if req.required}
    
    @property
    def optional_capabilities(self) -> Set[CapabilityType]:
        """Get set of optional capability types"""
        return {req.capability_type for req in self.requirements if not req.required}
    
    @property
    def is_pending(self) -> bool:
        """Check if task is pending execution"""
        return self.status == TaskStatus.PENDING
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed"""
        return self.status == TaskStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if task has failed"""
        return self.status == TaskStatus.FAILED
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get task execution time in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def _validate_task(self):
        """Validate task parameters"""
        if not self.intent.strip():
            raise TaskValidationError("Task intent cannot be empty")
        
        if self.max_execution_time <= 0:
            raise TaskValidationError("Max execution time must be positive")
        
        # Validate requirements
        for req in self.requirements:
            if req.min_confidence < 0.0 or req.min_confidence > 1.0:
                raise TaskValidationError(f"Invalid confidence level in requirement: {req.min_confidence}")
    
    def add_requirement(self, requirement: TaskRequirement):
        """Add a capability requirement to the task"""
        self.requirements.add(requirement)
        self.updated_at = datetime.now(timezone.utc)
        logger.debug(f"Added requirement {requirement.capability_type.value} to task {self.id}")
    
    def remove_requirement(self, capability_type: CapabilityType, domain: Optional[str] = None):
        """Remove a capability requirement from the task"""
        to_remove = None
        for req in self.requirements:
            if req.capability_type == capability_type and (domain is None or req.domain == domain):
                to_remove = req
                break
        
        if to_remove:
            self.requirements.remove(to_remove)
            self.updated_at = datetime.now(timezone.utc)
            logger.debug(f"Removed requirement {capability_type.value} from task {self.id}")
    
    def update_context(self, context_updates: Dict[str, Any]):
        """Update task context with new information"""
        for key, value in context_updates.items():
            if hasattr(self.context, key):
                current_value = getattr(self.context, key)
                if isinstance(current_value, dict):
                    current_value.update(value)
                elif isinstance(current_value, set):
                    current_value.update(value)
                elif isinstance(current_value, list):
                    current_value.extend(value)
                else:
                    setattr(self.context, key, value)
        
        self.updated_at = datetime.now(timezone.utc)
        logger.debug(f"Updated context for task {self.id}")
    
    def assign_to_agent(self, agent_id: str):
        """Assign task to a specific agent"""
        if self.status != TaskStatus.PENDING:
            raise TaskValidationError(f"Cannot assign task in status {self.status.value}")
        
        self.assigned_agent_id = agent_id
        self.status = TaskStatus.ASSIGNED
        self.updated_at = datetime.now(timezone.utc)
        
        logger.info(f"📌 Task {self.id} assigned to agent {agent_id}")
    
    def start_execution(self):
        """Mark task as started"""
        if self.status != TaskStatus.ASSIGNED:
            raise TaskValidationError(f"Cannot start task in status {self.status.value}")
        
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc)
        self.updated_at = self.started_at
        
        logger.info(f"▶️ Task {self.id} execution started")
    
    def complete_task(self, result: Dict[str, Any]):
        """Mark task as completed with result"""
        if self.status != TaskStatus.IN_PROGRESS:
            raise TaskValidationError(f"Cannot complete task in status {self.status.value}")
        
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = self.completed_at
        
        # Store result
        result_entry = {
            'timestamp': self.completed_at.isoformat(),
            'agent_id': self.assigned_agent_id,
            'result': result,
            'execution_time': self.execution_time
        }
        self.results.append(result_entry)
        
        logger.info(f"✅ Task {self.id} completed successfully")
    
    def fail_task(self, error: str, error_details: Optional[Dict[str, Any]] = None):
        """Mark task as failed with error information"""
        if self.status == TaskStatus.COMPLETED:
            raise TaskValidationError("Cannot fail a completed task")
        
        self.status = TaskStatus.FAILED
        self.updated_at = datetime.now(timezone.utc)
        
        # Store error
        error_entry = {
            'timestamp': self.updated_at.isoformat(),
            'agent_id': self.assigned_agent_id,
            'error': error,
            'details': error_details or {},
            'attempt_number': len(self.execution_attempts) + 1
        }
        self.errors.append(error_entry)
        
        logger.error(f"❌ Task {self.id} failed: {error}")
    
    def cancel_task(self, reason: str = "Cancelled by user"):
        """Cancel task execution"""
        if self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            raise TaskValidationError(f"Cannot cancel task in status {self.status.value}")
        
        self.status = TaskStatus.CANCELLED
        self.updated_at = datetime.now(timezone.utc)
        
        logger.info(f"🚫 Task {self.id} cancelled: {reason}")
    
    def add_subtask(self, subtask_id: str):
        """Add a subtask (for task decomposition)"""
        self.subtasks.append(subtask_id)
        self.updated_at = datetime.now(timezone.utc)
        
        # If this is the first subtask, mark as decomposed
        if len(self.subtasks) == 1 and self.status == TaskStatus.PENDING:
            self.status = TaskStatus.DECOMPOSED
        
        logger.debug(f"Added subtask {subtask_id} to task {self.id}")
    
    def add_dependency(self, task_id: str):
        """Add a task dependency"""
        self.dependencies.add(task_id)
        self.updated_at = datetime.now(timezone.utc)
        logger.debug(f"Added dependency {task_id} to task {self.id}")
    
    def remove_dependency(self, task_id: str):
        """Remove a task dependency"""
        self.dependencies.discard(task_id)
        self.updated_at = datetime.now(timezone.utc)
        logger.debug(f"Removed dependency {task_id} from task {self.id}")
    
    def add_feedback(self, feedback: Dict[str, Any]):
        """Add feedback about task execution"""
        feedback_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'feedback': feedback
        }
        self.feedback.append(feedback_entry)
        self.updated_at = datetime.now(timezone.utc)
        
        logger.debug(f"Added feedback to task {self.id}")
    
    def validate_success(self) -> bool:
        """Validate if task meets success criteria"""
        if not self.success_criteria:
            return self.status == TaskStatus.COMPLETED
        
        if not self.is_completed:
            return False
        
        # Check success criteria against results
        if not self.results:
            return False
        
        latest_result = self.results[-1]['result']
        
        for criterion, expected_value in self.success_criteria.items():
            if criterion not in latest_result:
                return False
            
            actual_value = latest_result[criterion]
            
            # Simple equality check (can be extended for complex criteria)
            if actual_value != expected_value:
                return False
        
        return True
    
    def can_execute_now(self) -> bool:
        """Check if task can be executed now (dependencies met)"""
        if self.status != TaskStatus.PENDING:
            return False
        
        # For now, assume all dependencies are resolved
        # In a full implementation, this would check dependency status
        return True
    
    def estimate_complexity(self) -> float:
        """Estimate task complexity (0.0 to 1.0)"""
        complexity = 0.0
        
        # Base complexity from requirements
        complexity += len(self.requirements) * 0.1
        
        # Add complexity for each capability type
        capability_weights = {
            CapabilityType.NLP_PROCESSING: 0.2,
            CapabilityType.DATABASE_CONNECTOR: 0.3,
            CapabilityType.API_CONNECTOR: 0.2,
            CapabilityType.DATA_ANALYSIS: 0.4,
            CapabilityType.CODE_GENERATION: 0.5,
            CapabilityType.REASONING: 0.6,
            CapabilityType.PLANNING: 0.7,
            CapabilityType.LEARNING: 0.4,
            CapabilityType.COMMUNICATION: 0.2,
            CapabilityType.TOOL_CREATION: 0.8
        }
        
        for req in self.requirements:
            complexity += capability_weights.get(req.capability_type, 0.3)
        
        # Context complexity
        context_factors = [
            len(self.context.input_data),
            len(self.context.parameters),
            len(self.context.constraints),
            len(self.dependencies)
        ]
        complexity += sum(min(factor * 0.05, 0.2) for factor in context_factors)
        
        return min(complexity, 1.0)
    
    def serialize(self) -> str:
        """Serialize task to JSON string"""
        task_data = {
            'id': self.id,
            'intent': self.intent,
            'task_type': self.task_type.value,
            'context': self.context.to_dict(),
            'requirements': [
                {
                    'capability_type': req.capability_type.value,
                    'domain': req.domain,
                    'min_confidence': req.min_confidence,
                    'required': req.required,
                    'alternatives': [alt.value for alt in req.alternatives],
                    'metadata': req.metadata
                }
                for req in self.requirements
            ],
            'priority': self.priority.value,
            'max_execution_time': self.max_execution_time,
            'expected_output_format': self.expected_output_format,
            'success_criteria': self.success_criteria,
            'parent_task_id': self.parent_task_id,
            'status': self.status.value,
            'assigned_agent_id': self.assigned_agent_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'execution_attempts': self.execution_attempts,
            'subtasks': self.subtasks,
            'dependencies': list(self.dependencies),
            'results': self.results,
            'errors': self.errors,
            'feedback': self.feedback,
            'serialization_timestamp': datetime.now(timezone.utc).isoformat(),
            'framework_version': '0.1.0'
        }
        
        return json.dumps(task_data, indent=2)
    
    @classmethod
    def deserialize(cls, serialized_data: str) -> 'Task':
        """Deserialize task from JSON string"""
        try:
            data = json.loads(serialized_data)
            
            # Reconstruct context
            context = TaskContext.from_dict(data['context'])
            
            # Reconstruct requirements
            requirements = set()
            for req_data in data.get('requirements', []):
                requirements.add(TaskRequirement(
                    capability_type=CapabilityType(req_data['capability_type']),
                    domain=req_data.get('domain'),
                    min_confidence=req_data.get('min_confidence', 0.7),
                    required=req_data.get('required', True),
                    alternatives=set(CapabilityType(alt) for alt in req_data.get('alternatives', [])),
                    metadata=req_data.get('metadata', {})
                ))
            
            # Create task
            task = cls(
                intent=data['intent'],
                task_type=TaskType(data['task_type']),
                context=context,
                requirements=requirements,
                priority=TaskPriority(data['priority']),
                max_execution_time=data['max_execution_time'],
                expected_output_format=data.get('expected_output_format'),
                success_criteria=data.get('success_criteria', {}),
                parent_task_id=data.get('parent_task_id'),
                task_id=data['id']
            )
            
            # Restore state
            task.status = TaskStatus(data['status'])
            task.assigned_agent_id = data.get('assigned_agent_id')
            task.created_at = datetime.fromisoformat(data['created_at'])
            task.updated_at = datetime.fromisoformat(data['updated_at'])
            
            if data.get('started_at'):
                task.started_at = datetime.fromisoformat(data['started_at'])
            if data.get('completed_at'):
                task.completed_at = datetime.fromisoformat(data['completed_at'])
            
            task.execution_attempts = data.get('execution_attempts', [])
            task.subtasks = data.get('subtasks', [])
            task.dependencies = set(data.get('dependencies', []))
            task.results = data.get('results', [])
            task.errors = data.get('errors', [])
            task.feedback = data.get('feedback', [])
            
            logger.info(f"🔄 Task {task.id} deserialized successfully")
            return task
            
        except Exception as e:
            raise BodhiError(f"Failed to deserialize task: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive task status"""
        return {
            'id': self.id,
            'intent': self.intent,
            'task_type': self.task_type.value,
            'status': self.status.value,
            'priority': self.priority.value,
            'assigned_agent_id': self.assigned_agent_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'execution_time': self.execution_time,
            'complexity': self.estimate_complexity(),
            'requirements_count': len(self.requirements),
            'subtasks_count': len(self.subtasks),
            'dependencies_count': len(self.dependencies),
            'results_count': len(self.results),
            'errors_count': len(self.errors),
            'can_execute': self.can_execute_now(),
            'meets_success_criteria': self.validate_success() if self.is_completed else None
        }
    
    def __str__(self) -> str:
        return f"Task({self.intent[:50]}{'...' if len(self.intent) > 50 else ''}, {self.status.value})"
    
    def __repr__(self) -> str:
        return f"Task(id='{self.id}', intent='{self.intent}', status='{self.status.value}')"


# Utility functions for task creation

def create_nlp_task(query: str, context: Optional[Dict[str, Any]] = None) -> Task:
    """Create a task for natural language processing"""
    task_context = TaskContext()
    if context:
        task_context.input_data.update(context)
    
    requirements = {
        TaskRequirement(
            capability_type=CapabilityType.NLP_PROCESSING,
            domain="general",
            min_confidence=0.7,
            required=True
        )
    }
    
    return Task(
        intent=f"Process natural language query: {query}",
        task_type=TaskType.QUERY,
        context=task_context,
        requirements=requirements,
        priority=TaskPriority.NORMAL
    )


def create_sql_task(natural_query: str, database_context: Optional[Dict[str, Any]] = None) -> Task:
    """Create a task for NLP to SQL conversion"""
    task_context = TaskContext()
    task_context.input_data['natural_query'] = natural_query
    if database_context:
        task_context.environment.update(database_context)
    
    requirements = {
        TaskRequirement(
            capability_type=CapabilityType.NLP_PROCESSING,
            domain="sql",
            min_confidence=0.8,
            required=True
        ),
        TaskRequirement(
            capability_type=CapabilityType.DATABASE_CONNECTOR,
            min_confidence=0.7,
            required=True
        )
    }
    
    return Task(
        intent=f"Convert natural language to SQL: {natural_query}",
        task_type=TaskType.TRANSFORMATION,
        context=task_context,
        requirements=requirements,
        priority=TaskPriority.NORMAL,
        expected_output_format="sql_query"
    )


def create_analysis_task(data_description: str, analysis_type: str = "general") -> Task:
    """Create a task for data analysis"""
    task_context = TaskContext()
    task_context.input_data['data_description'] = data_description
    task_context.parameters['analysis_type'] = analysis_type
    
    requirements = {
        TaskRequirement(
            capability_type=CapabilityType.DATA_ANALYSIS,
            domain=analysis_type,
            min_confidence=0.8,
            required=True
        )
    }
    
    return Task(
        intent=f"Perform {analysis_type} analysis on: {data_description}",
        task_type=TaskType.ANALYSIS,
        context=task_context,
        requirements=requirements,
        priority=TaskPriority.NORMAL
    ) 