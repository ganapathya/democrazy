"""
🧘 Agent Primitive - Core Intelligence Unit
==========================================

The Agent is the fundamental unit of intelligence in the Bodhi framework.
Each agent is serializable, can communicate with other agents via A2A,
and executes within a controlled sandbox environment.

Key Features:
- Rich DNA specification (capabilities, tools, knowledge, goals)
- Serializable for persistence and distribution
- A2A communication protocol support
- Sandboxed execution environment
- Learning and adaptation capabilities
- MCP-compliant tool integration
"""

import asyncio
import json
import uuid
from typing import Dict, List, Set, Any, Optional, Protocol, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import logging

# Framework imports (will be implemented)
from ..utils.exceptions import BodhiError, AgentExecutionError
from ..security.sandbox import AgentSandbox
from ..communication.a2a import A2AProtocol, AgentMessage
from ..tools.mcp import MCPTool
from ..learning.feedback import FeedbackEntry
from .capability import CapabilityType, Capability

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    LEARNING = "learning"
    COMMUNICATING = "communicating"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


@dataclass
class AgentDNA:
    """
    Genetic blueprint for agent creation and evolution
    
    The DNA contains all the information needed to reconstruct
    an agent with its capabilities, knowledge, and behavior patterns.
    """
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "BodhiAgent"
    version: str = "1.0.0"
    
    # Core characteristics
    capabilities: Set[Capability] = field(default_factory=set)
    knowledge_domains: List[str] = field(default_factory=list)
    tools: Set[str] = field(default_factory=set)
    goals: List[str] = field(default_factory=list)
    
    # Behavioral traits
    learning_rate: float = 0.1
    collaboration_preference: float = 0.5
    risk_tolerance: float = 0.3
    creativity_factor: float = 0.4
    
    # Operational parameters
    max_memory_size: int = 10000
    max_execution_time: int = 300  # seconds
    max_concurrent_tasks: int = 5
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    creator_agent_id: Optional[str] = None
    generation: int = 1
    
    # Serialization metadata
    schema_version: str = "1.0"
    
    def __post_init__(self):
        """Validate DNA parameters"""
        if not 0.0 <= self.learning_rate <= 1.0:
            raise ValueError("Learning rate must be between 0.0 and 1.0")
        if not 0.0 <= self.collaboration_preference <= 1.0:
            raise ValueError("Collaboration preference must be between 0.0 and 1.0")
        if not 0.0 <= self.risk_tolerance <= 1.0:
            raise ValueError("Risk tolerance must be between 0.0 and 1.0")
        if not 0.0 <= self.creativity_factor <= 1.0:
            raise ValueError("Creativity factor must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DNA to dictionary for serialization"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'version': self.version,
            'capabilities': [
                {
                    'type': cap.type.value,
                    'domain': cap.domain,
                    'confidence': cap.confidence,
                    'prerequisites': list(cap.prerequisites),
                    'metadata': cap.metadata
                }
                for cap in self.capabilities
            ],
            'knowledge_domains': self.knowledge_domains,
            'tools': list(self.tools),
            'goals': self.goals,
            'learning_rate': self.learning_rate,
            'collaboration_preference': self.collaboration_preference,
            'risk_tolerance': self.risk_tolerance,
            'creativity_factor': self.creativity_factor,
            'max_memory_size': self.max_memory_size,
            'max_execution_time': self.max_execution_time,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'created_at': self.created_at.isoformat(),
            'creator_agent_id': self.creator_agent_id,
            'generation': self.generation,
            'schema_version': self.schema_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentDNA':
        """Create DNA from dictionary (deserialization)"""
        capabilities = set()
        for cap_data in data.get('capabilities', []):
            capabilities.add(Capability(
                type=CapabilityType(cap_data['type']),
                domain=cap_data['domain'],
                confidence=cap_data.get('confidence', 1.0),
                prerequisites=set(cap_data.get('prerequisites', [])),
                metadata=cap_data.get('metadata', {})
            ))
        
        return cls(
            agent_id=data['agent_id'],
            name=data.get('name', 'BodhiAgent'),
            version=data.get('version', '1.0.0'),
            capabilities=capabilities,
            knowledge_domains=data.get('knowledge_domains', []),
            tools=set(data.get('tools', [])),
            goals=data.get('goals', []),
            learning_rate=data.get('learning_rate', 0.1),
            collaboration_preference=data.get('collaboration_preference', 0.5),
            risk_tolerance=data.get('risk_tolerance', 0.3),
            creativity_factor=data.get('creativity_factor', 0.4),
            max_memory_size=data.get('max_memory_size', 10000),
            max_execution_time=data.get('max_execution_time', 300),
            max_concurrent_tasks=data.get('max_concurrent_tasks', 5),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now(timezone.utc).isoformat())),
            creator_agent_id=data.get('creator_agent_id'),
            generation=data.get('generation', 1),
            schema_version=data.get('schema_version', '1.0')
        )


class Agent:
    """
    Core Agent class - the fundamental unit of intelligence in Bodhi
    
    An Agent is an autonomous entity that can:
    - Execute tasks within its capabilities
    - Communicate with other agents via A2A protocol
    - Learn from experience and adapt
    - Operate within a sandboxed environment
    - Serialize/deserialize for persistence
    """
    
    def __init__(self, dna: AgentDNA, sandbox: Optional[AgentSandbox] = None):
        self.dna = dna
        self.state = AgentState.INITIALIZING
        self.sandbox = sandbox or AgentSandbox(self.dna.agent_id)
        
        # Runtime state
        self.memory: List[Dict[str, Any]] = []
        self.active_tools: Dict[str, MCPTool] = {}
        self.communication_channels: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # A2A Communication
        self.a2a_protocol = A2AProtocol(self.dna.agent_id)
        
        # Learning system
        self.feedback_entries: List[FeedbackEntry] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Lifecycle tracking
        self.created_at = datetime.now(timezone.utc)
        self.last_active = self.created_at
        self.task_count = 0
        
        logger.info(f"🧘 Agent {self.dna.name} ({self.dna.agent_id}) initialized")
        self.state = AgentState.ACTIVE
    
    @property
    def id(self) -> str:
        """Agent unique identifier"""
        return self.dna.agent_id
    
    @property
    def name(self) -> str:
        """Agent name"""
        return self.dna.name
    
    @property
    def capabilities(self) -> Set[Capability]:
        """Agent capabilities"""
        return self.dna.capabilities
    
    def has_capability(self, capability_type: CapabilityType, domain: str = None) -> bool:
        """Check if agent has a specific capability"""
        for cap in self.capabilities:
            if cap.type == capability_type:
                if domain is None or cap.domain == domain:
                    return True
        return False
    
    def get_capability_confidence(self, capability_type: CapabilityType, domain: str = None) -> float:
        """Get confidence level for a specific capability"""
        for cap in self.capabilities:
            if cap.type == capability_type:
                if domain is None or cap.domain == domain:
                    return cap.confidence
        return 0.0
    
    async def execute_task(self, task: 'Task') -> 'ExecutionResult':
        """
        Execute a task within the agent's sandbox environment
        
        This is the main entry point for task execution. The agent will:
        1. Validate task requirements against capabilities
        2. Set up sandbox environment
        3. Execute task logic
        4. Record results and learn from feedback
        """
        from .task import Task
        from .result import ExecutionResult
        
        if self.state != AgentState.ACTIVE:
            raise AgentExecutionError(f"Agent {self.id} is not active (state: {self.state})")
        
        self.last_active = datetime.now(timezone.utc)
        self.task_count += 1
        
        logger.info(f"🎯 Agent {self.name} executing task: {task.intent}")
        
        try:
            # Validate task requirements
            if not self._can_handle_task(task):
                return ExecutionResult(
                    success=False,
                    error="Agent lacks required capabilities for this task",
                    agent_id=self.id,
                    task_id=task.id
                )
            
            # Execute in sandbox
            async with self.sandbox.execution_context():
                result = await self._execute_task_logic(task)
                
            # Record execution
            self._record_execution(task, result)
            
            # Learn from execution
            await self._learn_from_execution(task, result)
            
            logger.info(f"✅ Agent {self.name} completed task successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Agent {self.name} execution failed: {str(e)}")
            error_result = ExecutionResult(
                success=False,
                error=str(e),
                agent_id=self.id,
                task_id=task.id
            )
            self._record_execution(task, error_result)
            return error_result
    
    async def _execute_task_logic(self, task: 'Task') -> 'ExecutionResult':
        """
        Core task execution logic - to be overridden by specialized agents
        
        This method contains the actual task processing logic.
        Subclasses should override this to implement domain-specific behavior.
        """
        from core.result import ExecutionResult
        
        # Default implementation - basic processing
        await asyncio.sleep(0.1)  # Simulate processing
        
        return ExecutionResult(
            success=True,
            data={
                'agent_name': self.name,
                'agent_id': self.id,
                'task_intent': task.intent,
                'capabilities_used': [cap.type.value for cap in self.capabilities],
                'processing_time': 0.1
            },
            agent_id=self.id,
            task_id=task.id,
            message=f"Task processed by agent {self.name}"
        )
    
    def _can_handle_task(self, task: 'Task') -> bool:
        """Check if agent can handle the given task"""
        # Basic capability matching
        required_capabilities = task.required_capabilities
        agent_capabilities = {cap.type for cap in self.capabilities}
        
        return required_capabilities.issubset(agent_capabilities)
    
    def _record_execution(self, task: 'Task', result: 'ExecutionResult'):
        """Record task execution for learning and analysis"""
        execution_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'task_id': task.id,
            'task_intent': task.intent,
            'result_success': result.success,
            'execution_time': getattr(result, 'execution_time', 0.0) or 0.0,
            'capabilities_used': [cap.type.value for cap in self.capabilities]
        }
        
        self.execution_history.append(execution_record)
        
        # Limit history size
        if len(self.execution_history) > self.dna.max_memory_size:
            self.execution_history = self.execution_history[-self.dna.max_memory_size:]
    
    async def _learn_from_execution(self, task: 'Task', result: 'ExecutionResult'):
        """Learn from task execution to improve future performance"""
        # Basic learning implementation
        if result.success:
            # Increase confidence in used capabilities
            for cap in self.capabilities:
                if cap.type in task.required_capabilities:
                    # Slightly increase confidence (capped at 1.0)
                    new_confidence = min(1.0, cap.confidence + self.dna.learning_rate * 0.1)
                    # Note: In a full implementation, we'd need to handle immutable capabilities
        
        # Update performance metrics
        self.performance_metrics['success_rate'] = self._calculate_success_rate()
        self.performance_metrics['avg_execution_time'] = self._calculate_avg_execution_time()
    
    def _calculate_success_rate(self) -> float:
        """Calculate recent success rate"""
        if not self.execution_history:
            return 0.0
        
        recent_executions = self.execution_history[-10:]  # Last 10 executions
        successful = sum(1 for exec in recent_executions if exec['result_success'])
        return successful / len(recent_executions)
    
    def _calculate_avg_execution_time(self) -> float:
        """Calculate average execution time"""
        if not self.execution_history:
            return 0.0
        
        recent_executions = self.execution_history[-10:]
        total_time = sum(exec.get('execution_time', 0.0) for exec in recent_executions)
        return total_time / len(recent_executions)
    
    async def communicate_with_agent(self, target_agent_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to another agent via A2A protocol
        
        This enables agents to collaborate and share information.
        """
        self.state = AgentState.COMMUNICATING
        
        try:
            agent_message = AgentMessage(
                sender_id=self.id,
                recipient_id=target_agent_id,
                content=message,
                message_type="collaboration"
            )
            
            response = await self.a2a_protocol.send_message(agent_message)
            
            logger.info(f"📡 Agent {self.name} communicated with {target_agent_id}")
            return response
            
        finally:
            self.state = AgentState.ACTIVE
    
    def add_tool(self, tool: MCPTool):
        """Add an MCP-compliant tool to the agent's toolkit"""
        self.active_tools[tool.name] = tool
        logger.info(f"🔧 Agent {self.name} acquired tool: {tool.name}")
    
    def remove_tool(self, tool_name: str):
        """Remove a tool from the agent's toolkit"""
        if tool_name in self.active_tools:
            del self.active_tools[tool_name]
            logger.info(f"🗑️ Agent {self.name} removed tool: {tool_name}")
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.active_tools.keys())
    
    async def use_tool(self, tool_name: str, input_data: Any) -> Any:
        """Use a specific tool with given input"""
        if tool_name not in self.active_tools:
            raise ValueError(f"Tool '{tool_name}' not available to agent {self.name}")
        
        tool = self.active_tools[tool_name]
        return await tool.execute(input_data)
    
    def serialize(self) -> str:
        """Serialize agent to JSON string"""
        agent_data = {
            'dna': self.dna.to_dict(),
            'state': self.state.value,
            'memory': self.memory[-100:],  # Last 100 memory entries
            'execution_history': self.execution_history[-50:],  # Last 50 executions
            'performance_metrics': self.performance_metrics,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat(),
            'task_count': self.task_count,
            'serialization_timestamp': datetime.now(timezone.utc).isoformat(),
            'framework_version': '0.1.0'
        }
        
        return json.dumps(agent_data, indent=2)
    
    @classmethod
    def deserialize(cls, serialized_data: str, sandbox: Optional[AgentSandbox] = None) -> 'Agent':
        """Deserialize agent from JSON string"""
        try:
            data = json.loads(serialized_data)
            
            # Reconstruct DNA
            dna = AgentDNA.from_dict(data['dna'])
            
            # Create agent
            agent = cls(dna, sandbox)
            
            # Restore state
            agent.state = AgentState(data['state'])
            agent.memory = data.get('memory', [])
            agent.execution_history = data.get('execution_history', [])
            agent.performance_metrics = data.get('performance_metrics', {})
            agent.created_at = datetime.fromisoformat(data['created_at'])
            agent.last_active = datetime.fromisoformat(data['last_active'])
            agent.task_count = data.get('task_count', 0)
            
            logger.info(f"🔄 Agent {agent.name} deserialized successfully")
            return agent
            
        except Exception as e:
            raise BodhiError(f"Failed to deserialize agent: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            'id': self.id,
            'name': self.name,
            'state': self.state.value,
            'capabilities': [
                {
                    'type': cap.type.value,
                    'domain': cap.domain,
                    'confidence': cap.confidence
                }
                for cap in self.capabilities
            ],
            'tools_count': len(self.active_tools),
            'memory_size': len(self.memory),
            'execution_count': len(self.execution_history),
            'performance_metrics': self.performance_metrics,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat(),
            'task_count': self.task_count,
            'generation': self.dna.generation
        }
    
    def __str__(self) -> str:
        return f"Agent({self.name}, {len(self.capabilities)} capabilities, {self.state.value})"
    
    def __repr__(self) -> str:
        return f"Agent(id='{self.id}', name='{self.name}', capabilities={len(self.capabilities)})" 