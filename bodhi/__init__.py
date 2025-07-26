"""
🧘 Bodhi: Meta-Agent Framework
=============================

A sophisticated framework for emergent intelligence and self-assembling AI systems.

Key Features:
- Dynamic agent creation and evolution
- Meta-learning and pattern recognition
- Cross-domain knowledge synthesis
- Agent-to-Agent (A2A) communication
- MCP-compliant tools
- Secure sandboxed execution
- Advanced NLP2SQL capabilities

Example Usage:
    >>> from bodhi import Agent, Task, AgentFactory
    >>> from bodhi.meta_agent import NLP2SQLMetaAgent
    >>> 
    >>> # Create a meta-agent that learns and creates specialists
    >>> meta_agent = NLP2SQLMetaAgent()
    >>> result = await meta_agent.process_natural_language_query(
    ...     "Show customers from New York", {"database_type": "postgresql"}
    ... )
"""

__version__ = "0.1.0"
__author__ = "Democrazy Team"
__description__ = "Meta-Agent Framework for Emergent Intelligence"

# Core primitives - The foundation of the framework
from .core.agent import Agent, AgentDNA
from .core.task import Task, TaskContext, TaskStatus, TaskPriority, TaskType
from .core.factory import AgentFactory, CreationStrategy, DNATemplate
from .core.capability import Capability, CapabilityType
from .core.requirement import Requirement
from .core.result import Result, ExecutionResult, ResultStatus

# Specialists - Specialized agents for different domains
from .specialists.postgres_specialist import PostgreSQLSpecialist
from .specialists.mongodb_specialist import MongoDBSpecialist

# Meta-Agent System - The heart of emergent intelligence
from .meta_agent.nlp2sql_meta_agent import NLP2SQLMetaAgent, DatabaseType, QueryComplexity

# Communication layer - Agent-to-Agent protocols
from .communication.a2a import A2AProtocol, AgentMessage

# Tools and MCP - Model Context Protocol compliance
from .tools.mcp import MCPTool, MCPInterface

# Security and sandboxing - Secure execution environment
from .security.sandbox import AgentSandbox

# Learning systems - Meta-learning and adaptation
from .learning.feedback import FeedbackCollector, FeedbackEntry

# Utilities - Framework utilities and exceptions
from .utils.exceptions import (
    BodhiError, AgentError, AgentCreationError, AgentExecutionError,
    TaskError, TaskValidationError, CommunicationError, SandboxError
)

# Convenience imports for common use cases
from .examples.demo_bodhi_primitives import demo_agent_creation_and_dna, demo_task_creation_and_requirements
from .examples.demo_emergent_nlp2sql import demonstrate_emergent_superintelligence

__all__ = [
    # Core primitives
    'Agent', 'AgentDNA', 'Task', 'TaskContext', 'TaskStatus', 'TaskPriority', 'TaskType',
    'AgentFactory', 'CreationStrategy', 'DNATemplate', 'Capability', 'CapabilityType',
    'Requirement', 'Result', 'ExecutionResult', 'ResultStatus',
    
    # Specialists
    'PostgreSQLSpecialist', 'MongoDBSpecialist',
    
    # Meta-Agent System
    'NLP2SQLMetaAgent', 'DatabaseType', 'QueryComplexity',
    
    # Communication
    'A2AProtocol', 'AgentMessage',
    
    # Tools
    'MCPTool', 'MCPInterface',
    
    # Security
    'AgentSandbox',
    
    # Learning
    'FeedbackCollector', 'FeedbackEntry',
    
    # Utilities
    'BodhiError', 'AgentError', 'AgentCreationError', 'AgentExecutionError',
    'TaskError', 'TaskValidationError', 'CommunicationError', 'SandboxError',
    
    # Examples
    'demo_agent_creation_and_dna', 'demo_task_creation_and_requirements',
    'demonstrate_emergent_superintelligence'
]

FRAMEWORK_INFO = {
    'name': 'Bodhi Meta-Agent Framework',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'capabilities': [
        'Dynamic Agent Creation',
        'Meta-Learning Systems', 
        'Cross-Domain Synthesis',
        'Emergent Intelligence',
        'Self-Assembly',
        'A2A Communication',
        'MCP Tool Compliance',
        'Secure Sandboxing',
        'NLP2SQL Specialization'
    ],
    'core_primitives': ['Agent', 'Task', 'Factory', 'Capability', 'Requirement'],
    'specialist_types': ['PostgreSQL', 'MongoDB'],
    'emergent_behaviors': [
        'Cross-Database Learning',
        'Multi-Specialist Ecosystem',
        'Adaptive Performance',
        'Pattern Generalization',
        'Knowledge Synthesis'
    ]
}

def get_framework_info():
    """Get comprehensive framework information"""
    return FRAMEWORK_INFO.copy()

def print_framework_banner():
    """Print the Bodhi framework banner"""
    print("""
    🧘 ====================================== 🧘
         BODHI META-AGENT FRAMEWORK
       Emergent Intelligence & Self-Assembly
    🧘 ====================================== 🧘
    
    Version: {version}
    Author: {author}
    
    Key Capabilities:
    • Dynamic Specialist Creation
    • Meta-Learning & Adaptation  
    • Cross-Domain Synthesis
    • Emergent Superintelligence
    • A2A Communication
    • MCP Tool Compliance
    
    🚀 Ready for Emergent Intelligence!
    """.format(**FRAMEWORK_INFO))

# Auto-print banner on import (can be disabled)
import os
if os.getenv('BODHI_SHOW_BANNER', '1') == '1':
    print_framework_banner()
