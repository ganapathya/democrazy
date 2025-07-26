"""
🧘 Bodhi Framework Exceptions
============================

Custom exception classes for the Bodhi meta-agent framework.
These provide structured error handling and debugging capabilities.
"""

from typing import Optional, Dict, Any


class BodhiError(Exception):
    """Base exception for all Bodhi framework errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context
        }


class AgentError(BodhiError):
    """Base exception for agent-related errors"""
    pass


class AgentCreationError(AgentError):
    """Raised when agent creation fails"""
    pass


class AgentExecutionError(AgentError):
    """Raised when agent execution fails"""
    pass


class TaskError(BodhiError):
    """Base exception for task-related errors"""
    pass


class TaskValidationError(TaskError):
    """Raised when task validation fails"""
    pass


class TaskExecutionError(TaskError):
    """Raised when task execution fails"""
    pass


class CommunicationError(BodhiError):
    """Raised when agent communication fails"""
    pass


class SandboxError(BodhiError):
    """Raised when sandbox operations fail"""
    pass


class SerializationError(BodhiError):
    """Raised when serialization/deserialization fails"""
    pass


class ToolError(BodhiError):
    """Raised when tool operations fail"""
    pass


class RegistryError(BodhiError):
    """Raised when registry operations fail"""
    pass


class FactoryError(BodhiError):
    """Raised when factory operations fail"""
    pass 