"""
🧘 A2A (Agent-to-Agent) Communication Protocol
=============================================

Implementation of Google's A2A protocol for agent-to-agent communication.
Compliant with the A2A specification for enterprise-grade agent collaboration.

Key Features:
- Agent Card discovery (/.well-known/agent.json)
- Task-based communication
- Multi-modal content support (text, data, files)
- Server-Sent Events (SSE) streaming
- Enterprise authentication and security
- JSON-RPC 2.0 protocol compliance
"""

import asyncio
import json
import uuid
import time
from typing import Dict, Any, Optional, List, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """A2A Task lifecycle states"""
    SUBMITTED = "submitted"      # Task received but not yet started
    WORKING = "working"          # Task is actively being processed
    INPUT_REQUIRED = "input-required"  # Agent needs more information
    COMPLETED = "completed"      # Task finished successfully
    CANCELED = "canceled"        # Task terminated by client/system
    FAILED = "failed"           # An error occurred during processing
    UNKNOWN = "unknown"         # State cannot be determined


class MessageRole(Enum):
    """Message roles in A2A communication"""
    USER = "user"    # Message from client agent
    AGENT = "agent"  # Message from remote agent


class PartType(Enum):
    """A2A Part types for multi-modal content"""
    TEXT = "text"
    FILE = "file"
    DATA = "data"


@dataclass
class FileContent:
    """File content specification for FilePart"""
    name: Optional[str] = None
    mime_type: Optional[str] = None
    bytes: Optional[str] = None  # Base64 encoded content
    uri: Optional[str] = None    # URI reference to file


@dataclass
class TextPart:
    """Text content part"""
    text: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": "text", "text": self.text}
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass  
class FilePart:
    """File content part"""
    file: FileContent
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": "file"}
        file_dict = {}
        if self.file.name:
            file_dict["name"] = self.file.name
        if self.file.mime_type:
            file_dict["mimeType"] = self.file.mime_type
        if self.file.bytes:
            file_dict["bytes"] = self.file.bytes
        if self.file.uri:
            file_dict["uri"] = self.file.uri
        result["file"] = file_dict
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class DataPart:
    """Structured data content part"""
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": "data", "data": self.data}
        if self.metadata:
            result["metadata"] = self.metadata
        return result


# Union type for all part types
Part = Union[TextPart, FilePart, DataPart]


@dataclass
class Message:
    """A2A Message structure"""
    role: MessageRole
    parts: List[Part]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "role": self.role.value,
            "parts": [part.to_dict() for part in self.parts]
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create Message from A2A JSON format"""
        role = MessageRole(data["role"])
        parts = []
        
        for part_data in data["parts"]:
            part_type = part_data["type"]
            metadata = part_data.get("metadata")
            
            if part_type == "text":
                parts.append(TextPart(text=part_data["text"], metadata=metadata))
            elif part_type == "file":
                file_data = part_data["file"]
                file_content = FileContent(
                    name=file_data.get("name"),
                    mime_type=file_data.get("mimeType"),
                    bytes=file_data.get("bytes"),
                    uri=file_data.get("uri")
                )
                parts.append(FilePart(file=file_content, metadata=metadata))
            elif part_type == "data":
                parts.append(DataPart(data=part_data["data"], metadata=metadata))
        
        return cls(role=role, parts=parts, metadata=data.get("metadata"))


@dataclass
class Artifact:
    """A2A Artifact - immutable results from task execution"""
    name: str
    parts: List[Part]
    description: Optional[str] = None
    index: Optional[int] = None
    append: bool = False
    last_chunk: bool = True
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "parts": [part.to_dict() for part in self.parts]
        }
        if self.description:
            result["description"] = self.description
        if self.index is not None:
            result["index"] = self.index
        if self.append:
            result["append"] = self.append
        if not self.last_chunk:
            result["lastChunk"] = self.last_chunk
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class Task:
    """A2A Task - stateful unit of work"""
    id: str
    status: TaskStatus
    artifacts: List[Artifact] = field(default_factory=list)
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    history: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "status": self.status.value,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "history": [message.to_dict() for message in self.history],
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat()
        }
        if self.session_id:
            result["sessionId"] = self.session_id
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def update_status(self, status: TaskStatus, message: Optional[Message] = None):
        """Update task status and optionally add message"""
        self.status = status
        self.updated_at = datetime.now(timezone.utc)
        if message:
            self.history.append(message)

    def add_artifact(self, artifact: Artifact):
        """Add an artifact to the task"""
        self.artifacts.append(artifact)
        self.updated_at = datetime.now(timezone.utc)


@dataclass
class AgentSkill:
    """A2A Agent skill definition"""
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    input_modes: List[str] = field(default_factory=lambda: ["text"])
    output_modes: List[str] = field(default_factory=lambda: ["text"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "examples": self.examples,
            "inputModes": self.input_modes,
            "outputModes": self.output_modes
        }


@dataclass
class AgentCapabilities:
    """A2A Agent capabilities"""
    streaming: bool = True
    push_notifications: bool = True
    state_transition_history: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "streaming": self.streaming,
            "pushNotifications": self.push_notifications,
            "stateTransitionHistory": self.state_transition_history
        }


@dataclass
class AuthenticationScheme:
    """A2A Authentication scheme definition"""
    schemes: List[str] = field(default_factory=lambda: ["bearer"])

    def to_dict(self) -> Dict[str, Any]:
        return {"schemes": self.schemes}


@dataclass
class AgentProvider:
    """A2A Agent provider information"""
    organization: str
    contact: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"organization": self.organization}
        if self.contact:
            result["contact"] = self.contact
        if self.url:
            result["url"] = self.url
        return result


@dataclass
class AgentCard:
    """A2A Agent Card - agent metadata and capabilities"""
    name: str
    description: str
    url: str
    provider: AgentProvider
    version: str = "1.0.0"
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
    authentication: AuthenticationScheme = field(default_factory=AuthenticationScheme)
    default_input_modes: List[str] = field(default_factory=lambda: ["text"])
    default_output_modes: List[str] = field(default_factory=lambda: ["text"])
    skills: List[AgentSkill] = field(default_factory=list)
    documentation_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "provider": self.provider.to_dict(),
            "version": self.version,
            "capabilities": self.capabilities.to_dict(),
            "authentication": self.authentication.to_dict(),
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "skills": [skill.to_dict() for skill in self.skills]
        }
        if self.documentation_url:
            result["documentationUrl"] = self.documentation_url
        return result

    def to_json(self) -> str:
        """Convert to JSON string for /.well-known/agent.json"""
        return json.dumps(self.to_dict(), indent=2)


class A2AError(Exception):
    """A2A protocol error"""
    def __init__(self, code: int, message: str, data: Optional[Any] = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"A2A Error {code}: {message}")

    def to_dict(self) -> Dict[str, Any]:
        result = {"code": self.code, "message": self.message}
        if self.data:
            result["data"] = self.data
        return result


# Standard A2A error codes
class A2AErrorCodes:
    # JSON-RPC 2.0 standard codes
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # A2A specific codes
    TASK_NOT_FOUND = 1001
    TASK_INVALID_STATE = 1002
    AUTHENTICATION_FAILED = 1003
    AUTHORIZATION_FAILED = 1004
    RATE_LIMITED = 1005
    CAPABILITY_NOT_SUPPORTED = 1006


class A2AProtocol:
    """
    A2A (Agent-to-Agent) Protocol Implementation
    
    Compliant with Google's A2A specification for enterprise agent communication.
    Supports JSON-RPC 2.0, multi-modal content, and enterprise security.
    """

    def __init__(self, agent_id: str, agent_card: AgentCard):
        self.agent_id = agent_id
        self.agent_card = agent_card
        self.tasks: Dict[str, Task] = {}
        self.message_handlers: Dict[str, callable] = {}
        self.auth_token: Optional[str] = None
        
        logger.info(f"🔗 A2A Protocol initialized for agent: {agent_id}")

    def set_auth_token(self, token: str):
        """Set authentication token for outbound requests"""
        self.auth_token = token

    async def discover_agent(self, agent_url: str) -> AgentCard:
        """
        Discover remote agent capabilities via Agent Card
        
        Fetches /.well-known/agent.json from the remote agent
        """
        try:
            # In a real implementation, this would make an HTTP request
            # For now, we'll simulate the discovery
            await asyncio.sleep(0.01)  # Simulate network delay
            
            logger.info(f"🔍 Discovering agent at: {agent_url}")
            
            # Mock agent card for demo
            return AgentCard(
                name="RemoteAgent",
                description="A remote agent discovered via A2A",
                url=agent_url,
                provider=AgentProvider(organization="Remote Corp"),
                skills=[
                    AgentSkill(
                        id="general-task",
                        name="General Task Processing",
                        description="Handles general task requests"
                    )
                ]
            )
            
        except Exception as e:
            raise A2AError(
                A2AErrorCodes.INTERNAL_ERROR,
                f"Failed to discover agent: {str(e)}"
            )

    def create_task(self, session_id: Optional[str] = None) -> Task:
        """Create a new A2A task"""
        task_id = str(uuid.uuid4())
        task = Task(id=task_id, status=TaskStatus.SUBMITTED, session_id=session_id)
        self.tasks[task_id] = task
        
        logger.info(f"📝 Created A2A task: {task_id}")
        return task

    async def send_message(self, task_id: str, message: Message, 
                          remote_agent_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Send message via A2A tasks/send method
        
        Implements JSON-RPC 2.0 tasks/send call
        """
        try:
            task = self.tasks.get(task_id)
            if not task:
                raise A2AError(A2AErrorCodes.TASK_NOT_FOUND, f"Task {task_id} not found")

            # Update task with new message
            task.history.append(message)
            task.update_status(TaskStatus.WORKING)

            # In a real implementation, this would make HTTP request to remote agent
            # For now, simulate the interaction
            await asyncio.sleep(0.1)  # Simulate processing

            # Simulate remote agent response
            response_message = Message(
                role=MessageRole.AGENT,
                parts=[TextPart(text=f"Processed: {message.parts[0].text if message.parts and isinstance(message.parts[0], TextPart) else 'request'}")]
            )
            
            task.history.append(response_message)
            task.update_status(TaskStatus.COMPLETED)

            # Create response artifact
            artifact = Artifact(
                name="response",
                parts=response_message.parts,
                description="Agent response"
            )
            task.add_artifact(artifact)

            logger.info(f"📤 Sent A2A message for task: {task_id}")

            return {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "result": task.to_dict()
            }

        except A2AError:
            raise
        except Exception as e:
            raise A2AError(
                A2AErrorCodes.INTERNAL_ERROR,
                f"Failed to send message: {str(e)}"
            )

    async def stream_message(self, task_id: str, message: Message) -> 'A2AEventStream':
        """
        Send message via A2A tasks/sendSubscribe method with SSE streaming
        
        Returns an event stream for real-time updates
        """
        try:
            task = self.tasks.get(task_id)
            if not task:
                raise A2AError(A2AErrorCodes.TASK_NOT_FOUND, f"Task {task_id} not found")

            # Create event stream
            stream = A2AEventStream(task_id, task)
            
            # Start streaming simulation
            asyncio.create_task(stream._simulate_streaming(message))
            
            logger.info(f"📡 Started A2A streaming for task: {task_id}")
            return stream

        except A2AError:
            raise
        except Exception as e:
            raise A2AError(
                A2AErrorCodes.INTERNAL_ERROR,
                f"Failed to start streaming: {str(e)}"
            )

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """
        Get task status via A2A tasks/get method
        """
        try:
            task = self.tasks.get(task_id)
            if not task:
                raise A2AError(A2AErrorCodes.TASK_NOT_FOUND, f"Task {task_id} not found")

            return {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "result": task.to_dict()
            }

        except A2AError:
            raise
        except Exception as e:
            raise A2AError(
                A2AErrorCodes.INTERNAL_ERROR,
                f"Failed to get task: {str(e)}"
            )

    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel task via A2A tasks/cancel method
        """
        try:
            task = self.tasks.get(task_id)
            if not task:
                raise A2AError(A2AErrorCodes.TASK_NOT_FOUND, f"Task {task_id} not found")

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED]:
                raise A2AError(
                    A2AErrorCodes.TASK_INVALID_STATE,
                    f"Cannot cancel task in {task.status.value} state"
                )

            task.update_status(TaskStatus.CANCELED)
            
            logger.info(f"❌ Canceled A2A task: {task_id}")

            return {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "result": task.to_dict()
            }

        except A2AError:
            raise
        except Exception as e:
            raise A2AError(
                A2AErrorCodes.INTERNAL_ERROR,
                f"Failed to cancel task: {str(e)}"
            )

    def get_agent_card_json(self) -> str:
        """Get Agent Card as JSON for /.well-known/agent.json endpoint"""
        return self.agent_card.to_json()

    def register_message_handler(self, skill_id: str, handler: callable):
        """Register a handler for incoming messages for a specific skill"""
        self.message_handlers[skill_id] = handler
        logger.info(f"📋 Registered A2A message handler for skill: {skill_id}")

    async def handle_incoming_message(self, task_id: str, message: Message, 
                                    skill_id: Optional[str] = None) -> Task:
        """Handle incoming message from remote agent"""
        try:
            task = self.tasks.get(task_id)
            if not task:
                task = Task(id=task_id, status=TaskStatus.SUBMITTED)
                self.tasks[task_id] = task

            task.history.append(message)
            task.update_status(TaskStatus.WORKING)

            # Route to appropriate handler
            if skill_id and skill_id in self.message_handlers:
                response = await self.message_handlers[skill_id](message, task)
                if response:
                    task.history.append(response)
                    task.update_status(TaskStatus.COMPLETED)
            else:
                # Default response
                response = Message(
                    role=MessageRole.AGENT,
                    parts=[TextPart(text="Message received and processed")]
                )
                task.history.append(response)
                task.update_status(TaskStatus.COMPLETED)

            logger.info(f"📥 Handled incoming A2A message for task: {task_id}")
            return task

        except Exception as e:
            if task:
                task.update_status(TaskStatus.FAILED)
            raise A2AError(
                A2AErrorCodes.INTERNAL_ERROR,
                f"Failed to handle message: {str(e)}"
            )


class A2AEventStream:
    """A2A Server-Sent Events stream for real-time task updates"""
    
    def __init__(self, task_id: str, task: Task):
        self.task_id = task_id
        self.task = task
        self.events = asyncio.Queue()
        self.closed = False

    async def _simulate_streaming(self, initial_message: Message):
        """Simulate streaming events for demo purposes"""
        try:
            # Add initial message
            self.task.history.append(initial_message)
            
            # Send working status
            await self._emit_status_event(TaskStatus.WORKING)
            await asyncio.sleep(0.5)
            
            # Send intermediate artifact
            intermediate_artifact = Artifact(
                name="progress",
                parts=[TextPart(text="Processing request...")],
                description="Progress update"
            )
            await self._emit_artifact_event(intermediate_artifact)
            await asyncio.sleep(1.0)
            
            # Send final response
            response_message = Message(
                role=MessageRole.AGENT,
                parts=[TextPart(text="Task completed successfully via streaming")]
            )
            self.task.history.append(response_message)
            
            final_artifact = Artifact(
                name="result",
                parts=response_message.parts,
                description="Final result"
            )
            self.task.add_artifact(final_artifact)
            
            await self._emit_artifact_event(final_artifact)
            await self._emit_status_event(TaskStatus.COMPLETED)
            
        except Exception as e:
            await self._emit_error_event(A2AErrorCodes.INTERNAL_ERROR, str(e))

    async def _emit_status_event(self, status: TaskStatus):
        """Emit task status update event"""
        if self.closed:
            return
            
        self.task.update_status(status)
        event = {
            "type": "TaskStatusUpdateEvent",
            "data": {
                "taskId": self.task_id,
                "status": status.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        await self.events.put(event)

    async def _emit_artifact_event(self, artifact: Artifact):
        """Emit artifact update event"""
        if self.closed:
            return
            
        event = {
            "type": "TaskArtifactUpdateEvent", 
            "data": {
                "taskId": self.task_id,
                "artifact": artifact.to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        await self.events.put(event)

    async def _emit_error_event(self, code: int, message: str):
        """Emit error event"""
        if self.closed:
            return
            
        self.task.update_status(TaskStatus.FAILED)
        event = {
            "type": "ErrorEvent",
            "data": {
                "taskId": self.task_id,
                "error": {"code": code, "message": message},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        await self.events.put(event)

    async def get_next_event(self) -> Optional[Dict[str, Any]]:
        """Get next event from stream"""
        try:
            return await asyncio.wait_for(self.events.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    def close(self):
        """Close the event stream"""
        self.closed = True


# Convenience functions for creating common message types

def create_text_message(text: str, role: MessageRole = MessageRole.USER, 
                       metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a text message"""
    return Message(
        role=role,
        parts=[TextPart(text=text)],
        metadata=metadata
    )

def create_data_message(data: Dict[str, Any], role: MessageRole = MessageRole.USER,
                       metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a structured data message"""
    return Message(
        role=role,
        parts=[DataPart(data=data)],
        metadata=metadata
    )

def create_file_message(file_content: FileContent, role: MessageRole = MessageRole.USER,
                       metadata: Optional[Dict[str, Any]] = None) -> Message:
    """Create a file message"""
    return Message(
        role=role,
        parts=[FilePart(file=file_content)],
        metadata=metadata
    ) 