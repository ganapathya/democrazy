# Bodhi Meta-Agent Framework

A Python framework for building dynamic AI agent systems with on-demand specialist creation, cross-domain learning capabilities, and industry-standard agent communication.

## Overview

Bodhi is a meta-agent framework that creates specialized agents dynamically based on task requirements. The system includes Google's A2A (Agent-to-Agent) protocol for enterprise-grade agent communication, sophisticated NLP2SQL implementation with cross-database learning, and adaptive pattern recognition.

## Features

- **Dynamic Agent Creation**: Creates specialized agents on-demand based on task requirements
- **Google A2A Protocol**: Full implementation of Google's Agent-to-Agent communication standard
- **Cross-Domain Learning**: Learns patterns across different database types and query structures
- **Agent Evolution**: Uses genetic algorithms to improve agent performance over time
- **NLP2SQL Capabilities**: Converts natural language to SQL queries for PostgreSQL and MongoDB
- **Enterprise Communication**: JSON-RPC 2.0 compliant A2A messaging with multi-modal content
- **Real-time Streaming**: Server-Sent Events (SSE) for live task updates
- **Agent Discovery**: Industry-standard agent card discovery (/.well-known/agent.json)
- **MCP Tool Compliance**: Supports Model Context Protocol for tool integration
- **Secure Execution**: Sandboxed agent execution environment

## Installation

### Requirements

- Python 3.8+
- Optional: `psycopg2` for PostgreSQL support
- Optional: `pymongo` for MongoDB support

### Setup

```bash
git clone <repository-url>
cd democrazy
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from bodhi import Agent, Task, AgentFactory, NLP2SQLMetaAgent

# Create a basic agent
factory = AgentFactory()
agent = factory.create_agent_from_template("nlp_specialist")

# Use the meta-agent for NLP2SQL
meta_agent = NLP2SQLMetaAgent()
result = await meta_agent.process_natural_language_query(
    "Show me all customers from New York",
    {"database_type": "postgresql"}
)
```

### A2A Agent Communication

```python
from bodhi import Agent, AgentDNA
from bodhi.communication.a2a import create_text_message, MessageRole

# Create agents with A2A support
agent1 = Agent(dna1)  # Automatically includes A2A protocol
agent2 = Agent(dna2)

# Communicate between agents
message = {"task": "analyze_data", "dataset": "customer_orders"}
result = await agent1.communicate_with_agent(agent2.id, message)

# Get agent's A2A card for discovery
agent_card_json = agent1.get_agent_card_json()
```

### Running Demos

```bash
# Core framework demonstration
python bodhi/examples/demo_bodhi_primitives.py

# NLP2SQL with emergent intelligence
python bodhi/examples/demo_emergent_nlp2sql.py

# Google A2A protocol demonstration
python bodhi/examples/demo_a2a_protocol.py
```

## Architecture

```
bodhi/
├── core/                    # Core primitives
│   ├── agent.py            # Agent and AgentDNA classes with A2A
│   ├── task.py             # Task management
│   ├── factory.py          # Agent creation and evolution
│   ├── capability.py       # Agent capabilities
│   └── result.py           # Execution results
├── specialists/             # Specialized agents
│   ├── postgres_specialist.py
│   └── mongodb_specialist.py
├── meta_agent/              # Meta-agent system
│   └── nlp2sql_meta_agent.py
├── communication/           # A2A protocol implementation
│   └── a2a.py              # Google A2A standard compliance
├── tools/                   # MCP-compliant tools
├── security/               # Agent sandboxing
├── learning/               # Learning systems
└── examples/               # Demos and examples
    ├── demo_bodhi_primitives.py
    ├── demo_emergent_nlp2sql.py
    └── demo_a2a_protocol.py
```

## Core Components

### Agent

The base agent class with DNA-based configuration and built-in A2A protocol:

```python
from bodhi.core.agent import Agent, AgentDNA
from bodhi.core.capability import Capability, CapabilityType

# Define agent capabilities
dna = AgentDNA(
    name="MyAgent",
    capabilities={
        Capability(type=CapabilityType.NLP_PROCESSING, domain="text", confidence=0.8)
    },
    knowledge_domains=["natural_language", "databases"],
    tools={"nlp_processor", "sql_generator"}
)

agent = Agent(dna)  # Automatically includes A2A protocol

# Agent discovery and communication
agent_card = agent.get_agent_card_json()  # For /.well-known/agent.json
discovered_agent = await agent.discover_remote_agent("https://api.example.com/agent")
```

### A2A Protocol

Google's Agent-to-Agent communication standard implementation:

```python
from bodhi.communication.a2a import (
    A2AProtocol, AgentCard, create_text_message,
    create_data_message, create_file_message
)

# Create A2A protocol instance
protocol = A2AProtocol(agent_id, agent_card)

# Create and send multi-modal messages
text_msg = create_text_message("Hello, please process this data")
data_msg = create_data_message({"metrics": [1, 2, 3], "type": "analytics"})
file_msg = create_file_message(FileContent(name="data.csv", uri="https://..."))

# Task-based communication
task = protocol.create_task()
result = await protocol.send_message(task.id, text_msg)

# Streaming communication
stream = await protocol.stream_message(task.id, data_msg)
while event := await stream.get_next_event():
    print(f"Event: {event['type']}")
```

### Task

Structured task representation with requirements:

```python
from bodhi.core.task import Task, TaskType
from bodhi.core.requirement import Requirement

task = Task(
    intent="Convert natural language to SQL",
    task_type=TaskType.TRANSFORMATION,
    context={"natural_query": "Show me all users"}
)
```

### AgentFactory

Creates and evolves agents based on requirements:

```python
from bodhi.core.factory import AgentFactory

factory = AgentFactory()

# Create from template
agent = factory.create_agent_from_template("database_expert")

# Create for specific task
agent = factory.create_agent_for_task(task)
```

### NLP2SQL Meta-Agent

The main meta-agent for natural language to SQL conversion:

```python
from bodhi.meta_agent.nlp2sql_meta_agent import NLP2SQLMetaAgent

meta_agent = NLP2SQLMetaAgent()

# Process natural language query
result = await meta_agent.process_natural_language_query(
    "Get customers who placed orders in the last month",
    {"database_type": "postgresql"}
)

# Get system intelligence report
report = meta_agent.get_system_intelligence_report()
```

## Database Specialists

### PostgreSQL Specialist

Handles PostgreSQL-specific query generation:

```python
from bodhi.specialists.postgres_specialist import PostgreSQLSpecialist

specialist = PostgreSQLSpecialist(dna, connection_config={
    "host": "localhost",
    "database": "mydb",
    "user": "postgres"
})
```

### MongoDB Specialist

Handles MongoDB aggregation pipelines:

```python
from bodhi.specialists.mongodb_specialist import MongoDBSpecialist

specialist = MongoDBSpecialist(dna, connection_config={
    "host": "localhost",
    "database": "mydb"
})
```

## A2A Protocol Features

### Agent Card Discovery

Agents automatically expose their capabilities via Agent Cards:

```json
{
  "name": "DataAnalystAgent",
  "description": "Bodhi agent with data analysis capabilities",
  "url": "https://agents.bodhi.ai/agent_001",
  "provider": {
    "organization": "Bodhi Framework",
    "contact": "agent@bodhi.ai"
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateTransitionHistory": true
  },
  "skills": [
    {
      "id": "data_analysis_finance",
      "name": "Data Analysis",
      "description": "data_analysis capability in finance",
      "inputModes": ["text", "data"],
      "outputModes": ["text", "data"]
    }
  ]
}
```

### Multi-Modal Communication

Support for text, structured data, and files:

```python
# Text communication
text_msg = create_text_message("Analyze customer data", metadata={"priority": "high"})

# Structured data
data_msg = create_data_message({
    "query_type": "analytics",
    "filters": {"region": "US", "date_range": "30d"}
})

# File references
file_msg = create_file_message(FileContent(
    name="dataset.csv",
    mime_type="text/csv",
    uri="https://data.company.com/dataset.csv"
))
```

### Task Lifecycle Management

Complete task state management:

- `submitted` → `working` → `completed`
- `submitted` → `working` → `input-required` → `working` → `completed`
- `submitted` → `working` → `failed`
- `submitted` → `canceled`

## Configuration

### Environment Variables

- `BODHI_SHOW_BANNER`: Set to '0' to disable framework banner (default: '1')

### Database Configuration

Configure database connections through the specialist constructors or meta-agent configuration.

## API Reference

### Core Classes

- `Agent`: Base agent class with DNA-based configuration and A2A protocol
- `AgentDNA`: Agent blueprint defining capabilities and behavior
- `Task`: Structured task representation
- `AgentFactory`: Creates and evolves agents
- `Capability`: Represents agent capabilities
- `Requirement`: Defines task requirements

### A2A Protocol Classes

- `A2AProtocol`: Main protocol implementation
- `AgentCard`: Agent metadata and capability advertisement
- `Task`: A2A task with lifecycle management
- `Message`: Multi-modal message structure
- `Artifact`: Task execution results
- `A2AEventStream`: Server-Sent Events streaming

### Specialists

- `PostgreSQLSpecialist`: PostgreSQL query generation specialist
- `MongoDBSpecialist`: MongoDB aggregation pipeline specialist

### Meta-Agent

- `NLP2SQLMetaAgent`: Main meta-agent for natural language processing

## Examples

The `bodhi/examples/` directory contains:

- `demo_bodhi_primitives.py`: Core framework demonstration
- `demo_emergent_nlp2sql.py`: NLP2SQL with learning and evolution
- `demo_a2a_protocol.py`: Complete A2A protocol demonstration

### A2A Protocol Demo

The A2A demo showcases all protocol features:

1. **Agent Card Creation**: Agent metadata and skill advertisement
2. **Basic Communication**: Simple agent-to-agent messaging
3. **Multi-modal Content**: Text, data, and file message types
4. **Task Management**: Complete lifecycle with state transitions
5. **Streaming Communication**: Real-time updates via Server-Sent Events
6. **Agent Discovery**: Remote agent capability discovery
7. **Error Handling**: Comprehensive error scenarios and recovery

```bash
python bodhi/examples/demo_a2a_protocol.py
```

## Testing

Run the framework validation:

```python
# All imports and basic functionality
python -c "from bodhi import Agent, Task, AgentFactory, NLP2SQLMetaAgent; print('Framework working')"

# A2A protocol validation
python -c "from bodhi.communication.a2a import A2AProtocol, AgentCard; print('A2A working')"
```

### Regression Testing

All demos have been regression tested to ensure compatibility:

```bash
# Test core primitives
python bodhi/examples/demo_bodhi_primitives.py

# Test emergent intelligence
python bodhi/examples/demo_emergent_nlp2sql.py

# Test A2A protocol (100% success rate)
python bodhi/examples/demo_a2a_protocol.py
```

## Development

### Adding New Specialists

1. Create a new specialist class inheriting from `Agent`
2. Implement `_execute_task_logic` method
3. Add specialist to the meta-agent's specialist creation logic
4. Ensure A2A protocol compatibility

### Extending A2A Capabilities

1. Add new skill types to agent cards
2. Implement custom message handlers
3. Register skill handlers with `register_a2a_skill_handler`
4. Follow JSON-RPC 2.0 and A2A specification standards

### Extending Capabilities

1. Add new capability types to `CapabilityType` enum
2. Update agent DNA templates in `AgentFactory`
3. Implement corresponding task execution logic
4. Ensure A2A skill mapping

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
