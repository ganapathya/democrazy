# Bodhi Meta-Agent Framework

A Python framework for building dynamic AI agent systems with on-demand specialist creation and cross-domain learning capabilities.

## Overview

Bodhi is a meta-agent framework that creates specialized agents dynamically based on task requirements. The system includes a sophisticated NLP2SQL implementation that demonstrates cross-database learning and adaptive pattern recognition.

## Features

- **Dynamic Agent Creation**: Creates specialized agents on-demand based on task requirements
- **Cross-Domain Learning**: Learns patterns across different database types and query structures
- **Agent Evolution**: Uses genetic algorithms to improve agent performance over time
- **NLP2SQL Capabilities**: Converts natural language to SQL queries for PostgreSQL and MongoDB
- **Agent-to-Agent Communication**: Built-in A2A protocol for agent collaboration
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

### Running Demos

```bash
# Core framework demonstration
python bodhi/examples/demo_bodhi_primitives.py

# NLP2SQL with emergent intelligence
python bodhi/examples/demo_emergent_nlp2sql.py
```

## Architecture

```
bodhi/
├── core/                    # Core primitives
│   ├── agent.py            # Agent and AgentDNA classes
│   ├── task.py             # Task management
│   ├── factory.py          # Agent creation and evolution
│   ├── capability.py       # Agent capabilities
│   └── result.py           # Execution results
├── specialists/             # Specialized agents
│   ├── postgres_specialist.py
│   └── mongodb_specialist.py
├── meta_agent/              # Meta-agent system
│   └── nlp2sql_meta_agent.py
├── communication/           # A2A protocols
├── tools/                   # MCP-compliant tools
├── security/               # Sandboxing
├── learning/               # Learning systems
└── examples/               # Demos and examples
```

## Core Components

### Agent

The base agent class with DNA-based configuration:

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

agent = Agent(dna)
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

## Configuration

### Environment Variables

- `BODHI_SHOW_BANNER`: Set to '0' to disable framework banner (default: '1')

### Database Configuration

Configure database connections through the specialist constructors or meta-agent configuration.

## API Reference

### Core Classes

- `Agent`: Base agent class with DNA-based configuration
- `AgentDNA`: Agent blueprint defining capabilities and behavior
- `Task`: Structured task representation
- `AgentFactory`: Creates and evolves agents
- `Capability`: Represents agent capabilities
- `Requirement`: Defines task requirements

### Specialists

- `PostgreSQLSpecialist`: PostgreSQL query generation specialist
- `MongoDBSpecialist`: MongoDB aggregation pipeline specialist

### Meta-Agent

- `NLP2SQLMetaAgent`: Main meta-agent for natural language processing

## Examples

The `bodhi/examples/` directory contains:

- `demo_bodhi_primitives.py`: Core framework demonstration
- `demo_emergent_nlp2sql.py`: NLP2SQL with learning and evolution

## Testing

Run the framework validation:

```python
# All imports and basic functionality
python -c "from bodhi import Agent, Task, AgentFactory, NLP2SQLMetaAgent; print('Framework working')"
```

## Development

### Adding New Specialists

1. Create a new specialist class inheriting from `Agent`
2. Implement `_execute_task_logic` method
3. Add specialist to the meta-agent's specialist creation logic

### Extending Capabilities

1. Add new capability types to `CapabilityType` enum
2. Update agent DNA templates in `AgentFactory`
3. Implement corresponding task execution logic

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
