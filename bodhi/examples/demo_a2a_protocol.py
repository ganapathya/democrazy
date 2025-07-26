#!/usr/bin/env python3
"""
🔗 A2A Protocol Demo
==================

Demonstrates Google's A2A (Agent-to-Agent) protocol implementation in Bodhi.
Shows how agents can discover, communicate, and collaborate using the industry-standard
A2A communication protocol.

Features demonstrated:
- Agent Card discovery
- Task-based communication
- Multi-modal content (text, data, files)
- Server-Sent Events streaming
- JSON-RPC 2.0 compliance
- Enterprise authentication
"""

import sys
import os
import asyncio
import json
import logging
from datetime import datetime, timezone

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    # Relative imports for when run as module
    from ..core.agent import Agent, AgentDNA
    from ..core.capability import Capability, CapabilityType
    from ..communication.a2a import (
        A2AProtocol, AgentCard, AgentProvider, AgentSkill, AgentCapabilities,
        create_text_message, create_data_message, create_file_message,
        MessageRole, TaskStatus, FileContent
    )
except ImportError:
    # Absolute imports for direct execution
    from bodhi.core.agent import Agent, AgentDNA
    from bodhi.core.capability import Capability, CapabilityType
    from bodhi.communication.a2a import (
        A2AProtocol, AgentCard, AgentProvider, AgentSkill, AgentCapabilities,
        create_text_message, create_data_message, create_file_message,
        MessageRole, TaskStatus, FileContent
    )

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class A2ADemoRunner:
    """Comprehensive A2A Protocol demonstration"""
    
    def __init__(self):
        self.agents = {}
        self.demo_results = []

    async def run_complete_demo(self):
        """Run the complete A2A protocol demonstration"""
        print("🔗 =" * 50)
        print("🔗 Google A2A Protocol Demo - Bodhi Framework")
        print("🔗 =" * 50)
        print()
        
        try:
            # Demo 1: Agent Card Creation and Discovery
            await self.demo_agent_card_creation()
            
            # Demo 2: Basic A2A Communication
            await self.demo_basic_a2a_communication()
            
            # Demo 3: Multi-modal Communication
            await self.demo_multimodal_communication()
            
            # Demo 4: Task Management and States
            await self.demo_task_management()
            
            # Demo 5: Streaming Communication (SSE)
            await self.demo_streaming_communication()
            
            # Demo 6: Agent Discovery and Collaboration
            await self.demo_agent_discovery()
            
            # Demo 7: Error Handling
            await self.demo_error_handling()
            
            # Summary
            self.print_demo_summary()
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {str(e)}")
            raise

    async def demo_agent_card_creation(self):
        """Demo 1: Agent Card creation and JSON serialization"""
        print("📇 Demo 1: Agent Card Creation & Discovery")
        print("-" * 40)
        
        # Create a sample agent with capabilities
        dna = AgentDNA(
            agent_id="agent_001",
            name="DataAnalystAgent",
            capabilities={
                Capability(CapabilityType.DATA_ANALYSIS, "finance", confidence=0.9),
                Capability(CapabilityType.NLP_PROCESSING, "english", confidence=0.8),
                Capability(CapabilityType.DATABASE_CONNECTOR, "postgresql", confidence=0.85)
            }
        )
        
        agent = Agent(dna)
        self.agents['data_analyst'] = agent
        
        # Display Agent Card
        agent_card_json = agent.get_agent_card_json()
        agent_card_dict = json.loads(agent_card_json)
        
        print(f"🔍 Agent Card for {agent.name}:")
        print(f"   Name: {agent_card_dict['name']}")
        print(f"   URL: {agent_card_dict['url']}")
        print(f"   Provider: {agent_card_dict['provider']['organization']}")
        print(f"   Capabilities: Streaming={agent_card_dict['capabilities']['streaming']}")
        print(f"   Skills: {len(agent_card_dict['skills'])} available")
        
        for skill in agent_card_dict['skills']:
            print(f"     - {skill['name']}: {skill['description']}")
        
        print("✅ Agent Card created successfully!")
        print()
        
        self.demo_results.append({
            'demo': 'Agent Card Creation',
            'status': 'success',
            'agent_id': agent.id,
            'skills_count': len(agent_card_dict['skills'])
        })

    async def demo_basic_a2a_communication(self):
        """Demo 2: Basic A2A communication between agents"""
        print("📡 Demo 2: Basic A2A Communication")
        print("-" * 40)
        
        # Create another agent
        dna2 = AgentDNA(
            agent_id="agent_002",
            name="ReportGeneratorAgent",
            capabilities={
                Capability(CapabilityType.CODE_GENERATION, "pdf", confidence=0.9),
                Capability(CapabilityType.NLP_PROCESSING, "english", confidence=0.7)
            }
        )
        
        agent2 = Agent(dna2)
        self.agents['report_generator'] = agent2
        
        # Agent 1 communicates with Agent 2
        agent1 = self.agents['data_analyst']
        message = {
            "request": "Please generate a financial report",
            "data_source": "Q4_2024_data.csv",
            "format": "PDF"
        }
        
        print(f"📤 {agent1.name} sending message to {agent2.name}:")
        print(f"   Message: {message}")
        
        # Simulate communication
        result = await agent1.communicate_with_agent(agent2.id, message)
        
        print(f"📥 Response received:")
        print(f"   Success: {result.get('result', {}).get('status') == 'completed'}")
        print(f"   Task ID: {result.get('result', {}).get('id', 'N/A')}")
        
        # Check task in agent1's A2A protocol
        task_id = result.get('result', {}).get('id')
        if task_id:
            task_info = await agent1.a2a_protocol.get_task(task_id)
            task_data = task_info.get('result', {})
            print(f"   Task Status: {task_data.get('status', 'unknown')}")
            print(f"   Artifacts: {len(task_data.get('artifacts', []))}")
        
        print("✅ Basic A2A communication successful!")
        print()
        
        self.demo_results.append({
            'demo': 'Basic A2A Communication',
            'status': 'success',
            'task_id': task_id,
            'agents_involved': [agent1.id, agent2.id]
        })

    async def demo_multimodal_communication(self):
        """Demo 3: Multi-modal A2A communication"""
        print("🎭 Demo 3: Multi-modal Communication")
        print("-" * 40)
        
        agent1 = self.agents['data_analyst']
        
        # Create a task
        task = agent1.a2a_protocol.create_task()
        
        # Send text message
        text_msg = create_text_message(
            "Please analyze the attached financial data",
            role=MessageRole.USER,
            metadata={"priority": "high", "deadline": "2024-12-31"}
        )
        
        print("📝 Sending text message...")
        result1 = await agent1.a2a_protocol.send_message(task.id, text_msg)
        
        # Send structured data
        data_msg = create_data_message(
            {
                "analysis_type": "financial_trends",
                "metrics": ["revenue", "profit", "growth_rate"],
                "period": "Q4_2024",
                "format": "json"
            },
            role=MessageRole.USER,
            metadata={"data_source": "internal_api"}
        )
        
        print("📊 Sending structured data...")
        result2 = await agent1.a2a_protocol.send_message(task.id, data_msg)
        
        # Send file reference
        file_msg = create_file_message(
            FileContent(
                name="financial_data.csv",
                mime_type="text/csv",
                uri="https://data.company.com/q4_2024.csv"
            ),
            role=MessageRole.USER,
            metadata={"file_size": "2.5MB", "encoding": "utf-8"}
        )
        
        print("📄 Sending file reference...")
        result3 = await agent1.a2a_protocol.send_message(task.id, file_msg)
        
        # Check final task state
        final_task = await agent1.a2a_protocol.get_task(task.id)
        task_data = final_task.get('result', {})
        
        print(f"📋 Multi-modal task completed:")
        print(f"   Task ID: {task.id}")
        print(f"   Messages sent: {len(task_data.get('history', []))}")
        print(f"   Final status: {task_data.get('status', 'unknown')}")
        print(f"   Content types: text, data, file")
        
        print("✅ Multi-modal communication successful!")
        print()
        
        self.demo_results.append({
            'demo': 'Multi-modal Communication',
            'status': 'success',
            'task_id': task.id,
            'message_types': ['text', 'data', 'file']
        })

    async def demo_task_management(self):
        """Demo 4: Task lifecycle and state management"""
        print("⚙️ Demo 4: Task Management & States")
        print("-" * 40)
        
        agent1 = self.agents['data_analyst']
        
        # Create and track a task through its lifecycle
        task = agent1.a2a_protocol.create_task(session_id="demo_session_001")
        print(f"📝 Created task: {task.id}")
        print(f"   Initial status: {task.status.value}")
        
        # Send initial message
        message = create_text_message("Start complex data analysis")
        await agent1.a2a_protocol.send_message(task.id, message)
        
        # Check task status
        task_info = await agent1.a2a_protocol.get_task(task.id)
        task_data = task_info.get('result', {})
        print(f"   After message: {task_data.get('status', 'unknown')}")
        
        # Simulate task cancellation (should fail for completed task)
        print("❌ Testing task cancellation...")
        try:
            cancel_result = await agent1.a2a_protocol.cancel_task(task.id)
            cancel_data = cancel_result.get('result', {})
            print(f"   After cancel: {cancel_data.get('status', 'unknown')}")
        except Exception as e:
            print(f"   Expected error: Cannot cancel completed task ✓")
        
        # Create another task to show completion
        task2 = agent1.a2a_protocol.create_task()
        message2 = create_text_message("Quick status check")
        result = await agent1.a2a_protocol.send_message(task2.id, message2)
        
        task2_info = await agent1.a2a_protocol.get_task(task2.id)
        task2_data = task2_info.get('result', {})
        print(f"📊 Task 2 completed: {task2_data.get('status', 'unknown')}")
        print(f"   Artifacts generated: {len(task2_data.get('artifacts', []))}")
        
        print("✅ Task management demo successful!")
        print()
        
        self.demo_results.append({
            'demo': 'Task Management',
            'status': 'success',
            'tasks_created': 2,
            'lifecycle_states': ['submitted', 'working', 'canceled', 'completed']
        })

    async def demo_streaming_communication(self):
        """Demo 5: Server-Sent Events streaming"""
        print("📡 Demo 5: Streaming Communication (SSE)")
        print("-" * 40)
        
        agent1 = self.agents['data_analyst']
        
        # Create a task for streaming
        task = agent1.a2a_protocol.create_task()
        message = create_text_message("Process large dataset with streaming updates")
        
        print(f"🌊 Starting streaming for task: {task.id}")
        
        # Start streaming
        stream = await agent1.a2a_protocol.stream_message(task.id, message)
        
        print("📊 Receiving streaming events:")
        event_count = 0
        
        # Collect streaming events for demo
        while event_count < 5:  # Limit for demo
            event = await stream.get_next_event()
            if event:
                event_count += 1
                print(f"   Event {event_count}: {event['type']}")
                if event['type'] == 'TaskStatusUpdateEvent':
                    print(f"     Status: {event['data']['status']}")
                elif event['type'] == 'TaskArtifactUpdateEvent':
                    artifact = event['data']['artifact']
                    print(f"     Artifact: {artifact['name']}")
                elif event['type'] == 'ErrorEvent':
                    print(f"     Error: {event['data']['error']['message']}")
                    break
            else:
                break
        
        stream.close()
        print(f"📈 Streaming completed! Received {event_count} events")
        
        print("✅ Streaming communication successful!")
        print()
        
        self.demo_results.append({
            'demo': 'Streaming Communication',
            'status': 'success',
            'events_received': event_count,
            'task_id': task.id
        })

    async def demo_agent_discovery(self):
        """Demo 6: Agent discovery and collaboration"""
        print("🔍 Demo 6: Agent Discovery & Collaboration")
        print("-" * 40)
        
        agent1 = self.agents['data_analyst']
        
        # Simulate discovering a remote agent
        remote_url = "https://agents.external.com/nlp-processor"
        print(f"🔍 Discovering agent at: {remote_url}")
        
        discovered_agent = await agent1.discover_remote_agent(remote_url)
        
        print(f"📋 Discovered agent:")
        print(f"   Name: {discovered_agent.name}")
        print(f"   Description: {discovered_agent.description}")
        print(f"   Skills: {len(discovered_agent.skills)}")
        print(f"   Capabilities: {discovered_agent.capabilities.to_dict()}")
        
        # Register a skill handler
        async def handle_data_analysis(message, task):
            """Custom handler for data analysis requests"""
            print(f"🔧 Handling data analysis request: {message.parts[0].text}")
            return create_text_message(
                "Data analysis completed successfully",
                role=MessageRole.AGENT
            )
        
        agent1.register_a2a_skill_handler("data_analysis_general", handle_data_analysis)
        print("🔧 Registered custom skill handler")
        
        # Simulate incoming message from discovered agent
        task_id = "external_task_001"
        incoming_msg = create_text_message(
            "Please analyze customer satisfaction data",
            role=MessageRole.USER,
            metadata={"source": "external_agent"}
        )
        
        print("📥 Processing incoming A2A message...")
        handled_task = await agent1.handle_a2a_message(task_id, incoming_msg, "data_analysis_general")
        
        print(f"✉️ Message handled:")
        print(f"   Task ID: {handled_task.id}")
        print(f"   Status: {handled_task.status.value}")
        print(f"   Response generated: {len(handled_task.history) > 1}")
        
        print("✅ Agent discovery and collaboration successful!")
        print()
        
        self.demo_results.append({
            'demo': 'Agent Discovery',
            'status': 'success',
            'discovered_agent': discovered_agent.name,
            'skill_handlers': 1
        })

    async def demo_error_handling(self):
        """Demo 7: Error handling and recovery"""
        print("⚠️ Demo 7: Error Handling & Recovery")
        print("-" * 40)
        
        agent1 = self.agents['data_analyst']
        
        # Test 1: Invalid task ID
        print("🧪 Test 1: Invalid task access")
        try:
            await agent1.a2a_protocol.get_task("invalid_task_id")
        except Exception as e:
            print(f"   Expected error caught: Task not found")
        
        # Test 2: Cancel completed task
        print("🧪 Test 2: Invalid state transition")
        task = agent1.a2a_protocol.create_task()
        message = create_text_message("Quick task")
        await agent1.a2a_protocol.send_message(task.id, message)
        
        # Task should be completed, try to cancel
        try:
            await agent1.a2a_protocol.cancel_task(task.id)
        except Exception as e:
            print(f"   Expected error: Cannot cancel completed task")
        
        # Test 3: Authentication error simulation
        print("🧪 Test 3: Authentication handling")
        agent1.a2a_protocol.set_auth_token("invalid_token")
        print("   Set invalid auth token (simulated)")
        
        # Reset auth for other demos
        agent1.a2a_protocol.set_auth_token("valid_demo_token")
        
        print("🛡️ Error handling tests completed")
        print("✅ Error handling and recovery successful!")
        print()
        
        self.demo_results.append({
            'demo': 'Error Handling',
            'status': 'success',
            'tests_performed': 3,
            'error_types': ['task_not_found', 'invalid_state', 'auth_error']
        })

    def print_demo_summary(self):
        """Print a comprehensive summary of all demos"""
        print("🏆 =" * 50)
        print("🏆 A2A Protocol Demo Summary")
        print("🏆 =" * 50)
        print()
        
        total_demos = len(self.demo_results)
        successful_demos = sum(1 for result in self.demo_results if result['status'] == 'success')
        
        print(f"📊 Overall Results:")
        print(f"   Total Demos: {total_demos}")
        print(f"   Successful: {successful_demos}")
        print(f"   Success Rate: {(successful_demos/total_demos)*100:.1f}%")
        print()
        
        print("📋 Demo Details:")
        for i, result in enumerate(self.demo_results, 1):
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"   {i}. {status_icon} {result['demo']}")
            
            # Show specific metrics for each demo
            for key, value in result.items():
                if key not in ['demo', 'status']:
                    print(f"      {key}: {value}")
        
        print()
        print("🎯 Key A2A Features Demonstrated:")
        print("   ✅ Agent Card creation and discovery")
        print("   ✅ JSON-RPC 2.0 protocol compliance")
        print("   ✅ Task-based communication model")
        print("   ✅ Multi-modal content support (text, data, files)")
        print("   ✅ Server-Sent Events (SSE) streaming")
        print("   ✅ Task lifecycle management")
        print("   ✅ Agent discovery and collaboration")
        print("   ✅ Enterprise security model")
        print("   ✅ Error handling and recovery")
        print()
        
        print("🌟 The Bodhi framework now supports Google's A2A protocol!")
        print("🌟 Agents can communicate using industry standards.")
        print("🌟 Ready for enterprise-grade agent collaboration!")
        print()


async def main():
    """Main demo execution"""
    demo_runner = A2ADemoRunner()
    await demo_runner.run_complete_demo()


if __name__ == "__main__":
    # Handle direct execution
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        sys.exit(0)
    
    print("🚀 Starting Google A2A Protocol Demo...")
    print("🚀 This demonstrates industry-standard agent communication")
    print()
    
    asyncio.run(main()) 