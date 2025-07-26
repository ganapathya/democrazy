"""
🧘 Factory Primitive - Agent Creation System
==========================================

The AgentFactory is responsible for creating specialized agents based on task requirements.
It implements sophisticated DNA generation, capability mapping, and evolutionary algorithms
for optimal agent creation.

Key Features:
- Task-driven agent creation
- DNA evolution and optimization
- Capability synthesis and specialization
- Tool assignment and configuration
- Learning-based improvement
- Multi-generation agent breeding
"""

import uuid
import json
import random
from typing import Dict, List, Set, Any, Optional, Tuple, Type
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging

from .agent import Agent, AgentDNA
from .capability import Capability, CapabilityType
from .task import Task, TaskRequirement
from ..utils.exceptions import BodhiError, AgentCreationError
from ..security.sandbox import AgentSandbox

logger = logging.getLogger(__name__)


class CreationStrategy(Enum):
    """Strategies for agent creation"""
    BASIC = "basic"  # Create basic agent with minimal capabilities
    SPECIALIZED = "specialized"  # Create highly specialized agent
    HYBRID = "hybrid"  # Combine capabilities from multiple domains
    EVOLUTIONARY = "evolutionary"  # Use genetic algorithms for optimization
    COLLABORATIVE = "collaborative"  # Design for multi-agent collaboration


class DNATemplate(Enum):
    """Pre-defined DNA templates for common agent types"""
    NLP_SPECIALIST = "nlp_specialist"
    DATABASE_EXPERT = "database_expert"
    API_CONNECTOR = "api_connector"
    DATA_ANALYST = "data_analyst"
    CODE_GENERATOR = "code_generator"
    REASONING_ENGINE = "reasoning_engine"
    LEARNING_AGENT = "learning_agent"
    ORCHESTRATOR = "orchestrator"
    COMMUNICATOR = "communicator"
    TOOL_BUILDER = "tool_builder"


@dataclass
class CreationContext:
    """Context information for agent creation"""
    # Task information
    source_task: Optional[Task] = None
    capability_requirements: Set[CapabilityType] = field(default_factory=set)
    domain_specialization: Optional[str] = None
    
    # Creation parameters
    strategy: CreationStrategy = CreationStrategy.SPECIALIZED
    template: Optional[DNATemplate] = None
    generation: int = 1
    parent_agent_ids: List[str] = field(default_factory=list)
    
    # Performance requirements
    min_confidence: float = 0.7
    max_creation_time: int = 30  # seconds
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Environment
    sandbox_config: Dict[str, Any] = field(default_factory=dict)
    available_tools: Set[str] = field(default_factory=set)
    collaboration_requirements: Set[str] = field(default_factory=set)
    
    # Learning and adaptation
    historical_performance: Dict[str, float] = field(default_factory=dict)
    feedback_data: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'source_task_id': self.source_task.id if self.source_task else None,
            'capability_requirements': [cap.value for cap in self.capability_requirements],
            'domain_specialization': self.domain_specialization,
            'strategy': self.strategy.value,
            'template': self.template.value if self.template else None,
            'generation': self.generation,
            'parent_agent_ids': self.parent_agent_ids,
            'min_confidence': self.min_confidence,
            'max_creation_time': self.max_creation_time,
            'resource_constraints': self.resource_constraints,
            'sandbox_config': self.sandbox_config,
            'available_tools': list(self.available_tools),
            'collaboration_requirements': list(self.collaboration_requirements),
            'historical_performance': self.historical_performance,
            'feedback_data': self.feedback_data
        }


class DNAEvolutionEngine:
    """Engine for evolving agent DNA using genetic algorithms"""
    
    def __init__(self):
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.selection_pressure = 0.8
        
    def evolve_dna(self, parent_dnas: List[AgentDNA], 
                   performance_scores: List[float],
                   target_capabilities: Set[CapabilityType]) -> AgentDNA:
        """Evolve new DNA from parent agents using genetic algorithms"""
        
        if not parent_dnas:
            raise AgentCreationError("Cannot evolve without parent DNA")
        
        logger.info(f"🧬 Evolving DNA from {len(parent_dnas)} parents")
        
        # Selection - choose best performing parents
        selected_parents = self._select_parents(parent_dnas, performance_scores)
        
        # Crossover - combine DNA from selected parents
        if len(selected_parents) >= 2:
            child_dna = self._crossover(selected_parents[0], selected_parents[1])
        else:
            child_dna = self._clone_with_variation(selected_parents[0])
        
        # Mutation - introduce random variations
        if random.random() < self.mutation_rate:
            child_dna = self._mutate(child_dna, target_capabilities)
        
        # Optimization - fine-tune for target capabilities
        child_dna = self._optimize_for_capabilities(child_dna, target_capabilities)
        
        return child_dna
    
    def _select_parents(self, dnas: List[AgentDNA], scores: List[float]) -> List[AgentDNA]:
        """Select best performing DNA for breeding"""
        if len(dnas) != len(scores):
            raise ValueError("DNA list and score list must have same length")
        
        # Sort by performance score
        paired = list(zip(dnas, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        
        # Select top performers (with some randomness)
        selection_size = max(1, int(len(paired) * self.selection_pressure))
        selected = [dna for dna, _ in paired[:selection_size]]
        
        # Add some random selection for diversity
        if len(paired) > selection_size:
            remaining = [dna for dna, _ in paired[selection_size:]]
            if remaining:
                selected.append(random.choice(remaining))
        
        return selected[:2]  # Return top 2 for crossover
    
    def _crossover(self, parent1: AgentDNA, parent2: AgentDNA) -> AgentDNA:
        """Create child DNA by combining traits from two parents"""
        
        # Combine capabilities (union with confidence averaging)
        combined_capabilities = set()
        cap_map = {}
        
        for cap in parent1.capabilities:
            cap_map[cap.type] = cap
        
        for cap in parent2.capabilities:
            if cap.type in cap_map:
                # Average confidence for shared capabilities
                existing_cap = cap_map[cap.type]
                avg_confidence = (existing_cap.confidence + cap.confidence) / 2
                combined_capabilities.add(Capability(
                    type=cap.type,
                    domain=existing_cap.domain,  # Keep first parent's domain
                    confidence=avg_confidence,
                    prerequisites=existing_cap.prerequisites.union(cap.prerequisites),
                    metadata={**existing_cap.metadata, **cap.metadata}
                ))
            else:
                combined_capabilities.add(cap)
        
        # Add remaining capabilities from parent1
        for cap in parent1.capabilities:
            if cap.type not in [c.type for c in combined_capabilities]:
                combined_capabilities.add(cap)
        
        # Combine other traits
        child_dna = AgentDNA(
            name=f"Hybrid_{parent1.name}_{parent2.name}",
            capabilities=combined_capabilities,
            knowledge_domains=list(set(parent1.knowledge_domains + parent2.knowledge_domains)),
            tools=parent1.tools.union(parent2.tools),
            goals=parent1.goals + [g for g in parent2.goals if g not in parent1.goals],
            learning_rate=(parent1.learning_rate + parent2.learning_rate) / 2,
            collaboration_preference=(parent1.collaboration_preference + parent2.collaboration_preference) / 2,
            risk_tolerance=(parent1.risk_tolerance + parent2.risk_tolerance) / 2,
            creativity_factor=(parent1.creativity_factor + parent2.creativity_factor) / 2,
            max_memory_size=max(parent1.max_memory_size, parent2.max_memory_size),
            max_execution_time=max(parent1.max_execution_time, parent2.max_execution_time),
            max_concurrent_tasks=max(parent1.max_concurrent_tasks, parent2.max_concurrent_tasks),
            generation=max(parent1.generation, parent2.generation) + 1
        )
        
        return child_dna
    
    def _clone_with_variation(self, parent: AgentDNA) -> AgentDNA:
        """Clone parent DNA with small variations"""
        
        # Create slight variations in behavioral traits
        variation = random.uniform(-0.1, 0.1)
        
        child_dna = AgentDNA(
            name=f"Clone_{parent.name}",
            capabilities=parent.capabilities.copy(),
            knowledge_domains=parent.knowledge_domains.copy(),
            tools=parent.tools.copy(),
            goals=parent.goals.copy(),
            learning_rate=max(0.0, min(1.0, parent.learning_rate + variation)),
            collaboration_preference=max(0.0, min(1.0, parent.collaboration_preference + variation)),
            risk_tolerance=max(0.0, min(1.0, parent.risk_tolerance + variation)),
            creativity_factor=max(0.0, min(1.0, parent.creativity_factor + variation)),
            max_memory_size=parent.max_memory_size,
            max_execution_time=parent.max_execution_time,
            max_concurrent_tasks=parent.max_concurrent_tasks,
            generation=parent.generation + 1
        )
        
        return child_dna
    
    def _mutate(self, dna: AgentDNA, target_capabilities: Set[CapabilityType]) -> AgentDNA:
        """Introduce random mutations in DNA"""
        
        mutated_capabilities = dna.capabilities.copy()
        
        # Randomly adjust capability confidence
        for cap in list(mutated_capabilities):
            if random.random() < 0.3:  # 30% chance to mutate each capability
                mutation = random.uniform(-0.2, 0.2)
                new_confidence = max(0.1, min(1.0, cap.confidence + mutation))
                
                # Create new capability with mutated confidence
                mutated_capabilities.remove(cap)
                mutated_capabilities.add(Capability(
                    type=cap.type,
                    domain=cap.domain,
                    confidence=new_confidence,
                    prerequisites=cap.prerequisites,
                    metadata=cap.metadata
                ))
        
        # Possibly add new capabilities from target set
        missing_capabilities = target_capabilities - {cap.type for cap in mutated_capabilities}
        if missing_capabilities and random.random() < 0.4:  # 40% chance to add capability
            new_cap_type = random.choice(list(missing_capabilities))
            mutated_capabilities.add(Capability(
                type=new_cap_type,
                domain="general",
                confidence=random.uniform(0.5, 0.8),
                prerequisites=set(),
                metadata={"source": "mutation"}
            ))
        
        # Create mutated DNA
        mutated_dna = AgentDNA(
            name=f"Mutant_{dna.name}",
            capabilities=mutated_capabilities,
            knowledge_domains=dna.knowledge_domains.copy(),
            tools=dna.tools.copy(),
            goals=dna.goals.copy(),
            learning_rate=dna.learning_rate,
            collaboration_preference=dna.collaboration_preference,
            risk_tolerance=dna.risk_tolerance,
            creativity_factor=dna.creativity_factor,
            max_memory_size=dna.max_memory_size,
            max_execution_time=dna.max_execution_time,
            max_concurrent_tasks=dna.max_concurrent_tasks,
            generation=dna.generation
        )
        
        return mutated_dna
    
    def _optimize_for_capabilities(self, dna: AgentDNA, target_capabilities: Set[CapabilityType]) -> AgentDNA:
        """Optimize DNA for specific target capabilities"""
        
        optimized_capabilities = dna.capabilities.copy()
        
        # Boost confidence for target capabilities
        for cap in list(optimized_capabilities):
            if cap.type in target_capabilities:
                optimized_capabilities.remove(cap)
                optimized_capabilities.add(Capability(
                    type=cap.type,
                    domain=cap.domain,
                    confidence=min(1.0, cap.confidence + 0.1),  # Boost confidence
                    prerequisites=cap.prerequisites,
                    metadata={**cap.metadata, "optimized": True}
                ))
        
        # Add missing target capabilities
        existing_types = {cap.type for cap in optimized_capabilities}
        for cap_type in target_capabilities:
            if cap_type not in existing_types:
                optimized_capabilities.add(Capability(
                    type=cap_type,
                    domain="general",
                    confidence=0.7,
                    prerequisites=set(),
                    metadata={"source": "optimization"}
                ))
        
        # Create optimized DNA
        optimized_dna = AgentDNA(
            name=f"Optimized_{dna.name}",
            capabilities=optimized_capabilities,
            knowledge_domains=dna.knowledge_domains,
            tools=dna.tools,
            goals=dna.goals + [f"Excel at {cap.value}" for cap in target_capabilities],
            learning_rate=dna.learning_rate,
            collaboration_preference=dna.collaboration_preference,
            risk_tolerance=dna.risk_tolerance,
            creativity_factor=dna.creativity_factor,
            max_memory_size=dna.max_memory_size,
            max_execution_time=dna.max_execution_time,
            max_concurrent_tasks=dna.max_concurrent_tasks,
            generation=dna.generation
        )
        
        return optimized_dna


class AgentFactory:
    """
    Sophisticated factory for creating specialized agents
    
    The AgentFactory implements multiple creation strategies:
    - Template-based creation for common agent types
    - Task-driven specialized agent generation
    - Evolutionary algorithms for DNA optimization
    - Multi-agent collaboration design
    """
    
    def __init__(self):
        self.dna_templates: Dict[DNATemplate, AgentDNA] = {}
        self.evolution_engine = DNAEvolutionEngine()
        self.creation_history: List[Dict[str, Any]] = []
        self.performance_registry: Dict[str, List[float]] = {}
        
        # Initialize built-in templates
        self._initialize_templates()
        
        logger.info("🏭 AgentFactory initialized with DNA templates and evolution engine")
    
    def _initialize_templates(self):
        """Initialize built-in DNA templates for common agent types"""
        
        # NLP Specialist Template
        self.dna_templates[DNATemplate.NLP_SPECIALIST] = AgentDNA(
            name="NLPSpecialist",
            capabilities={
                Capability(CapabilityType.NLP_PROCESSING, "general", 0.9),
                Capability(CapabilityType.REASONING, "linguistic", 0.8),
                Capability(CapabilityType.COMMUNICATION, "natural_language", 0.9)
            },
            knowledge_domains=["natural_language_processing", "linguistics", "text_analysis", "intent_recognition"],
            tools={"text_tokenizer", "intent_classifier", "entity_extractor", "sentiment_analyzer"},
            goals=[
                "Process natural language with high accuracy",
                "Understand context and intent",
                "Generate coherent responses",
                "Learn from language patterns"
            ],
            learning_rate=0.15,
            collaboration_preference=0.7,
            creativity_factor=0.6
        )
        
        # Database Expert Template
        self.dna_templates[DNATemplate.DATABASE_EXPERT] = AgentDNA(
            name="DatabaseExpert",
            capabilities={
                Capability(CapabilityType.DATABASE_CONNECTOR, "sql", 0.95),
                Capability(CapabilityType.DATA_ANALYSIS, "relational", 0.8),
                Capability(CapabilityType.NLP_PROCESSING, "sql", 0.7)
            },
            knowledge_domains=["sql", "database_design", "query_optimization", "data_modeling"],
            tools={"sql_executor", "schema_analyzer", "query_optimizer", "connection_manager"},
            goals=[
                "Generate efficient SQL queries",
                "Optimize database performance",
                "Understand data relationships",
                "Convert natural language to SQL"
            ],
            learning_rate=0.12,
            collaboration_preference=0.5,
            risk_tolerance=0.2
        )
        
        # API Connector Template
        self.dna_templates[DNATemplate.API_CONNECTOR] = AgentDNA(
            name="APIConnector",
            capabilities={
                Capability(CapabilityType.API_CONNECTOR, "rest", 0.9),
                Capability(CapabilityType.COMMUNICATION, "http", 0.85),
                Capability(CapabilityType.DATA_ANALYSIS, "json", 0.7)
            },
            knowledge_domains=["rest_apis", "http_protocols", "authentication", "data_formats"],
            tools={"http_client", "auth_handler", "response_parser", "rate_limiter"},
            goals=[
                "Connect to external APIs reliably",
                "Handle authentication and rate limiting",
                "Parse and transform API responses",
                "Optimize API usage patterns"
            ],
            learning_rate=0.1,
            collaboration_preference=0.6,
            risk_tolerance=0.4
        )
        
        # Add more templates...
        logger.info(f"🧬 Initialized {len(self.dna_templates)} DNA templates")
    
    def create_agent_for_task(self, task: Task, 
                            creation_context: Optional[CreationContext] = None) -> Agent:
        """
        Create an agent specifically designed to handle the given task
        
        This is the main factory method that analyzes task requirements
        and creates an optimal agent using the best strategy.
        """
        
        if creation_context is None:
            creation_context = CreationContext(
                source_task=task,
                capability_requirements=task.required_capabilities,
                strategy=CreationStrategy.SPECIALIZED
            )
        
        logger.info(f"🏭 Creating agent for task: {task.intent}")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Analyze task requirements
            analysis = self._analyze_task_requirements(task)
            
            # Choose optimal creation strategy
            strategy = self._choose_creation_strategy(task, analysis, creation_context)
            
            # Generate agent DNA
            agent_dna = self._generate_agent_dna(task, analysis, strategy, creation_context)
            
            # Create agent instance
            sandbox = self._create_agent_sandbox(agent_dna, creation_context)
            agent = Agent(agent_dna, sandbox)
            
            # Configure agent tools
            self._configure_agent_tools(agent, task, creation_context)
            
            # Record creation
            creation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._record_creation(agent, task, creation_context, creation_time)
            
            logger.info(f"✅ Created agent {agent.name} ({agent.id}) in {creation_time:.2f}s")
            return agent
            
        except Exception as e:
            logger.error(f"❌ Agent creation failed: {str(e)}")
            raise AgentCreationError(f"Failed to create agent for task: {str(e)}")
    
    def create_agent_from_template(self, template: DNATemplate,
                                 customizations: Optional[Dict[str, Any]] = None) -> Agent:
        """Create an agent from a predefined DNA template"""
        
        if template not in self.dna_templates:
            raise AgentCreationError(f"Unknown DNA template: {template.value}")
        
        base_dna = self.dna_templates[template]
        
        # Apply customizations
        if customizations:
            agent_dna = self._customize_dna(base_dna, customizations)
        else:
            agent_dna = base_dna
        
        # Create agent
        sandbox = AgentSandbox(agent_dna.agent_id)
        agent = Agent(agent_dna, sandbox)
        
        logger.info(f"🧬 Created agent from template: {template.value}")
        return agent
    
    def evolve_agent(self, parent_agents: List[Agent], 
                   target_capabilities: Set[CapabilityType],
                   performance_data: Optional[Dict[str, float]] = None) -> Agent:
        """
        Create a new agent by evolving DNA from successful parent agents
        
        This implements genetic algorithms to breed better agents
        """
        
        if not parent_agents:
            raise AgentCreationError("Cannot evolve without parent agents")
        
        logger.info(f"🧬 Evolving agent from {len(parent_agents)} parents")
        
        # Extract parent DNA and performance scores
        parent_dnas = [agent.dna for agent in parent_agents]
        
        if performance_data:
            scores = [performance_data.get(agent.id, 0.5) for agent in parent_agents]
        else:
            # Use current performance metrics
            scores = [
                agent.performance_metrics.get('success_rate', 0.5) 
                for agent in parent_agents
            ]
        
        # Evolve new DNA
        evolved_dna = self.evolution_engine.evolve_dna(
            parent_dnas, scores, target_capabilities
        )
        
        # Create evolved agent
        sandbox = AgentSandbox(evolved_dna.agent_id)
        evolved_agent = Agent(evolved_dna, sandbox)
        
        logger.info(f"🎉 Evolved agent {evolved_agent.name} (generation {evolved_dna.generation})")
        return evolved_agent
    
    def _analyze_task_requirements(self, task: Task) -> Dict[str, Any]:
        """Analyze task to understand requirements and complexity"""
        
        analysis = {
            'required_capabilities': task.required_capabilities,
            'optional_capabilities': task.optional_capabilities,
            'complexity': task.estimate_complexity(),
            'domain_hints': [],
            'collaboration_needed': len(task.context.collaboration_agents) > 0,
            'security_requirements': list(task.context.permissions),
            'performance_requirements': {
                'max_execution_time': task.max_execution_time,
                'expected_format': task.expected_output_format
            }
        }
        
        # Extract domain hints from task context
        if task.context.input_data:
            if 'database' in str(task.context.input_data).lower():
                analysis['domain_hints'].append('database')
            if 'api' in str(task.context.input_data).lower():
                analysis['domain_hints'].append('api')
            if 'sql' in str(task.context.input_data).lower():
                analysis['domain_hints'].append('sql')
        
        # Analyze intent for additional context
        intent_lower = task.intent.lower()
        if any(word in intent_lower for word in ['convert', 'transform', 'translate']):
            analysis['domain_hints'].append('transformation')
        if any(word in intent_lower for word in ['analyze', 'examine', 'study']):
            analysis['domain_hints'].append('analysis')
        if any(word in intent_lower for word in ['generate', 'create', 'build']):
            analysis['domain_hints'].append('generation')
        
        return analysis
    
    def _choose_creation_strategy(self, task: Task, analysis: Dict[str, Any], 
                                context: CreationContext) -> CreationStrategy:
        """Choose the optimal creation strategy based on task analysis"""
        
        # Use explicit strategy if provided
        if context.strategy != CreationStrategy.BASIC:
            return context.strategy
        
        # Choose strategy based on task characteristics
        complexity = analysis['complexity']
        num_capabilities = len(analysis['required_capabilities'])
        
        if complexity > 0.8 or num_capabilities > 3:
            return CreationStrategy.EVOLUTIONARY
        elif analysis['collaboration_needed']:
            return CreationStrategy.COLLABORATIVE
        elif len(analysis['domain_hints']) > 1:
            return CreationStrategy.HYBRID
        else:
            return CreationStrategy.SPECIALIZED
    
    def _generate_agent_dna(self, task: Task, analysis: Dict[str, Any],
                          strategy: CreationStrategy, context: CreationContext) -> AgentDNA:
        """Generate agent DNA based on strategy and requirements"""
        
        if strategy == CreationStrategy.SPECIALIZED:
            return self._create_specialized_dna(task, analysis, context)
        elif strategy == CreationStrategy.HYBRID:
            return self._create_hybrid_dna(task, analysis, context)
        elif strategy == CreationStrategy.EVOLUTIONARY:
            return self._create_evolutionary_dna(task, analysis, context)
        elif strategy == CreationStrategy.COLLABORATIVE:
            return self._create_collaborative_dna(task, analysis, context)
        else:
            return self._create_basic_dna(task, analysis, context)
    
    def _create_specialized_dna(self, task: Task, analysis: Dict[str, Any],
                              context: CreationContext) -> AgentDNA:
        """Create DNA for a highly specialized agent"""
        
        primary_capability = next(iter(analysis['required_capabilities']))
        
        # Map capability to template
        template_mapping = {
            CapabilityType.NLP_PROCESSING: DNATemplate.NLP_SPECIALIST,
            CapabilityType.DATABASE_CONNECTOR: DNATemplate.DATABASE_EXPERT,
            CapabilityType.API_CONNECTOR: DNATemplate.API_CONNECTOR,
            CapabilityType.DATA_ANALYSIS: DNATemplate.DATA_ANALYST,
            CapabilityType.CODE_GENERATION: DNATemplate.CODE_GENERATOR,
            CapabilityType.REASONING: DNATemplate.REASONING_ENGINE
        }
        
        template = template_mapping.get(primary_capability, DNATemplate.NLP_SPECIALIST)
        base_dna = self.dna_templates[template]
        
        # Enhance capabilities for task requirements
        enhanced_capabilities = base_dna.capabilities.copy()
        
        for req_cap in analysis['required_capabilities']:
            if req_cap not in {cap.type for cap in enhanced_capabilities}:
                enhanced_capabilities.add(Capability(
                    type=req_cap,
                    domain=context.domain_specialization or "general",
                    confidence=0.8,
                    prerequisites=set(),
                    metadata={"source": "task_requirement"}
                ))
        
        # Create specialized DNA
        specialized_dna = AgentDNA(
            name=f"Specialized_{primary_capability.value}_{random.randint(1000, 9999)}",
            capabilities=enhanced_capabilities,
            knowledge_domains=base_dna.knowledge_domains + analysis['domain_hints'],
            tools=base_dna.tools.copy(),
            goals=base_dna.goals + [f"Excel at task: {task.intent[:50]}"],
            learning_rate=base_dna.learning_rate,
            collaboration_preference=base_dna.collaboration_preference,
            risk_tolerance=base_dna.risk_tolerance,
            creativity_factor=base_dna.creativity_factor,
            max_memory_size=base_dna.max_memory_size,
            max_execution_time=min(base_dna.max_execution_time, task.max_execution_time),
            max_concurrent_tasks=base_dna.max_concurrent_tasks,
            generation=context.generation
        )
        
        return specialized_dna
    
    def _create_hybrid_dna(self, task: Task, analysis: Dict[str, Any],
                         context: CreationContext) -> AgentDNA:
        """Create DNA that combines multiple capability domains"""
        
        # Find relevant templates for required capabilities
        relevant_templates = []
        for capability in analysis['required_capabilities']:
            for template, dna in self.dna_templates.items():
                if any(cap.type == capability for cap in dna.capabilities):
                    relevant_templates.append(dna)
                    break
        
        if not relevant_templates:
            # Fall back to specialized approach
            return self._create_specialized_dna(task, analysis, context)
        
        # Combine capabilities from relevant templates
        combined_capabilities = set()
        combined_knowledge = []
        combined_tools = set()
        combined_goals = []
        
        for template_dna in relevant_templates:
            combined_capabilities.update(template_dna.capabilities)
            combined_knowledge.extend(template_dna.knowledge_domains)
            combined_tools.update(template_dna.tools)
            combined_goals.extend(template_dna.goals)
        
        # Average behavioral traits
        avg_learning_rate = sum(dna.learning_rate for dna in relevant_templates) / len(relevant_templates)
        avg_collaboration = sum(dna.collaboration_preference for dna in relevant_templates) / len(relevant_templates)
        avg_risk_tolerance = sum(dna.risk_tolerance for dna in relevant_templates) / len(relevant_templates)
        avg_creativity = sum(dna.creativity_factor for dna in relevant_templates) / len(relevant_templates)
        
        # Create hybrid DNA
        hybrid_dna = AgentDNA(
            name=f"Hybrid_{'_'.join(analysis['domain_hints'])}_{random.randint(1000, 9999)}",
            capabilities=combined_capabilities,
            knowledge_domains=list(set(combined_knowledge)),
            tools=combined_tools,
            goals=list(set(combined_goals)),
            learning_rate=avg_learning_rate,
            collaboration_preference=avg_collaboration,
            risk_tolerance=avg_risk_tolerance,
            creativity_factor=avg_creativity,
            max_memory_size=max(dna.max_memory_size for dna in relevant_templates),
            max_execution_time=min(max(dna.max_execution_time for dna in relevant_templates), task.max_execution_time),
            max_concurrent_tasks=max(dna.max_concurrent_tasks for dna in relevant_templates),
            generation=context.generation
        )
        
        return hybrid_dna
    
    def _create_evolutionary_dna(self, task: Task, analysis: Dict[str, Any],
                               context: CreationContext) -> AgentDNA:
        """Create DNA using evolutionary algorithms"""
        
        if context.parent_agent_ids:
            # Use actual parent agents if available
            # This would require access to agent registry
            # For now, simulate with templates
            parent_dnas = list(self.dna_templates.values())[:2]
            performance_scores = [0.8, 0.7]  # Simulated scores
        else:
            # Use relevant templates as "ancestors"
            parent_dnas = list(self.dna_templates.values())[:3]
            performance_scores = [0.7, 0.8, 0.6]
        
        # Evolve DNA using evolution engine
        evolved_dna = self.evolution_engine.evolve_dna(
            parent_dnas, performance_scores, analysis['required_capabilities']
        )
        
        return evolved_dna
    
    def _create_collaborative_dna(self, task: Task, analysis: Dict[str, Any],
                                context: CreationContext) -> AgentDNA:
        """Create DNA optimized for multi-agent collaboration"""
        
        # Start with a base template
        base_template = DNATemplate.COMMUNICATOR
        base_dna = self.dna_templates[base_template]
        
        # Enhance collaboration capabilities
        enhanced_capabilities = base_dna.capabilities.copy()
        enhanced_capabilities.add(Capability(
            type=CapabilityType.COMMUNICATION,
            domain="collaboration",
            confidence=0.95,
            prerequisites=set(),
            metadata={"optimized_for": "multi_agent"}
        ))
        
        # Add task-specific capabilities
        for req_cap in analysis['required_capabilities']:
            enhanced_capabilities.add(Capability(
                type=req_cap,
                domain="collaborative",
                confidence=0.7,
                prerequisites=set(),
                metadata={"source": "collaboration_requirement"}
            ))
        
        # Create collaborative DNA
        collaborative_dna = AgentDNA(
            name=f"Collaborative_{task.task_type.value}_{random.randint(1000, 9999)}",
            capabilities=enhanced_capabilities,
            knowledge_domains=base_dna.knowledge_domains + ["collaboration", "coordination"],
            tools=base_dna.tools.union({"message_router", "task_coordinator", "consensus_builder"}),
            goals=base_dna.goals + [
                "Collaborate effectively with other agents",
                "Coordinate task execution",
                "Share knowledge and insights"
            ],
            learning_rate=base_dna.learning_rate,
            collaboration_preference=0.9,  # High collaboration preference
            risk_tolerance=0.3,  # Conservative for collaboration
            creativity_factor=base_dna.creativity_factor,
            max_memory_size=base_dna.max_memory_size,
            max_execution_time=base_dna.max_execution_time,
            max_concurrent_tasks=base_dna.max_concurrent_tasks,
            generation=context.generation
        )
        
        return collaborative_dna
    
    def _create_basic_dna(self, task: Task, analysis: Dict[str, Any],
                        context: CreationContext) -> AgentDNA:
        """Create basic DNA with minimal capabilities"""
        
        # Create minimal capabilities for task
        basic_capabilities = set()
        for req_cap in analysis['required_capabilities']:
            basic_capabilities.add(Capability(
                type=req_cap,
                domain="basic",
                confidence=0.6,
                prerequisites=set(),
                metadata={"source": "basic_creation"}
            ))
        
        # Create basic DNA
        basic_dna = AgentDNA(
            name=f"Basic_{task.task_type.value}_{random.randint(1000, 9999)}",
            capabilities=basic_capabilities,
            knowledge_domains=["general"],
            tools={"basic_processor"},
            goals=[f"Complete task: {task.intent[:50]}"],
            learning_rate=0.1,
            collaboration_preference=0.5,
            risk_tolerance=0.5,
            creativity_factor=0.3,
            generation=context.generation
        )
        
        return basic_dna
    
    def _customize_dna(self, base_dna: AgentDNA, customizations: Dict[str, Any]) -> AgentDNA:
        """Apply customizations to base DNA"""
        
        # Create a copy of base DNA
        custom_dna = AgentDNA(
            name=customizations.get('name', base_dna.name),
            capabilities=base_dna.capabilities.copy(),
            knowledge_domains=base_dna.knowledge_domains.copy(),
            tools=base_dna.tools.copy(),
            goals=base_dna.goals.copy(),
            learning_rate=customizations.get('learning_rate', base_dna.learning_rate),
            collaboration_preference=customizations.get('collaboration_preference', base_dna.collaboration_preference),
            risk_tolerance=customizations.get('risk_tolerance', base_dna.risk_tolerance),
            creativity_factor=customizations.get('creativity_factor', base_dna.creativity_factor),
            max_memory_size=customizations.get('max_memory_size', base_dna.max_memory_size),
            max_execution_time=customizations.get('max_execution_time', base_dna.max_execution_time),
            max_concurrent_tasks=customizations.get('max_concurrent_tasks', base_dna.max_concurrent_tasks),
            generation=base_dna.generation
        )
        
        return custom_dna
    
    def _create_agent_sandbox(self, dna: AgentDNA, context: CreationContext) -> AgentSandbox:
        """Create appropriate sandbox for agent"""
        sandbox_config = context.sandbox_config or {}
        return AgentSandbox(dna.agent_id, sandbox_config)
    
    def _configure_agent_tools(self, agent: Agent, task: Task, context: CreationContext):
        """Configure agent with appropriate tools"""
        # Tool configuration will be implemented with the tools module
        pass
    
    def _record_creation(self, agent: Agent, task: Task, context: CreationContext, creation_time: float):
        """Record agent creation for analysis and improvement"""
        
        creation_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'agent_id': agent.id,
            'agent_name': agent.name,
            'task_id': task.id,
            'task_intent': task.intent,
            'strategy': context.strategy.value,
            'template': context.template.value if context.template else None,
            'generation': agent.dna.generation,
            'capabilities': [cap.type.value for cap in agent.capabilities],
            'creation_time': creation_time,
            'complexity': task.estimate_complexity()
        }
        
        self.creation_history.append(creation_record)
        
        # Limit history size
        if len(self.creation_history) > 1000:
            self.creation_history = self.creation_history[-500:]
    
    def get_creation_stats(self) -> Dict[str, Any]:
        """Get comprehensive creation statistics"""
        
        if not self.creation_history:
            return {'total_created': 0}
        
        # Calculate statistics
        total_created = len(self.creation_history)
        avg_creation_time = sum(record['creation_time'] for record in self.creation_history) / total_created
        
        # Strategy distribution
        strategy_counts = {}
        for record in self.creation_history:
            strategy = record['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Generation distribution
        generation_counts = {}
        for record in self.creation_history:
            gen = record['generation']
            generation_counts[gen] = generation_counts.get(gen, 0) + 1
        
        return {
            'total_created': total_created,
            'avg_creation_time': avg_creation_time,
            'strategy_distribution': strategy_counts,
            'generation_distribution': generation_counts,
            'templates_available': len(self.dna_templates),
            'latest_creation': self.creation_history[-1]['timestamp']
        }
    
    def __str__(self) -> str:
        return f"AgentFactory(templates={len(self.dna_templates)}, created={len(self.creation_history)})" 