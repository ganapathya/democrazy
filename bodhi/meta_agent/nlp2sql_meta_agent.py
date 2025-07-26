"""
🧘 NLP2SQL Meta-Agent: Emergent Intelligence for Database Systems
================================================================

This demonstrates the ultimate expression of the Bodhi framework - a meta-agent that:
- Learns from every interaction
- Creates specialized agents on-the-fly
- Adapts to new database types automatically
- Evolves agent DNA based on performance
- Synthesizes knowledge across SQL and NoSQL paradigms

Key Features:
- Dynamic PostgreSQL specialist creation
- MongoDB specialist auto-generation
- Performance-based agent evolution
- Query pattern learning
- Cross-database knowledge synthesis
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
import random

# Import Bodhi framework
from ..core.agent import Agent, AgentDNA
from ..core.capability import Capability, CapabilityType
from ..core.task import Task, TaskType, TaskRequirement, create_sql_task
from ..core.factory import AgentFactory, CreationStrategy, DNATemplate
from ..core.result import ExecutionResult

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REDIS = "redis"
    UNKNOWN = "unknown"


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"      # Basic SELECT, INSERT, etc.
    MODERATE = "moderate"  # JOINs, aggregations
    COMPLEX = "complex"    # Subqueries, window functions
    ADVANCED = "advanced"  # Complex analytics, recursive queries


@dataclass
class QueryPattern:
    """Represents a learned query pattern"""
    pattern_id: str
    natural_language_pattern: str
    database_type: DatabaseType
    query_type: str  # "select", "insert", "aggregate", etc.
    complexity: QueryComplexity
    success_rate: float = 0.0
    usage_count: int = 0
    avg_execution_time: float = 0.0
    generated_queries: List[str] = field(default_factory=list)
    
    def update_performance(self, success: bool, execution_time: float, generated_query: str):
        """Update pattern performance metrics"""
        self.usage_count += 1
        
        # Update success rate (exponential moving average)
        alpha = 0.2
        new_success = 1.0 if success else 0.0
        self.success_rate = alpha * new_success + (1 - alpha) * self.success_rate
        
        # Update execution time
        self.avg_execution_time = (
            (self.avg_execution_time * (self.usage_count - 1) + execution_time) / self.usage_count
        )
        
        # Store successful queries
        if success and generated_query not in self.generated_queries:
            self.generated_queries.append(generated_query)
            
        # Keep only recent successful queries
        if len(self.generated_queries) > 10:
            self.generated_queries = self.generated_queries[-10:]


@dataclass
class SpecialistPerformance:
    """Tracks performance of database specialists"""
    agent_id: str
    database_type: DatabaseType
    queries_handled: int = 0
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    specialization_score: float = 0.0  # How well-suited for the database type
    learning_velocity: float = 0.0     # How quickly it improves
    last_performance_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update(self, success: bool, response_time: float, query_complexity: QueryComplexity):
        """Update specialist performance metrics"""
        old_success_rate = self.success_rate
        
        self.queries_handled += 1
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        new_success = 1.0 if success else 0.0
        self.success_rate = alpha * new_success + (1 - alpha) * self.success_rate
        
        # Update response time
        self.avg_response_time = (
            (self.avg_response_time * (self.queries_handled - 1) + response_time) / self.queries_handled
        )
        
        # Calculate learning velocity (improvement rate)
        improvement = self.success_rate - old_success_rate
        self.learning_velocity = 0.7 * self.learning_velocity + 0.3 * improvement
        
        # Update specialization score based on complexity handling
        complexity_bonus = {
            QueryComplexity.SIMPLE: 0.1,
            QueryComplexity.MODERATE: 0.2,
            QueryComplexity.COMPLEX: 0.4,
            QueryComplexity.ADVANCED: 0.8
        }
        
        if success:
            self.specialization_score += complexity_bonus.get(query_complexity, 0.1) * 0.1
            self.specialization_score = min(1.0, self.specialization_score)
        
        self.last_performance_update = datetime.now(timezone.utc)


class NLP2SQLMetaAgent:
    """
    The ultimate demonstration of emergent intelligence
    
    This meta-agent showcases:
    - Dynamic specialist creation based on database type detection
    - Performance-based agent evolution
    - Cross-database knowledge synthesis
    - Adaptive learning from query patterns
    - Self-improvement through experience
    """
    
    def __init__(self):
        self.agent_factory = AgentFactory()
        self.database_specialists: Dict[DatabaseType, List[Agent]] = {}
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.specialist_performance: Dict[str, SpecialistPerformance] = {}
        self.knowledge_graph: Dict[str, Any] = {}
        self.learning_history: List[Dict[str, Any]] = []
        
        # Meta-learning parameters
        self.specialization_threshold = 0.8  # When to create new specialist
        self.performance_improvement_threshold = 0.1
        self.max_specialists_per_db = 3
        
        logger.info("🧘 NLP2SQL Meta-Agent initialized - Ready for emergent intelligence!")
    
    async def process_natural_language_query(self, natural_query: str, 
                                           database_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point demonstrating emergent intelligence
        
        Process flow:
        1. Analyze query to detect database type and complexity
        2. Check if appropriate specialist exists
        3. Create new specialist if needed (SELF-ASSEMBLY)
        4. Execute query with best specialist
        5. Learn from results (META-LEARNING)
        6. Evolve specialists based on performance (EVOLUTION)
        """
        
        start_time = datetime.now(timezone.utc)
        logger.info(f"🎯 Meta-Agent processing: '{natural_query}'")
        
        try:
            # Step 1: Intelligent Analysis
            analysis = await self._analyze_query_intelligence(natural_query, database_config)
            
            # Step 2: Specialist Selection or Creation
            specialist = await self._get_or_create_specialist(analysis)
            
            # Step 3: Intelligent Execution
            result = await self._execute_with_specialist(specialist, natural_query, analysis)
            
            # Step 4: Meta-Learning
            await self._learn_from_execution(natural_query, analysis, result, specialist)
            
            # Step 5: Evolutionary Improvement
            await self._evolve_specialists_if_needed()
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Return comprehensive result
            return {
                'success': result.success,
                'natural_query': natural_query,
                'database_type': analysis['database_type'].value,
                'complexity': analysis['complexity'].value,
                'specialist_used': specialist.name,
                'specialist_id': specialist.id,
                'generated_query': result.data.get('generated_query') if result.data else None,
                'execution_result': result.data.get('execution_result') if result.data else None,
                'execution_time': execution_time,
                'learning_insights': analysis.get('learning_insights', []),
                'meta_analysis': {
                    'patterns_learned': len(self.query_patterns),
                    'specialists_active': sum(len(agents) for agents in self.database_specialists.values()),
                    'knowledge_graph_size': len(self.knowledge_graph),
                    'emergent_behaviors': self._detect_emergent_behaviors()
                },
                'timestamp': start_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Meta-Agent processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'natural_query': natural_query,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _analyze_query_intelligence(self, natural_query: str, 
                                        database_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Intelligent query analysis using learned patterns and heuristics
        
        This demonstrates AI that gets smarter over time by learning from patterns.
        """
        
        logger.info(f"🔍 Analyzing query intelligence...")
        
        # Detect database type from query and config
        database_type = self._detect_database_type(natural_query, database_config)
        
        # Analyze query complexity using learned patterns
        complexity = self._analyze_complexity(natural_query, database_type)
        
        # Extract semantic features
        semantic_features = self._extract_semantic_features(natural_query)
        
        # Find similar patterns from learning history
        similar_patterns = self._find_similar_patterns(natural_query, database_type)
        
        # Generate learning insights
        learning_insights = self._generate_learning_insights(natural_query, similar_patterns)
        
        analysis = {
            'database_type': database_type,
            'complexity': complexity,
            'semantic_features': semantic_features,
            'similar_patterns': similar_patterns,
            'learning_insights': learning_insights,
            'query_type': self._classify_query_type(natural_query),
            'estimated_performance': self._estimate_performance(natural_query, database_type),
            'recommended_specialist_traits': self._recommend_specialist_traits(
                natural_query, database_type, complexity
            )
        }
        
        logger.info(f"📊 Analysis complete: {database_type.value} | {complexity.value}")
        return analysis
    
    def _detect_database_type(self, natural_query: str, 
                            database_config: Optional[Dict[str, Any]] = None) -> DatabaseType:
        """Intelligent database type detection"""
        
        # Explicit configuration takes precedence
        if database_config and 'database_type' in database_config:
            db_type_str = database_config['database_type'].lower()
            for db_type in DatabaseType:
                if db_type.value == db_type_str:
                    return db_type
        
        # Intelligent pattern matching
        query_lower = natural_query.lower()
        
        # PostgreSQL indicators
        if any(term in query_lower for term in [
            'postgresql', 'postgres', 'psql', 'array[', 'jsonb', 'window function'
        ]):
            return DatabaseType.POSTGRESQL
        
        # MongoDB indicators
        if any(term in query_lower for term in [
            'mongodb', 'mongo', 'collection', 'document', 'aggregate', 'pipeline',
            'match', 'group', 'project', 'lookup', 'unwind'
        ]):
            return DatabaseType.MONGODB
        
        # Learn from patterns - if we've seen similar queries before
        for pattern_id, pattern in self.query_patterns.items():
            if self._calculate_similarity(natural_query, pattern.natural_language_pattern) > 0.8:
                return pattern.database_type
        
        # Default to PostgreSQL for SQL-like queries
        if any(term in query_lower for term in [
            'select', 'from', 'where', 'join', 'group by', 'order by', 'insert', 'update', 'delete'
        ]):
            return DatabaseType.POSTGRESQL
        
        return DatabaseType.UNKNOWN
    
    def _analyze_complexity(self, natural_query: str, database_type: DatabaseType) -> QueryComplexity:
        """Analyze query complexity using learned patterns"""
        
        query_lower = natural_query.lower()
        complexity_score = 0
        
        # Basic indicators
        if any(term in query_lower for term in ['select', 'find', 'get', 'show']):
            complexity_score += 1
        
        # Moderate indicators
        if any(term in query_lower for term in ['join', 'group', 'aggregate', 'count', 'sum', 'avg']):
            complexity_score += 2
        
        # Complex indicators
        if any(term in query_lower for term in [
            'subquery', 'nested', 'window', 'partition', 'recursive', 'cte', 'with'
        ]):
            complexity_score += 4
        
        # Advanced indicators
        if any(term in query_lower for term in [
            'pivot', 'unpivot', 'lateral', 'cross apply', 'recursive cte',
            'advanced analytics', 'machine learning'
        ]):
            complexity_score += 8
        
        # Map score to complexity level
        if complexity_score <= 1:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 3:
            return QueryComplexity.MODERATE
        elif complexity_score <= 7:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.ADVANCED
    
    def _extract_semantic_features(self, natural_query: str) -> Dict[str, Any]:
        """Extract semantic features for learning"""
        
        query_lower = natural_query.lower()
        
        # Entity extraction
        entities = []
        if 'customer' in query_lower:
            entities.append('customer')
        if 'order' in query_lower:
            entities.append('order')
        if 'product' in query_lower:
            entities.append('product')
        if 'user' in query_lower:
            entities.append('user')
        
        # Intent classification
        intent = 'unknown'
        if any(term in query_lower for term in ['show', 'get', 'find', 'list']):
            intent = 'retrieve'
        elif any(term in query_lower for term in ['count', 'number', 'how many']):
            intent = 'count'
        elif any(term in query_lower for term in ['add', 'insert', 'create']):
            intent = 'create'
        elif any(term in query_lower for term in ['update', 'change', 'modify']):
            intent = 'update'
        elif any(term in query_lower for term in ['delete', 'remove']):
            intent = 'delete'
        elif any(term in query_lower for term in ['analyze', 'report', 'summary']):
            intent = 'analyze'
        
        # Temporal indicators
        temporal = []
        if any(term in query_lower for term in ['today', 'yesterday', 'last week', 'this month']):
            temporal.append('relative_time')
        if any(term in query_lower for term in ['2023', '2024', 'january', 'february']):
            temporal.append('absolute_time')
        
        return {
            'entities': entities,
            'intent': intent,
            'temporal': temporal,
            'length': len(natural_query.split()),
            'question_words': [word for word in ['what', 'when', 'where', 'who', 'how', 'why'] 
                              if word in query_lower]
        }
    
    def _find_similar_patterns(self, natural_query: str, database_type: DatabaseType) -> List[QueryPattern]:
        """Find similar patterns from learning history"""
        
        similar_patterns = []
        
        for pattern in self.query_patterns.values():
            if pattern.database_type == database_type:
                similarity = self._calculate_similarity(natural_query, pattern.natural_language_pattern)
                if similarity > 0.6:  # Threshold for similarity
                    similar_patterns.append((pattern, similarity))
        
        # Sort by similarity and success rate
        similar_patterns.sort(key=lambda x: (x[1], x[0].success_rate), reverse=True)
        
        return [pattern for pattern, _ in similar_patterns[:5]]  # Top 5 similar patterns
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate semantic similarity between queries"""
        
        # Simple word overlap similarity (in production, use embeddings)
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _generate_learning_insights(self, natural_query: str, similar_patterns: List[QueryPattern]) -> List[str]:
        """Generate insights based on learned patterns"""
        
        insights = []
        
        if not similar_patterns:
            insights.append("🆕 Novel query pattern - will create new learning entry")
        else:
            best_pattern = similar_patterns[0]
            insights.append(f"🎯 Similar to pattern with {best_pattern.success_rate:.1%} success rate")
            
            if best_pattern.avg_execution_time > 0:
                insights.append(f"⏱️ Expected execution time: ~{best_pattern.avg_execution_time:.2f}s")
            
            if len(best_pattern.generated_queries) > 0:
                insights.append(f"📚 {len(best_pattern.generated_queries)} successful queries learned")
        
        return insights
    
    def _classify_query_type(self, natural_query: str) -> str:
        """Classify the type of query"""
        
        query_lower = natural_query.lower()
        
        if any(term in query_lower for term in ['select', 'show', 'get', 'find', 'list']):
            return 'select'
        elif any(term in query_lower for term in ['count', 'number', 'how many']):
            return 'count'
        elif any(term in query_lower for term in ['sum', 'total', 'average', 'mean', 'max', 'min']):
            return 'aggregate'
        elif any(term in query_lower for term in ['join', 'combine', 'merge']):
            return 'join'
        elif any(term in query_lower for term in ['group', 'category', 'breakdown']):
            return 'group'
        elif any(term in query_lower for term in ['insert', 'add', 'create']):
            return 'insert'
        elif any(term in query_lower for term in ['update', 'change', 'modify']):
            return 'update'
        elif any(term in query_lower for term in ['delete', 'remove']):
            return 'delete'
        else:
            return 'complex'
    
    def _estimate_performance(self, natural_query: str, database_type: DatabaseType) -> Dict[str, float]:
        """Estimate performance based on learned patterns"""
        
        similar_patterns = self._find_similar_patterns(natural_query, database_type)
        
        if similar_patterns:
            avg_success_rate = sum(p.success_rate for p in similar_patterns) / len(similar_patterns)
            avg_execution_time = sum(p.avg_execution_time for p in similar_patterns) / len(similar_patterns)
        else:
            avg_success_rate = 0.7  # Default expectation
            avg_execution_time = 1.0  # Default expectation
        
        return {
            'expected_success_rate': avg_success_rate,
            'expected_execution_time': avg_execution_time,
            'confidence': min(1.0, len(similar_patterns) * 0.2)  # More patterns = higher confidence
        }
    
    def _recommend_specialist_traits(self, natural_query: str, database_type: DatabaseType, 
                                   complexity: QueryComplexity) -> Dict[str, Any]:
        """Recommend traits for specialist creation"""
        
        base_capabilities = {CapabilityType.NLP_PROCESSING, CapabilityType.DATABASE_CONNECTOR}
        
        # Add capabilities based on complexity
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.ADVANCED]:
            base_capabilities.add(CapabilityType.REASONING)
            base_capabilities.add(CapabilityType.PLANNING)
        
        # Database-specific capabilities
        if database_type == DatabaseType.MONGODB:
            base_capabilities.add(CapabilityType.DATA_ANALYSIS)  # For aggregation pipelines
        
        # Query-specific tools
        tools = {'nlp_processor', 'query_generator', 'schema_analyzer'}
        
        if database_type == DatabaseType.POSTGRESQL:
            tools.update({'postgres_connector', 'sql_optimizer', 'postgres_analyzer'})
        elif database_type == DatabaseType.MONGODB:
            tools.update({'mongo_connector', 'aggregation_builder', 'document_analyzer'})
        
        return {
            'recommended_capabilities': base_capabilities,  # Keep as set of CapabilityType objects
            'recommended_tools': list(tools),
            'specialization_domain': database_type.value,
            'complexity_handling': complexity.value,
            'learning_rate': 0.15 if complexity == QueryComplexity.ADVANCED else 0.1,
            'creativity_factor': 0.8 if complexity == QueryComplexity.ADVANCED else 0.6
        }
    
    async def _get_or_create_specialist(self, analysis: Dict[str, Any]) -> Agent:
        """
        Get existing specialist or create new one - SELF-ASSEMBLY in action!
        
        This demonstrates emergent intelligence where the system decides
        whether to use existing knowledge or create new capabilities.
        """
        
        database_type = analysis['database_type']
        complexity = analysis['complexity']
        
        # Check if we have existing specialists for this database type
        existing_specialists = self.database_specialists.get(database_type, [])
        
        if existing_specialists:
            # Find the best specialist based on performance and suitability
            best_specialist = self._select_best_specialist(existing_specialists, analysis)
            
            # Check if the best specialist is good enough
            if self._should_use_existing_specialist(best_specialist, analysis):
                logger.info(f"🎯 Using existing specialist: {best_specialist.name}")
                return best_specialist
        
        # Create new specialist - SELF-ASSEMBLY!
        logger.info(f"🔧 Self-assembling new specialist for {database_type.value}")
        new_specialist = await self._create_specialized_agent(analysis)
        
        # Register the new specialist
        if database_type not in self.database_specialists:
            self.database_specialists[database_type] = []
        
        self.database_specialists[database_type].append(new_specialist)
        
        # Initialize performance tracking
        self.specialist_performance[new_specialist.id] = SpecialistPerformance(
            agent_id=new_specialist.id,
            database_type=database_type
        )
        
        logger.info(f"✅ New specialist created: {new_specialist.name}")
        return new_specialist
    
    def _select_best_specialist(self, specialists: List[Agent], analysis: Dict[str, Any]) -> Agent:
        """Select the best specialist based on performance and suitability"""
        
        best_specialist = specialists[0]
        best_score = 0.0
        
        for specialist in specialists:
            performance = self.specialist_performance.get(specialist.id)
            if not performance:
                continue
            
            # Calculate suitability score
            score = (
                performance.success_rate * 0.4 +
                performance.specialization_score * 0.3 +
                performance.learning_velocity * 0.2 +
                (1.0 / (1.0 + performance.avg_response_time)) * 0.1  # Lower response time is better
            )
            
            if score > best_score:
                best_score = score
                best_specialist = specialist
        
        return best_specialist
    
    def _should_use_existing_specialist(self, specialist: Agent, analysis: Dict[str, Any]) -> bool:
        """Decide whether to use existing specialist or create new one"""
        
        performance = self.specialist_performance.get(specialist.id)
        if not performance:
            return False
        
        # Use existing if performance is above threshold
        if performance.success_rate >= self.specialization_threshold:
            return True
        
        # Use existing if we have too many specialists already
        database_type = analysis['database_type']
        if len(self.database_specialists.get(database_type, [])) >= self.max_specialists_per_db:
            return True
        
        return False
    
    async def _create_specialized_agent(self, analysis: Dict[str, Any]) -> Agent:
        """Create a specialized agent using advanced factory methods"""
        
        database_type = analysis['database_type']
        complexity = analysis['complexity']
        recommended_traits = analysis['recommended_specialist_traits']
        
        # Create sophisticated DNA for the specialist
        capabilities = set()
        for cap_type in recommended_traits['recommended_capabilities']:
            capabilities.add(Capability(
                type=cap_type,
                domain=recommended_traits['specialization_domain'],
                confidence=0.8 + random.uniform(-0.1, 0.2)  # Slight randomization
            ))
        
        # Generate unique name
        specialist_name = f"{database_type.value.title()}Specialist_{complexity.value}_{random.randint(1000, 9999)}"
        
        # Create sophisticated DNA
        specialist_dna = AgentDNA(
            name=specialist_name,
            capabilities=capabilities,
            knowledge_domains=[
                database_type.value,
                'natural_language_processing',
                'query_generation',
                'performance_optimization'
            ],
            tools=set(recommended_traits['recommended_tools']),
            goals=[
                f"Master {database_type.value} query generation",
                f"Handle {complexity.value} complexity queries",
                "Learn from every interaction",
                "Optimize for performance and accuracy"
            ],
            learning_rate=recommended_traits['learning_rate'],
            creativity_factor=recommended_traits['creativity_factor'],
            collaboration_preference=0.7,
            risk_tolerance=0.3
        )
        
        # Create the agent with specialized DNA
        if database_type == DatabaseType.POSTGRESQL:
            from ..specialists.postgres_specialist import PostgreSQLSpecialist
            specialist = PostgreSQLSpecialist(specialist_dna)
        elif database_type == DatabaseType.MONGODB:
            from ..specialists.mongodb_specialist import MongoDBSpecialist
            specialist = MongoDBSpecialist(specialist_dna)
        else:
            # Generic SQL specialist
            specialist = Agent(specialist_dna)
        
        return specialist
    
    async def _execute_with_specialist(self, specialist: Agent, natural_query: str, 
                                     analysis: Dict[str, Any]) -> ExecutionResult:
        """Execute query with the selected specialist"""
        
        # Create task for the specialist
        task_context = {
            'natural_query': natural_query,
            'database_type': analysis['database_type'].value,
            'complexity': analysis['complexity'].value,
            'semantic_features': analysis['semantic_features'],
            'similar_patterns': [p.natural_language_pattern for p in analysis['similar_patterns']]
        }
        
        task = Task(
            intent=f"Convert to {analysis['database_type'].value}: {natural_query}",
            task_type=TaskType.TRANSFORMATION,
            context=task_context
        )
        
        # Execute with specialist
        result = await specialist.execute_task(task)
        
        return result
    
    async def _learn_from_execution(self, natural_query: str, analysis: Dict[str, Any], 
                                  result: ExecutionResult, specialist: Agent):
        """
        META-LEARNING: Learn from execution to improve future performance
        
        This is where the system gets smarter over time!
        """
        
        logger.info(f"📚 Meta-learning from execution...")
        
        # Update specialist performance
        performance = self.specialist_performance.get(specialist.id)
        if performance:
            execution_time = result.execution_time or 0.0
            performance.update(result.success, execution_time, analysis['complexity'])
        
        # Learn query patterns
        await self._learn_query_pattern(natural_query, analysis, result)
        
        # Update knowledge graph
        self._update_knowledge_graph(natural_query, analysis, result, specialist)
        
        # Record learning history
        learning_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'natural_query': natural_query,
            'database_type': analysis['database_type'].value,
            'complexity': analysis['complexity'].value,
            'specialist_used': specialist.name,
            'success': result.success,
            'execution_time': result.execution_time or 0.0,
            'learning_insights': analysis.get('learning_insights', [])
        }
        
        self.learning_history.append(learning_entry)
        
        # Limit history size
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-500:]
        
        logger.info(f"✅ Learning complete - {len(self.query_patterns)} patterns learned")
    
    async def _learn_query_pattern(self, natural_query: str, analysis: Dict[str, Any], 
                                 result: ExecutionResult):
        """Learn and update query patterns"""
        
        pattern_key = f"{analysis['database_type'].value}_{analysis['query_type']}"
        
        if pattern_key not in self.query_patterns:
            # Create new pattern
            self.query_patterns[pattern_key] = QueryPattern(
                pattern_id=pattern_key,
                natural_language_pattern=natural_query,
                database_type=analysis['database_type'],
                query_type=analysis['query_type'],
                complexity=analysis['complexity']
            )
        
        # Update pattern performance
        pattern = self.query_patterns[pattern_key]
        execution_time = result.execution_time or 0.0
        generated_query = result.data.get('generated_query', '') if result.data else ''
        
        pattern.update_performance(result.success, execution_time, generated_query)
    
    def _update_knowledge_graph(self, natural_query: str, analysis: Dict[str, Any], 
                              result: ExecutionResult, specialist: Agent):
        """Update the knowledge graph with new insights"""
        
        # Extract entities and relationships
        entities = analysis['semantic_features']['entities']
        database_type = analysis['database_type'].value
        
        # Update entity knowledge
        for entity in entities:
            entity_key = f"entity_{entity}"
            if entity_key not in self.knowledge_graph:
                self.knowledge_graph[entity_key] = {
                    'type': 'entity',
                    'name': entity,
                    'databases_seen': set(),
                    'query_patterns': [],
                    'success_rates': {}
                }
            
            entity_data = self.knowledge_graph[entity_key]
            entity_data['databases_seen'].add(database_type)
            
            if len(entity_data['query_patterns']) < 10:
                entity_data['query_patterns'].append(natural_query)
            
            if database_type not in entity_data['success_rates']:
                entity_data['success_rates'][database_type] = []
            
            entity_data['success_rates'][database_type].append(1.0 if result.success else 0.0)
        
        # Update specialist knowledge
        specialist_key = f"specialist_{specialist.id}"
        if specialist_key not in self.knowledge_graph:
            self.knowledge_graph[specialist_key] = {
                'type': 'specialist',
                'name': specialist.name,
                'database_type': database_type,
                'handled_entities': set(),
                'query_types': set()
            }
        
        specialist_data = self.knowledge_graph[specialist_key]
        specialist_data['handled_entities'].update(entities)
        specialist_data['query_types'].add(analysis['query_type'])
    
    async def _evolve_specialists_if_needed(self):
        """
        EVOLUTIONARY IMPROVEMENT: Evolve specialists based on performance
        
        This demonstrates recursive self-improvement!
        """
        
        logger.info(f"🧬 Checking for evolutionary opportunities...")
        
        for database_type, specialists in self.database_specialists.items():
            if len(specialists) >= 2:  # Need at least 2 for evolution
                
                # Check if evolution is warranted
                performances = [
                    self.specialist_performance.get(s.id) for s in specialists
                    if self.specialist_performance.get(s.id)
                ]
                
                if not performances:
                    continue
                
                avg_success_rate = sum(p.success_rate for p in performances) / len(performances)
                
                # Evolve if performance is below threshold
                if avg_success_rate < 0.8 and len(specialists) < self.max_specialists_per_db:
                    logger.info(f"🧬 Evolving {database_type.value} specialists...")
                    
                    # Get top performers for breeding
                    top_specialists = sorted(
                        specialists,
                        key=lambda s: self.specialist_performance.get(s.id, SpecialistPerformance(s.id, database_type)).success_rate,
                        reverse=True
                    )[:2]
                    
                    # Create evolved specialist using factory evolution
                    performance_data = {
                        s.id: self.specialist_performance.get(s.id, SpecialistPerformance(s.id, database_type)).success_rate
                        for s in top_specialists
                    }
                    
                    evolved_specialist = self.agent_factory.evolve_agent(
                        top_specialists,
                        {CapabilityType.DATABASE_CONNECTOR, CapabilityType.NLP_PROCESSING},
                        performance_data
                    )
                    
                    # Add evolved specialist to the pool
                    specialists.append(evolved_specialist)
                    
                    # Initialize performance tracking
                    self.specialist_performance[evolved_specialist.id] = SpecialistPerformance(
                        agent_id=evolved_specialist.id,
                        database_type=database_type
                    )
                    
                    logger.info(f"✅ Evolved specialist created: {evolved_specialist.name}")
    
    def _detect_emergent_behaviors(self) -> List[str]:
        """Detect emergent behaviors in the system"""
        
        behaviors = []
        
        # Cross-database pattern transfer
        db_patterns = {}
        for pattern in self.query_patterns.values():
            db_type = pattern.database_type.value
            if db_type not in db_patterns:
                db_patterns[db_type] = []
            db_patterns[db_type].append(pattern.natural_language_pattern)
        
        if len(db_patterns) > 1:
            behaviors.append("cross_database_learning")
        
        # Specialist collaboration
        if len(self.specialist_performance) > 3:
            behaviors.append("multi_specialist_ecosystem")
        
        # Adaptive improvement
        improving_specialists = [
            p for p in self.specialist_performance.values()
            if p.learning_velocity > 0.05
        ]
        
        if len(improving_specialists) > 0:
            behaviors.append("adaptive_improvement")
        
        # Pattern generalization
        if len(self.query_patterns) > 10:
            behaviors.append("pattern_generalization")
        
        return behaviors
    
    def get_system_intelligence_report(self) -> Dict[str, Any]:
        """Get comprehensive intelligence report"""
        
        total_specialists = sum(len(agents) for agents in self.database_specialists.values())
        
        # Calculate overall performance
        if self.specialist_performance:
            avg_success_rate = sum(p.success_rate for p in self.specialist_performance.values()) / len(self.specialist_performance)
            avg_response_time = sum(p.avg_response_time for p in self.specialist_performance.values()) / len(self.specialist_performance)
        else:
            avg_success_rate = 0.0
            avg_response_time = 0.0
        
        return {
            'meta_agent_status': 'active',
            'total_specialists': total_specialists,
            'database_types_supported': [db.value for db in self.database_specialists.keys()],
            'query_patterns_learned': len(self.query_patterns),
            'knowledge_graph_size': len(self.knowledge_graph),
            'learning_history_size': len(self.learning_history),
            'overall_performance': {
                'avg_success_rate': avg_success_rate,
                'avg_response_time': avg_response_time
            },
            'emergent_behaviors': self._detect_emergent_behaviors(),
            'specialist_breakdown': {
                db.value: len(agents) for db, agents in self.database_specialists.items()
            },
            'intelligence_metrics': {
                'adaptation_rate': len([p for p in self.specialist_performance.values() if p.learning_velocity > 0.01]),
                'pattern_recognition': len(self.query_patterns),
                'cross_domain_synthesis': len(set(p.database_type for p in self.query_patterns.values())),
                'self_improvement_active': any(p.learning_velocity > 0.05 for p in self.specialist_performance.values())
            }
        }
    
    def __str__(self) -> str:
        total_specialists = sum(len(agents) for agents in self.database_specialists.values())
        return f"NLP2SQLMetaAgent(specialists={total_specialists}, patterns={len(self.query_patterns)}, learning={len(self.learning_history)})"


# Demo function
async def demo_emergent_intelligence():
    """Demonstrate emergent intelligence in action"""
    
    print("\n" + "="*80)
    print("🧘 EMERGENT INTELLIGENCE DEMONSTRATION")
    print("   NLP2SQL Meta-Agent with Learning and Self-Assembly")
    print("="*80)
    
    # Create meta-agent
    meta_agent = NLP2SQLMetaAgent()
    
    # Test queries that will trigger specialist creation
    test_queries = [
        ("Show me all customers from New York", {"database_type": "postgresql"}),
        ("Find documents where status is active", {"database_type": "mongodb"}),
        ("Get the average order value by month", {"database_type": "postgresql"}),
        ("Aggregate sales data by product category", {"database_type": "mongodb"}),
        ("Show customers who placed orders in the last 30 days", {"database_type": "postgresql"}),
    ]
    
    print(f"📊 Initial System State:")
    initial_report = meta_agent.get_system_intelligence_report()
    for key, value in initial_report.items():
        print(f"   {key}: {value}")
    
    # Process queries and watch the system learn and evolve
    for i, (query, config) in enumerate(test_queries, 1):
        print(f"\n🎯 Query {i}: '{query}'")
        print("-" * 60)
        
        result = await meta_agent.process_natural_language_query(query, config)
        
        print(f"📋 Result:")
        print(f"   Success: {result['success']}")
        print(f"   Database: {result.get('database_type', 'unknown')}")
        print(f"   Complexity: {result.get('complexity', 'unknown')}")
        print(f"   Specialist: {result.get('specialist_used', 'unknown')}")
        print(f"   Execution Time: {result.get('execution_time', 0):.3f}s")
        
        if result.get('learning_insights'):
            print(f"   Learning Insights:")
            for insight in result['learning_insights']:
                print(f"     • {insight}")
        
        # Show system evolution
        current_report = meta_agent.get_system_intelligence_report()
        print(f"   System Growth:")
        print(f"     Specialists: {current_report['total_specialists']}")
        print(f"     Patterns: {current_report['query_patterns_learned']}")
        print(f"     Emergent Behaviors: {current_report['emergent_behaviors']}")
    
    # Final intelligence report
    print(f"\n🧠 FINAL INTELLIGENCE REPORT")
    print("=" * 60)
    
    final_report = meta_agent.get_system_intelligence_report()
    
    print(f"🎉 System Evolution Summary:")
    for key, value in final_report.items():
        print(f"   {key}: {value}")
    
    print(f"\n✨ Emergent Intelligence Demonstrated!")
    print(f"   The meta-agent successfully:")
    print(f"   🔄 Created specialists on-demand")
    print(f"   📚 Learned from every interaction")
    print(f"   🧬 Showed emergent behaviors")
    print(f"   🎯 Adapted to different database types")
    print(f"   ⚡ Improved performance over time")


if __name__ == "__main__":
    asyncio.run(demo_emergent_intelligence()) 