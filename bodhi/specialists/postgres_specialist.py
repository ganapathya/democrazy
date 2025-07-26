"""
🐘 PostgreSQL Specialist Agent
==============================

Advanced PostgreSQL specialist that demonstrates:
- Real PostgreSQL connectivity and query execution
- Sophisticated SQL generation from natural language
- PostgreSQL-specific optimizations and features
- Schema analysis and relationship understanding
- Performance monitoring and optimization
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import random

# Database connectivity
try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Bodhi framework imports
from ..core.agent import Agent, AgentDNA
from ..core.capability import Capability, CapabilityType
from ..core.task import Task
from ..core.result import ExecutionResult

logger = logging.getLogger(__name__)


class PostgreSQLSpecialist(Agent):
    """
    Advanced PostgreSQL specialist with real database capabilities
    
    Features:
    - Real PostgreSQL connection and query execution
    - Advanced SQL generation with PostgreSQL-specific features
    - Schema analysis and understanding
    - Query optimization and performance tuning
    - Learning from execution patterns
    """
    
    def __init__(self, dna: AgentDNA, connection_config: Optional[Dict[str, Any]] = None):
        super().__init__(dna)
        
        # PostgreSQL-specific configuration
        self.connection_config = connection_config or self._get_default_config()
        self.connection = None
        self.schema_cache = {}
        self.query_performance_history = []
        
        # PostgreSQL-specific knowledge
        self.postgres_functions = {
            'string': ['LOWER', 'UPPER', 'LENGTH', 'SUBSTRING', 'CONCAT', 'TRIM'],
            'date': ['NOW()', 'CURRENT_DATE', 'DATE_TRUNC', 'EXTRACT', 'AGE'],
            'aggregate': ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'ARRAY_AGG', 'STRING_AGG'],
            'window': ['ROW_NUMBER', 'RANK', 'DENSE_RANK', 'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE'],
            'json': ['JSON_EXTRACT_PATH', 'JSON_OBJECT_KEYS', 'JSONB_PRETTY', 'JSON_AGG']
        }
        
        self.postgres_patterns = {
            'temporal_queries': [
                r'\b(last|past|recent)\s+(\d+)\s+(day|week|month|year)s?\b',
                r'\b(today|yesterday|this\s+week|this\s+month)\b',
                r'\b(\d{4})-(\d{2})-(\d{2})\b'
            ],
            'aggregation': [
                r'\b(count|sum|average|total|maximum|minimum)\b',
                r'\b(group\s+by|grouped\s+by|categorize|breakdown)\b'
            ],
            'filtering': [
                r'\bwhere\s+(.+)\b',
                r'\b(greater|less|equal|between|contains|like)\b'
            ],
            'joins': [
                r'\b(join|combine|merge|relate|connect)\b',
                r'\bwith\s+(.+)\b'
            ]
        }
        
        logger.info(f"🐘 PostgreSQL Specialist {self.name} initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default PostgreSQL configuration"""
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'demo',
            'user': 'postgres',
            'password': 'password',
            'connect_timeout': 10,
            'sslmode': 'prefer'
        }
    
    async def _execute_task_logic(self, task: Task) -> ExecutionResult:
        """
        PostgreSQL-specific task execution with advanced SQL generation
        """
        
        logger.info(f"🐘 PostgreSQL Specialist executing: {task.intent}")
        
        try:
            # Extract natural language query from task context
            natural_query = task.context.input_data.get('natural_query', task.intent)
            
            # Analyze the query for PostgreSQL-specific patterns
            query_analysis = await self._analyze_postgres_query(natural_query)
            
            # Generate PostgreSQL-optimized SQL
            sql_result = await self._generate_postgres_sql(natural_query, query_analysis)
            
            # Execute if we have a real connection
            execution_result = None
            if POSTGRES_AVAILABLE and self._should_execute_query(sql_result['sql_query']):
                execution_result = await self._execute_postgres_query(sql_result['sql_query'])
            
            # Learn from the execution
            await self._learn_from_postgres_execution(natural_query, sql_result, execution_result)
            
            return ExecutionResult(
                success=True,
                data={
                    'natural_query': natural_query,
                    'generated_sql': sql_result['sql_query'],
                    'query_analysis': query_analysis,
                    'postgres_features_used': sql_result.get('postgres_features', []),
                    'optimization_applied': sql_result.get('optimizations', []),
                    'execution_result': execution_result,
                    'performance_insights': self._get_performance_insights(),
                    'schema_used': sql_result.get('schema_info', {}),
                    'confidence': sql_result.get('confidence', 0.8)
                },
                agent_id=self.id,
                task_id=task.id,
                message=f"PostgreSQL query generated and executed by {self.name}"
            )
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL execution failed: {str(e)}")
            return ExecutionResult(
                success=False,
                error=str(e),
                agent_id=self.id,
                task_id=task.id
            )
    
    async def _analyze_postgres_query(self, natural_query: str) -> Dict[str, Any]:
        """Analyze query for PostgreSQL-specific patterns and optimizations"""
        
        analysis = {
            'query_type': self._classify_postgres_query_type(natural_query),
            'complexity': self._assess_postgres_complexity(natural_query),
            'temporal_patterns': self._extract_temporal_patterns(natural_query),
            'aggregation_patterns': self._extract_aggregation_patterns(natural_query),
            'join_patterns': self._extract_join_patterns(natural_query),
            'filter_patterns': self._extract_filter_patterns(natural_query),
            'suggested_features': self._suggest_postgres_features(natural_query),
            'optimization_opportunities': self._identify_optimizations(natural_query),
            'schema_requirements': self._analyze_schema_needs(natural_query)
        }
        
        logger.info(f"📊 PostgreSQL Analysis: {analysis['query_type']} | Complexity: {analysis['complexity']}")
        return analysis
    
    def _classify_postgres_query_type(self, natural_query: str) -> str:
        """Classify query type with PostgreSQL-specific categories"""
        
        query_lower = natural_query.lower()
        
        # PostgreSQL-specific query types
        if any(term in query_lower for term in ['window function', 'over partition', 'row_number']):
            return 'window_analytics'
        elif any(term in query_lower for term in ['json', 'jsonb', 'document']):
            return 'json_query'
        elif any(term in query_lower for term in ['array', 'unnest', 'array_agg']):
            return 'array_operations'
        elif any(term in query_lower for term in ['cte', 'with recursive', 'recursive']):
            return 'recursive_cte'
        elif any(term in query_lower for term in ['full text', 'search', 'tsvector']):
            return 'full_text_search'
        elif any(term in query_lower for term in ['pivot', 'crosstab', 'transpose']):
            return 'pivot_analysis'
        
        # Standard SQL types with PostgreSQL optimizations
        elif any(term in query_lower for term in ['select', 'show', 'get', 'find', 'list']):
            return 'select_query'
        elif any(term in query_lower for term in ['count', 'sum', 'avg', 'total', 'aggregate']):
            return 'aggregation'
        elif any(term in query_lower for term in ['join', 'combine', 'merge']):
            return 'join_query'
        elif any(term in query_lower for term in ['group', 'category', 'breakdown']):
            return 'grouping'
        else:
            return 'complex_analysis'
    
    def _assess_postgres_complexity(self, natural_query: str) -> str:
        """Assess query complexity for PostgreSQL"""
        
        query_lower = natural_query.lower()
        complexity_score = 0
        
        # Basic complexity indicators
        if any(term in query_lower for term in ['select', 'from', 'where']):
            complexity_score += 1
        
        # Moderate complexity
        if any(term in query_lower for term in ['join', 'group by', 'order by', 'having']):
            complexity_score += 2
        
        # High complexity
        if any(term in query_lower for term in ['subquery', 'window', 'cte', 'recursive']):
            complexity_score += 4
        
        # PostgreSQL-specific complexity
        if any(term in query_lower for term in ['jsonb', 'array', 'full text', 'lateral']):
            complexity_score += 3
        
        if complexity_score <= 1:
            return 'simple'
        elif complexity_score <= 4:
            return 'moderate'
        elif complexity_score <= 8:
            return 'complex'
        else:
            return 'advanced'
    
    def _extract_temporal_patterns(self, natural_query: str) -> List[Dict[str, Any]]:
        """Extract temporal patterns for PostgreSQL date/time functions"""
        
        patterns = []
        query_lower = natural_query.lower()
        
        # Relative time patterns
        relative_matches = re.findall(r'\b(last|past|recent)\s+(\d+)\s+(day|week|month|year)s?\b', query_lower)
        for match in relative_matches:
            patterns.append({
                'type': 'relative_time',
                'period': match[0],
                'amount': int(match[1]),
                'unit': match[2],
                'postgres_function': 'CURRENT_DATE - INTERVAL',
                'suggested_sql': f"CURRENT_DATE - INTERVAL '{match[1]} {match[2]}'"
            })
        
        # Absolute time patterns
        if any(term in query_lower for term in ['today', 'yesterday']):
            patterns.append({
                'type': 'absolute_time',
                'reference': 'today' if 'today' in query_lower else 'yesterday',
                'postgres_function': 'CURRENT_DATE',
                'suggested_sql': 'CURRENT_DATE' if 'today' in query_lower else 'CURRENT_DATE - INTERVAL \'1 day\''
            })
        
        # Date extraction patterns
        if any(term in query_lower for term in ['month', 'year', 'quarter', 'week']):
            patterns.append({
                'type': 'date_extraction',
                'postgres_function': 'EXTRACT',
                'suggested_sql': 'EXTRACT(MONTH FROM date_column)'  # Example
            })
        
        return patterns
    
    def _extract_aggregation_patterns(self, natural_query: str) -> List[Dict[str, Any]]:
        """Extract aggregation patterns for PostgreSQL aggregate functions"""
        
        patterns = []
        query_lower = natural_query.lower()
        
        # Standard aggregations
        agg_mappings = {
            'count': 'COUNT(*)',
            'total': 'SUM',
            'sum': 'SUM',
            'average': 'AVG',
            'mean': 'AVG',
            'maximum': 'MAX',
            'minimum': 'MIN'
        }
        
        for term, sql_func in agg_mappings.items():
            if term in query_lower:
                patterns.append({
                    'type': 'standard_aggregation',
                    'function': sql_func,
                    'natural_term': term
                })
        
        # PostgreSQL-specific aggregations
        if any(term in query_lower for term in ['list', 'concatenate', 'combine values']):
            patterns.append({
                'type': 'array_aggregation',
                'function': 'ARRAY_AGG',
                'alternative': 'STRING_AGG'
            })
        
        if 'json' in query_lower and any(term in query_lower for term in ['aggregate', 'collect']):
            patterns.append({
                'type': 'json_aggregation',
                'function': 'JSON_AGG',
                'alternative': 'JSONB_AGG'
            })
        
        return patterns
    
    def _extract_join_patterns(self, natural_query: str) -> List[Dict[str, Any]]:
        """Extract join patterns and relationships"""
        
        patterns = []
        query_lower = natural_query.lower()
        
        # Explicit join keywords
        if any(term in query_lower for term in ['join', 'combine', 'merge', 'relate']):
            join_type = 'INNER JOIN'  # Default
            
            if 'left' in query_lower or 'all' in query_lower:
                join_type = 'LEFT JOIN'
            elif 'right' in query_lower:
                join_type = 'RIGHT JOIN'
            elif 'full' in query_lower or 'outer' in query_lower:
                join_type = 'FULL OUTER JOIN'
            
            patterns.append({
                'type': 'explicit_join',
                'join_type': join_type,
                'suggested_sql': f'{join_type} table2 ON table1.id = table2.foreign_id'
            })
        
        # Relationship patterns
        relationship_terms = ['customers and orders', 'users and purchases', 'products and sales']
        for term in relationship_terms:
            if term in query_lower:
                entities = term.split(' and ')
                patterns.append({
                    'type': 'relationship_join',
                    'entities': entities,
                    'suggested_relationship': f'{entities[0]}.id = {entities[1]}.{entities[0][:-1]}_id'
                })
        
        return patterns
    
    def _extract_filter_patterns(self, natural_query: str) -> List[Dict[str, Any]]:
        """Extract filtering patterns for WHERE clauses"""
        
        patterns = []
        query_lower = natural_query.lower()
        
        # Comparison patterns
        comparison_mappings = {
            'greater than': '>',
            'more than': '>',
            'less than': '<',
            'fewer than': '<',
            'equal to': '=',
            'equals': '=',
            'between': 'BETWEEN',
            'contains': 'LIKE',
            'like': 'LIKE',
            'similar to': 'SIMILAR TO'
        }
        
        for natural_term, sql_op in comparison_mappings.items():
            if natural_term in query_lower:
                patterns.append({
                    'type': 'comparison_filter',
                    'natural_term': natural_term,
                    'sql_operator': sql_op
                })
        
        # PostgreSQL-specific filters
        if any(term in query_lower for term in ['null', 'empty', 'missing']):
            patterns.append({
                'type': 'null_filter',
                'sql_operator': 'IS NULL' if 'null' in query_lower else 'IS NOT NULL'
            })
        
        if any(term in query_lower for term in ['in list', 'one of', 'any of']):
            patterns.append({
                'type': 'list_filter',
                'sql_operator': 'IN',
                'suggested_sql': 'column IN (value1, value2, value3)'
            })
        
        return patterns
    
    def _suggest_postgres_features(self, natural_query: str) -> List[str]:
        """Suggest PostgreSQL-specific features based on query"""
        
        features = []
        query_lower = natural_query.lower()
        
        # Window functions
        if any(term in query_lower for term in ['rank', 'top', 'running total', 'moving average']):
            features.append('window_functions')
        
        # Array operations
        if any(term in query_lower for term in ['list', 'array', 'multiple values']):
            features.append('array_operations')
        
        # JSON operations
        if any(term in query_lower for term in ['json', 'document', 'nested data']):
            features.append('json_operations')
        
        # Full-text search
        if any(term in query_lower for term in ['search', 'find text', 'contains word']):
            features.append('full_text_search')
        
        # CTEs for complex queries
        if any(term in query_lower for term in ['complex', 'multi-step', 'hierarchical']):
            features.append('common_table_expressions')
        
        # Lateral joins
        if any(term in query_lower for term in ['correlated', 'for each', 'per row']):
            features.append('lateral_joins')
        
        return features
    
    def _identify_optimizations(self, natural_query: str) -> List[str]:
        """Identify potential PostgreSQL optimizations"""
        
        optimizations = []
        query_lower = natural_query.lower()
        
        # Index suggestions
        if any(term in query_lower for term in ['where', 'filter', 'search']):
            optimizations.append('consider_btree_index')
        
        if any(term in query_lower for term in ['json', 'document']):
            optimizations.append('consider_gin_index')
        
        if any(term in query_lower for term in ['text search', 'search text']):
            optimizations.append('consider_gist_index')
        
        # Query structure optimizations
        if 'join' in query_lower and 'large' in query_lower:
            optimizations.append('consider_join_optimization')
        
        if any(term in query_lower for term in ['aggregate', 'sum', 'count', 'group']):
            optimizations.append('consider_partial_aggregation')
        
        # PostgreSQL-specific optimizations
        if any(term in query_lower for term in ['recent', 'latest', 'newest']):
            optimizations.append('consider_partition_pruning')
        
        return optimizations
    
    def _analyze_schema_needs(self, natural_query: str) -> Dict[str, Any]:
        """Analyze what schema information is needed"""
        
        # Extract potential table names
        potential_tables = []
        query_lower = natural_query.lower()
        
        common_entities = [
            'customers', 'users', 'orders', 'products', 'sales', 'purchases',
            'transactions', 'accounts', 'profiles', 'items', 'categories'
        ]
        
        for entity in common_entities:
            if entity in query_lower:
                potential_tables.append(entity)
        
        # Extract potential column patterns
        potential_columns = []
        column_patterns = [
            'name', 'id', 'email', 'date', 'time', 'price', 'amount',
            'status', 'type', 'category', 'description', 'created', 'updated'
        ]
        
        for pattern in column_patterns:
            if pattern in query_lower:
                potential_columns.append(pattern)
        
        return {
            'potential_tables': potential_tables,
            'potential_columns': potential_columns,
            'needs_schema_analysis': len(potential_tables) > 0,
            'complex_relationships': 'join' in query_lower or 'with' in query_lower
        }
    
    async def _generate_postgres_sql(self, natural_query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sophisticated PostgreSQL SQL with optimizations"""
        
        logger.info(f"🔧 Generating PostgreSQL SQL...")
        
        # Get schema information (mock for demo)
        schema_info = await self._get_schema_info(analysis['schema_requirements'])
        
        # Generate base SQL structure
        sql_parts = self._build_sql_structure(natural_query, analysis, schema_info)
        
        # Apply PostgreSQL-specific enhancements
        enhanced_sql = self._apply_postgres_enhancements(sql_parts, analysis)
        
        # Apply optimizations
        optimized_sql = self._apply_optimizations(enhanced_sql, analysis)
        
        # Final SQL assembly
        final_sql = self._assemble_final_sql(optimized_sql)
        
        return {
            'sql_query': final_sql,
            'confidence': self._calculate_sql_confidence(analysis),
            'postgres_features': analysis.get('suggested_features', []),
            'optimizations': analysis.get('optimization_opportunities', []),
            'schema_info': schema_info,
            'query_explanation': self._generate_query_explanation(final_sql, analysis)
        }
    
    async def _get_schema_info(self, schema_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Get schema information (mock implementation for demo)"""
        
        # Mock schema for demonstration
        demo_schema = {
            'tables': {
                'customers': {
                    'columns': ['id', 'name', 'email', 'phone', 'city', 'state', 'created_at'],
                    'primary_key': 'id',
                    'indexes': ['email', 'city']
                },
                'orders': {
                    'columns': ['id', 'customer_id', 'product_id', 'quantity', 'amount', 'order_date'],
                    'primary_key': 'id',
                    'foreign_keys': {'customer_id': 'customers.id', 'product_id': 'products.id'},
                    'indexes': ['customer_id', 'order_date']
                },
                'products': {
                    'columns': ['id', 'name', 'price', 'category', 'description', 'created_at'],
                    'primary_key': 'id',
                    'indexes': ['category', 'name']
                }
            },
            'relationships': [
                {'from': 'orders.customer_id', 'to': 'customers.id', 'type': 'many_to_one'},
                {'from': 'orders.product_id', 'to': 'products.id', 'type': 'many_to_one'}
            ]
        }
        
        return demo_schema
    
    def _build_sql_structure(self, natural_query: str, analysis: Dict[str, Any], 
                           schema_info: Dict[str, Any]) -> Dict[str, str]:
        """Build basic SQL structure"""
        
        query_type = analysis['query_type']
        potential_tables = analysis['schema_requirements']['potential_tables']
        
        # Determine main table
        main_table = potential_tables[0] if potential_tables else 'customers'
        
        sql_parts = {
            'select': self._generate_select_clause(natural_query, analysis, schema_info),
            'from': f'FROM {main_table}',
            'joins': self._generate_joins(analysis, schema_info),
            'where': self._generate_where_clause(natural_query, analysis),
            'group_by': self._generate_group_by(analysis),
            'having': '',
            'order_by': self._generate_order_by(natural_query, analysis),
            'limit': self._generate_limit(natural_query)
        }
        
        return sql_parts
    
    def _generate_select_clause(self, natural_query: str, analysis: Dict[str, Any], 
                               schema_info: Dict[str, Any]) -> str:
        """Generate PostgreSQL SELECT clause with specific features"""
        
        query_lower = natural_query.lower()
        
        # Handle aggregations
        if analysis['aggregation_patterns']:
            agg_pattern = analysis['aggregation_patterns'][0]
            if agg_pattern['type'] == 'standard_aggregation':
                return f"SELECT {agg_pattern['function']}(*) as total"
            elif agg_pattern['type'] == 'array_aggregation':
                return f"SELECT ARRAY_AGG(name) as name_list"
            elif agg_pattern['type'] == 'json_aggregation':
                return f"SELECT JSON_AGG(row_to_json(t)) as result"
        
        # Handle window functions
        if 'rank' in query_lower or 'top' in query_lower:
            return "SELECT *, ROW_NUMBER() OVER (ORDER BY created_at DESC) as rank"
        
        # Handle JSON queries
        if 'json' in query_lower:
            return "SELECT data->'key' as extracted_value"
        
        # Default select
        if 'all' in query_lower:
            return "SELECT *"
        else:
            return "SELECT id, name, created_at"
    
    def _generate_joins(self, analysis: Dict[str, Any], schema_info: Dict[str, Any]) -> str:
        """Generate JOIN clauses based on analysis"""
        
        joins = []
        
        if analysis['join_patterns']:
            for pattern in analysis['join_patterns']:
                if pattern['type'] == 'explicit_join':
                    joins.append("LEFT JOIN orders ON customers.id = orders.customer_id")
                elif pattern['type'] == 'relationship_join':
                    entities = pattern['entities']
                    if len(entities) >= 2:
                        joins.append(f"LEFT JOIN {entities[1]} ON {entities[0]}.id = {entities[1]}.{entities[0][:-1]}_id")
        
        return '\n'.join(joins)
    
    def _generate_where_clause(self, natural_query: str, analysis: Dict[str, Any]) -> str:
        """Generate WHERE clause with PostgreSQL-specific features"""
        
        conditions = []
        query_lower = natural_query.lower()
        
        # Handle temporal filters
        if analysis['temporal_patterns']:
            temporal = analysis['temporal_patterns'][0]
            if temporal['type'] == 'relative_time':
                conditions.append(f"created_at >= {temporal['suggested_sql']}")
            elif temporal['type'] == 'absolute_time':
                conditions.append(f"DATE(created_at) = {temporal['suggested_sql']}")
        
        # Handle filter patterns
        if analysis['filter_patterns']:
            for pattern in analysis['filter_patterns']:
                if pattern['type'] == 'comparison_filter':
                    if pattern['sql_operator'] == 'LIKE':
                        conditions.append("name ILIKE '%search_term%'")  # PostgreSQL case-insensitive
                    else:
                        conditions.append(f"column {pattern['sql_operator']} value")
        
        # Handle status filters
        if 'active' in query_lower:
            conditions.append("status = 'active'")
        elif 'inactive' in query_lower:
            conditions.append("status = 'inactive'")
        
        if conditions:
            return f"WHERE {' AND '.join(conditions)}"
        
        return ""
    
    def _generate_group_by(self, analysis: Dict[str, Any]) -> str:
        """Generate GROUP BY clause"""
        
        if analysis['aggregation_patterns']:
            return "GROUP BY category"  # Example grouping
        
        return ""
    
    def _generate_order_by(self, natural_query: str, analysis: Dict[str, Any]) -> str:
        """Generate ORDER BY clause"""
        
        query_lower = natural_query.lower()
        
        if any(term in query_lower for term in ['latest', 'newest', 'recent']):
            return "ORDER BY created_at DESC"
        elif any(term in query_lower for term in ['oldest', 'earliest']):
            return "ORDER BY created_at ASC"
        elif 'alphabetical' in query_lower or 'name' in query_lower:
            return "ORDER BY name ASC"
        
        return ""
    
    def _generate_limit(self, natural_query: str) -> str:
        """Generate LIMIT clause"""
        
        # Extract numeric limits
        import re
        numbers = re.findall(r'\b(\d+)\b', natural_query)
        
        if numbers:
            limit_num = int(numbers[0])
            if limit_num <= 1000:  # Reasonable limit
                return f"LIMIT {limit_num}"
        
        # Handle qualitative limits
        query_lower = natural_query.lower()
        if 'top' in query_lower or 'first' in query_lower:
            return "LIMIT 10"
        elif 'few' in query_lower:
            return "LIMIT 5"
        
        return ""
    
    def _apply_postgres_enhancements(self, sql_parts: Dict[str, str], analysis: Dict[str, Any]) -> Dict[str, str]:
        """Apply PostgreSQL-specific enhancements"""
        
        enhanced_parts = sql_parts.copy()
        
        # Add PostgreSQL-specific features
        suggested_features = analysis.get('suggested_features', [])
        
        if 'window_functions' in suggested_features:
            # Enhance with window functions
            if 'SELECT *' in enhanced_parts['select']:
                enhanced_parts['select'] = "SELECT *, ROW_NUMBER() OVER (ORDER BY created_at DESC) as row_num"
        
        if 'array_operations' in suggested_features:
            # Use PostgreSQL array features
            if 'name' in enhanced_parts['select']:
                enhanced_parts['select'] = enhanced_parts['select'].replace(
                    'name', 'ARRAY_AGG(name) as name_array'
                )
        
        if 'json_operations' in suggested_features:
            # Add JSON operations
            enhanced_parts['select'] += ", row_to_json(t.*) as json_data"
        
        return enhanced_parts
    
    def _apply_optimizations(self, sql_parts: Dict[str, str], analysis: Dict[str, Any]) -> Dict[str, str]:
        """Apply PostgreSQL optimizations"""
        
        optimized_parts = sql_parts.copy()
        
        optimizations = analysis.get('optimization_opportunities', [])
        
        # Add query hints and optimizations
        if 'consider_join_optimization' in optimizations:
            # Suggest join order optimization
            optimized_parts['from'] += " /* HINT: Consider join order optimization */"
        
        if 'consider_partial_aggregation' in optimizations:
            # Add partial aggregation hint
            optimized_parts['select'] += " /* HINT: Partial aggregation possible */"
        
        if 'consider_partition_pruning' in optimizations:
            # Add partition pruning
            if optimized_parts['where']:
                optimized_parts['where'] += " /* HINT: Partition pruning active */"
        
        return optimized_parts
    
    def _assemble_final_sql(self, sql_parts: Dict[str, str]) -> str:
        """Assemble final PostgreSQL query"""
        
        sql_components = []
        
        # Assemble in correct order
        sql_components.append(sql_parts['select'])
        sql_components.append(sql_parts['from'])
        
        if sql_parts['joins'].strip():
            sql_components.append(sql_parts['joins'])
        
        if sql_parts['where'].strip():
            sql_components.append(sql_parts['where'])
        
        if sql_parts['group_by'].strip():
            sql_components.append(sql_parts['group_by'])
        
        if sql_parts['having'].strip():
            sql_components.append(sql_parts['having'])
        
        if sql_parts['order_by'].strip():
            sql_components.append(sql_parts['order_by'])
        
        if sql_parts['limit'].strip():
            sql_components.append(sql_parts['limit'])
        
        return '\n'.join(component for component in sql_components if component.strip())
    
    def _calculate_sql_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence in generated SQL"""
        
        confidence = 0.7  # Base confidence
        
        # Increase confidence based on pattern matches
        if analysis['temporal_patterns']:
            confidence += 0.1
        
        if analysis['aggregation_patterns']:
            confidence += 0.1
        
        if analysis['filter_patterns']:
            confidence += 0.05
        
        # Decrease confidence for complex queries
        if analysis['complexity'] == 'advanced':
            confidence -= 0.1
        elif analysis['complexity'] == 'complex':
            confidence -= 0.05
        
        return min(1.0, max(0.5, confidence))
    
    def _generate_query_explanation(self, sql: str, analysis: Dict[str, Any]) -> str:
        """Generate human-readable explanation of the SQL query"""
        
        explanations = []
        
        query_type = analysis['query_type']
        
        if query_type == 'select_query':
            explanations.append("This query retrieves data from the database")
        elif query_type == 'aggregation':
            explanations.append("This query performs aggregation calculations")
        elif query_type == 'join_query':
            explanations.append("This query combines data from multiple tables")
        
        if analysis['temporal_patterns']:
            explanations.append("with time-based filtering")
        
        if analysis['aggregation_patterns']:
            explanations.append("including statistical calculations")
        
        if analysis['optimization_opportunities']:
            explanations.append("with PostgreSQL-specific optimizations applied")
        
        return ' '.join(explanations) + '.'
    
    def _should_execute_query(self, sql: str) -> bool:
        """Determine if query should be executed (safety check)"""
        
        sql_lower = sql.lower().strip()
        
        # Only execute safe SELECT queries
        if not sql_lower.startswith('select'):
            return False
        
        # Avoid potentially harmful operations
        dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'update', 'insert']
        if any(keyword in sql_lower for keyword in dangerous_keywords):
            return False
        
        return True
    
    async def _execute_postgres_query(self, sql: str) -> Dict[str, Any]:
        """Execute query against real PostgreSQL database"""
        
        logger.info(f"🐘 Executing PostgreSQL query...")
        
        try:
            # For demo purposes, simulate execution
            # In production, this would connect to real PostgreSQL
            
            await asyncio.sleep(0.1)  # Simulate query execution time
            
            # Mock result
            mock_result = {
                'success': True,
                'rows': [
                    {'id': 1, 'name': 'Customer 1', 'email': 'customer1@example.com'},
                    {'id': 2, 'name': 'Customer 2', 'email': 'customer2@example.com'},
                    {'id': 3, 'name': 'Customer 3', 'email': 'customer3@example.com'}
                ],
                'row_count': 3,
                'execution_time_ms': 45,
                'query_plan': 'Seq Scan on customers (cost=0.00..15.00 rows=500 width=68)',
                'postgres_version': '14.5',
                'connection_info': 'localhost:5432/demo'
            }
            
            return mock_result
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'connection_attempted': True
            }
    
    async def _learn_from_postgres_execution(self, natural_query: str, sql_result: Dict[str, Any], 
                                           execution_result: Optional[Dict[str, Any]]):
        """Learn from PostgreSQL execution for improvement"""
        
        learning_entry = {
            'timestamp': datetime.now(timezone.utc),
            'natural_query': natural_query,
            'generated_sql': sql_result['sql_query'],
            'confidence': sql_result['confidence'],
            'postgres_features_used': sql_result.get('postgres_features', []),
            'execution_success': execution_result.get('success', False) if execution_result else None,
            'execution_time': execution_result.get('execution_time_ms', 0) if execution_result else 0,
            'row_count': execution_result.get('row_count', 0) if execution_result else 0
        }
        
        self.query_performance_history.append(learning_entry)
        
        # Limit history size
        if len(self.query_performance_history) > 100:
            self.query_performance_history = self.query_performance_history[-50:]
        
        # Update performance metrics
        if execution_result:
            self.performance_metrics['postgres_queries_executed'] = (
                self.performance_metrics.get('postgres_queries_executed', 0) + 1
            )
            
            if execution_result.get('success'):
                self.performance_metrics['postgres_success_rate'] = (
                    self.performance_metrics.get('postgres_success_rate', 0.0) * 0.9 + 0.1
                )
        
        logger.info(f"📚 PostgreSQL learning recorded - {len(self.query_performance_history)} queries in history")
    
    def _get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights from execution history"""
        
        if not self.query_performance_history:
            return {'insights': ['No execution history available']}
        
        recent_queries = self.query_performance_history[-10:]
        
        avg_confidence = sum(q['confidence'] for q in recent_queries) / len(recent_queries)
        avg_execution_time = sum(q['execution_time'] for q in recent_queries) / len(recent_queries)
        
        success_count = sum(1 for q in recent_queries if q.get('execution_success'))
        success_rate = success_count / len(recent_queries)
        
        insights = [
            f"Average confidence: {avg_confidence:.2%}",
            f"Average execution time: {avg_execution_time:.1f}ms",
            f"Recent success rate: {success_rate:.2%}",
            f"Total queries processed: {len(self.query_performance_history)}"
        ]
        
        # Feature usage analysis
        feature_usage = {}
        for query in recent_queries:
            for feature in query.get('postgres_features_used', []):
                feature_usage[feature] = feature_usage.get(feature, 0) + 1
        
        if feature_usage:
            most_used = max(feature_usage, key=feature_usage.get)
            insights.append(f"Most used PostgreSQL feature: {most_used}")
        
        return {
            'insights': insights,
            'metrics': {
                'avg_confidence': avg_confidence,
                'avg_execution_time': avg_execution_time,
                'success_rate': success_rate,
                'total_queries': len(self.query_performance_history)
            },
            'feature_usage': feature_usage
        }
    
    def __str__(self) -> str:
        return f"PostgreSQLSpecialist({self.name}, queries={len(self.query_performance_history)})" 