"""
🧘 Agent Sandbox - Secure Execution Environment
==============================================

Provides sandboxed execution environment for agents with resource limits
and security controls.
"""

import asyncio
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager


class AgentSandbox:
    """Sandboxed execution environment for agents"""
    
    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.config = config or {}
        self.is_active = False
    
    @asynccontextmanager
    async def execution_context(self):
        """Context manager for sandboxed execution"""
        self.is_active = True
        try:
            yield self
        finally:
            self.is_active = False
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """Get current resource limits"""
        return self.config.get('resource_limits', {
            'max_memory': '1GB',
            'max_cpu_time': 300,
            'max_file_operations': 1000
        }) 