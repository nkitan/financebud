"""
Production-Ready Financial Agent
===================================

Next-generation financial analysis agent with advanced features:
- Real-time data processing with streaming capabilities
- Advanced caching and performance optimization
- Comprehensive error handling and recovery
- Production monitoring and observability
- Security and validation
- Multi-provider LLM support with failover
- Advanced tool orchestration and batching
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback
import weakref
from enum import Enum

# Imports
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cachetools import TTLCache, LRUCache
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY

# Internal imports
from backend.config.settings import get_settings
from backend.database.db import get_db_manager
from backend.mcp.client import get_mcp_manager
from backend.agents.llm_providers import (
    LLMConfig, LLMProvider, ProviderType,
    create_provider, get_default_config
)
from backend.logging_config import get_logger_with_context
# Backwards-compatible exports of common tools
from backend.agents.financial_tools import (
    get_account_summary_tool,
    get_recent_transactions_tool,
    search_transactions_tool,
    find_recurring_payments_tool,
    get_transactions_by_date_range_tool,
    get_monthly_summary_tool,
    get_spending_by_category_tool,
    analyze_spending_trends_tool,
    execute_custom_query_tool,
    get_database_schema_tool
)

# Configuration
settings = get_settings()
logger = get_logger_with_context(__name__)

def _get_or_create_counter(name: str, documentation: str, labelnames: Optional[List[str]] = None):
    """Get existing counter from registry or create a new one safely."""
    # prometheus_client doesn't expose a direct lookup by name in a friendly way,
    # but attempting to create a duplicate will raise; handle that.
    try:
        if labelnames:
            return Counter(name, documentation, labelnames)
        return Counter(name, documentation)
    except ValueError:
        # Collector with this name already registered; fetch from registry
        for collector in REGISTRY.collectors:
            # collector may not have 'name' attribute; guard access
            if getattr(collector, 'name', None) == name:
                return collector
        # Fallback: raise original
        raise


def _get_or_create_histogram(name: str, documentation: str):
    try:
        return Histogram(name, documentation)
    except ValueError:
        for collector in REGISTRY.collectors:
            if getattr(collector, 'name', None) == name:
                return collector
        raise


def _get_or_create_gauge(name: str, documentation: str):
    try:
        return Gauge(name, documentation)
    except ValueError:
        for collector in REGISTRY.collectors:
            if getattr(collector, 'name', None) == name:
                return collector
        raise


# Prometheus metrics (created safely to avoid duplicate registration in tests)
tool_calls_total = _get_or_create_counter('financebud_tool_calls_total', 'Total tool calls', ['tool_name', 'status'])
query_duration = _get_or_create_histogram('financebud_query_duration_seconds', 'Query execution time')
active_sessions = _get_or_create_gauge('financebud_active_sessions', 'Number of active sessions')
llm_requests_total = _get_or_create_counter('financebud_llm_requests_total', 'Total LLM requests', ['provider', 'status'])


class SessionState(str, Enum):
    """Session states for tracking."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    ERROR = "error"
    TERMINATED = "terminated"


class ToolCallPriority(str, Enum):
    """Tool call priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolCallRequest:
    """Tool call request with metadata."""
    tool_name: str
    arguments: Dict[str, Any]
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: ToolCallPriority = ToolCallPriority.NORMAL
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResult:
    """Tool call result with comprehensive metadata."""
    request_id: str
    tool_name: str
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0
    cached: bool = False
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationMessage(BaseModel):
    """Conversation message with validation."""
    role: str = Field(..., pattern=r'^(system|user|assistant|tool)$')
    # Allow empty content for assistant/tool messages coming from LLM providers
    content: str = Field(..., min_length=0)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    function_name: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to LLM provider format with validation."""
        message: Dict[str, Any] = {
            "role": self.role,
            "content": self.content
        }
        
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        
        if self.role == "tool":
            # Tool message validation
            if not self.tool_call_id:
                self.tool_call_id = f"fallback_{uuid.uuid4().hex[:8]}"
                logger.warning(f"Generated fallback tool_call_id: {self.tool_call_id}")
            
            if not self.function_name:
                self.function_name = f"fallback_function_{uuid.uuid4().hex[:8]}"
                logger.warning(f"Generated fallback function_name: {self.function_name}")
            
            message["tool_call_id"] = self.tool_call_id
            message["name"] = self.function_name
        
        return message


class SessionManager:
    """Manages agent sessions with lifecycle tracking."""
    
    def __init__(self):
        self._sessions: Dict[str, 'FinancialAgent'] = {}
        self._session_stats: Dict[str, Dict[str, Any]] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    # Do not start background asyncio tasks at import time. Defer starting
    # the cleanup task until an event loop is available by calling
    # `start_cleanup_task()` from runtime code or when a loop is running.
    
    def create_session(self, session_id: Optional[str] = None, **kwargs) -> str:
        """Create a new agent session."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        agent = FinancialAgent(session_id=session_id, **kwargs)
        self._sessions[session_id] = agent
        self._session_stats[session_id] = {
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "state": SessionState.INITIALIZING,
            "total_messages": 0,
            "total_tool_calls": 0,
        }
        
        active_sessions.inc()
        logger.info(f"Created session {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional['FinancialAgent']:
        """Get an existing session."""
        return self._sessions.get(session_id)
    
    def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        if session_id in self._sessions:
            agent = self._sessions.pop(session_id)
            self._session_stats.pop(session_id, None)
            active_sessions.dec()
            logger.info(f"Removed session {session_id}")
            return True
        return False
    
    def update_session_activity(self, session_id: str):
        """Update session last activity timestamp."""
        if session_id in self._session_stats:
            self._session_stats[session_id]["last_activity"] = datetime.now()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_inactive_sessions())

    def start_cleanup_task(self):
        """Public method to start the cleanup task when an asyncio loop is running.

        This is safe to call from runtime code (e.g., when the application starts up).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop; do not start task now.
            logger.debug("No running event loop; cleanup task start deferred")
            return

        # If we have a running loop, ensure the task is scheduled
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = loop.create_task(self._cleanup_inactive_sessions())
    
    async def _cleanup_inactive_sessions(self):
        """Clean up inactive sessions periodically."""
        while True:
            try:
                current_time = datetime.now()
                inactive_sessions = []
                
                for session_id, stats in self._session_stats.items():
                    last_activity = stats["last_activity"]
                    if (current_time - last_activity).total_seconds() > 3600:  # 1 hour
                        inactive_sessions.append(session_id)
                
                for session_id in inactive_sessions:
                    self.remove_session(session_id)
                    logger.info(f"Cleaned up inactive session {session_id}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(60)


# Global session manager
session_manager = SessionManager()


class FinancialTool:
    """Financial tool with advanced caching and monitoring."""
    
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        cache_ttl: int = 0,
        timeout: float = 30.0,
        rate_limit: Optional[int] = None
    ):
        self.name = name
        self.description = description
        self.func = func
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.rate_limit = rate_limit
        
        # Caching
        if cache_ttl > 0:
            self._cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        else:
            self._cache = None
        
        # Rate limiting
        if rate_limit:
            self._rate_limiter = TTLCache(maxsize=rate_limit, ttl=60)  # per minute
        else:
            self._rate_limiter = None
        
        # Metrics
        self._call_count = 0
        self._error_count = 0
        self._total_execution_time = 0.0
        self._last_error = None
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI tool format with schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_input": {
                            "type": "string",
                            "description": "Input parameters for the financial tool (JSON string)"
                        }
                    },
                    "required": ["tool_input"]
                }
            }
        }
    
    def _get_cache_key(self, tool_input: str, user_context: str = "") -> str:
        """Generate cache key with user context."""
        import hashlib
        combined = f"{self.name}:{tool_input}:{user_context}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _check_rate_limit(self, user_id: str = "default") -> bool:
        """Check if rate limit is exceeded."""
        if not self._rate_limiter:
            return True
        
        current_count = self._rate_limiter.get(user_id, 0)
        if current_count >= self.rate_limit:
            return False
        
        self._rate_limiter[user_id] = current_count + 1
        return True
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def call(
        self,
        tool_input: str,
        user_context: str = "",
        user_id: str = "default"
    ) -> ToolCallResult:
        """Execute tool with error handling and monitoring."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Rate limiting check
            if not self._check_rate_limit(user_id):
                tool_calls_total.labels(tool_name=self.name, status='rate_limited').inc()
                raise Exception(f"Rate limit exceeded for tool {self.name}")
            
            # Cache check
            cache_key = self._get_cache_key(tool_input, user_context)
            if self._cache and cache_key in self._cache:
                execution_time = time.time() - start_time
                tool_calls_total.labels(tool_name=self.name, status='cache_hit').inc()
                
                return ToolCallResult(
                    request_id=request_id,
                    tool_name=self.name,
                    result=self._cache[cache_key],
                    success=True,
                    execution_time=execution_time,
                    cached=True
                )
            
            # Execute function with timeout
            try:
                result = await asyncio.wait_for(
                    self.func(tool_input),
                    timeout=self.timeout
                )
                
                execution_time = time.time() - start_time
                self._call_count += 1
                self._total_execution_time += execution_time
                
                # Cache successful results
                if self._cache and result:
                    self._cache[cache_key] = result
                
                tool_calls_total.labels(tool_name=self.name, status='success').inc()
                query_duration.observe(execution_time)
                
                logger.debug(f"Tool {self.name} executed successfully in {execution_time:.3f}s")
                
                return ToolCallResult(
                    request_id=request_id,
                    tool_name=self.name,
                    result=result,
                    success=True,
                    execution_time=execution_time,
                    cached=False,
                    metadata={
                        "input_length": len(tool_input),
                        "output_length": len(str(result)),
                        "user_context": user_context[:100]  # Truncated for privacy
                    }
                )
                
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                self._error_count += 1
                error_msg = f"Tool {self.name} timed out after {self.timeout}s"
                self._last_error = error_msg
                
                tool_calls_total.labels(tool_name=self.name, status='timeout').inc()
                
                return ToolCallResult(
                    request_id=request_id,
                    tool_name=self.name,
                    result=None,
                    success=False,
                    error=error_msg,
                    execution_time=execution_time
                )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._error_count += 1
            error_msg = f"Tool {self.name} failed: {str(e)}"
            self._last_error = error_msg
            
            tool_calls_total.labels(tool_name=self.name, status='error').inc()
            logger.error(f"Tool execution failed: {error_msg}")
            
            return ToolCallResult(
                request_id=request_id,
                tool_name=self.name,
                result=None,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                metadata={"exception_type": type(e).__name__}
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get tool performance metrics."""
        avg_execution_time = (
            self._total_execution_time / max(1, self._call_count)
        )
        
        success_rate = (
            (self._call_count - self._error_count) / max(1, self._call_count)
        )
        
        return {
            "name": self.name,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "last_error": self._last_error,
            "cache_enabled": self._cache is not None,
            "rate_limit_enabled": self._rate_limiter is not None,
            "timeout": self.timeout
        }


class FinancialAgent:
    """Financial agent with production-ready features."""
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id or "default"
        self.llm_config = llm_config or get_default_config()
        self.provider: Optional[LLMProvider] = None
        self.state = SessionState.INITIALIZING
        
        # Conversation management
        self.conversation_history: List[ConversationMessage] = []
        self.max_history_length = settings.llm.max_tool_calls_per_request * 10
        
        # Tool management
        self.tools: Dict[str, FinancialTool] = {}
        self._initialize_tools()
        
        # Performance tracking
        self.metrics = {
            "session_start": datetime.now(),
            "total_messages": 0,
            "total_tool_calls": 0,
            "total_tokens_used": 0,
            "avg_response_time": 0.0,
            "error_count": 0,
            "last_activity": datetime.now()
        }
        
        # Caching and optimization
        self.response_cache = LRUCache(maxsize=100)
        
        # Add system prompt
        self._add_system_prompt()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_monitoring()
        
        logger.info(f"FinancialAgent initialized: session={self.session_id}")
    
    def _add_system_prompt(self):
        """Add system prompt."""
        system_content = f"""You are FinanceBud, an advanced AI financial assistant specialized in analyzing Indian bank transaction data.

Key capabilities:
- Real-time financial data analysis with INR currency formatting
- Transaction categorization and pattern recognition
- Spending trend analysis and forecasting
- Recurring payment detection
- UPI transaction analysis
- Custom financial reporting

Guidelines:
- All amounts are in Indian Rupees (INR)
- Provide accurate, data-driven insights
- Use appropriate tools to fetch real transaction data
- Format responses clearly with proper currency symbols
- Be concise yet comprehensive in analysis
- Handle errors gracefully and suggest alternatives

Session: {self.session_id}
User Context: {self.user_id}
Timestamp: {datetime.now().isoformat()}"""
        
        self.conversation_history.append(
            ConversationMessage(
                role="system",
                content=system_content,
                metadata={"version": "2.0"}
            )
        )
    
    def _initialize_tools(self):
        """Initialize financial tools."""
        from backend.agents.financial_tools import get_financial_tools
        
        tool_configs = get_financial_tools()
        
        for tool_config in tool_configs:
            tool = FinancialTool(**tool_config)
            self.tools[tool.name] = tool
        
        logger.info(f"Initialized {len(self.tools)} financial tools")
    
    async def initialize(self):
        """Initialize the agent with async setup."""
        try:
            # Initialize LLM provider
            self.provider = create_provider(self.llm_config)
            
            # Test provider connection
            await self._test_provider_connection()
            
            # Initialize database manager
            db_manager = get_db_manager()
            
            # Initialize MCP manager
            mcp_manager = await get_mcp_manager()
            
            self.state = SessionState.ACTIVE
            
            logger.info(f"FinancialAgent fully initialized: {self.session_id}")
            
        except Exception as e:
            self.state = SessionState.ERROR
            logger.error(f"Agent initialization failed: {e}")
            raise
    
    async def _test_provider_connection(self):
        """Test LLM provider connection."""
        if not self.provider:
            raise Exception("Provider not initialized")
        
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            await self.provider.chat_completion(messages=test_messages)
            logger.debug("LLM provider connection test successful")
        except Exception as e:
            logger.warning(f"LLM provider test failed: {e}")
            # Don't raise here - provider might work for actual requests
    
    def _start_monitoring(self):
        """Start background monitoring task."""
        if not self._monitoring_task or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitor_session())
    
    async def _monitor_session(self):
        """Monitor session health and performance."""
        while self.state in [SessionState.ACTIVE, SessionState.IDLE]:
            try:
                # Update session activity in session manager
                session_manager.update_session_activity(self.session_id)
                
                # Log periodic metrics
                if logger.isEnabledFor(logging.DEBUG):
                    metrics = self.get_session_metrics()
                    logger.debug(f"Session metrics: {metrics}")
                
                # Clean up old conversation history
                await self._cleanup_conversation_history()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Session monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_conversation_history(self):
        """Clean up old conversation history to prevent memory bloat."""
        if len(self.conversation_history) > self.max_history_length:
            # Keep system message and recent messages
            system_messages = [msg for msg in self.conversation_history if msg.role == "system"]
            recent_messages = self.conversation_history[-self.max_history_length//2:]
            
            self.conversation_history = system_messages + recent_messages
            logger.debug(f"Cleaned conversation history: {len(self.conversation_history)} messages remaining")
    
    async def process_message(
        self,
        message: str,
        max_iterations: int = 5,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Process user message."""
        start_time = time.time()
        
        try:
            self.state = SessionState.ACTIVE
            self.metrics["total_messages"] += 1
            self.metrics["last_activity"] = datetime.now()
            
            # Check cache for similar recent queries
            cache_key = self._get_message_cache_key(message)
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                logger.info(f"Returning cached response for message: {message[:50]}...")
                if stream:
                    return self._stream_cached_response(cached_response)
                return cached_response
            
            # Add user message
            user_message = ConversationMessage(
                role="user",
                content=message,
                metadata={"user_id": self.user_id}
            )
            self.conversation_history.append(user_message)
            
            if stream:
                return self._process_message_stream(message, max_iterations, start_time, cache_key)
            else:
                response = await self._process_message_standard(message, max_iterations, start_time)
                
                # Cache successful responses
                if response and len(response) > 10:  # Only cache substantial responses
                    self.response_cache[cache_key] = response
                
                return response
                
        except Exception as e:
            self.metrics["error_count"] += 1
            self.state = SessionState.ERROR
            error_msg = f"Error processing message: {str(e)}"
            logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
            
            if stream:
                return self._stream_error_response(error_msg)
            return error_msg
        
        finally:
            # Update metrics
            processing_time = time.time() - start_time
            self._update_response_time_metric(processing_time)
    
    def _get_message_cache_key(self, message: str) -> str:
        """Generate cache key for message."""
        import hashlib
        # Include user context in cache key
        cache_input = f"{self.user_id}:{message.lower().strip()}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    async def _stream_cached_response(self, response: str) -> AsyncGenerator[str, None]:
        """Stream a cached response."""
        words = response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.05)  # Simulate typing
    
    async def _stream_error_response(self, error_msg: str) -> AsyncGenerator[str, None]:
        """Stream an error response."""
        yield f"❌ {error_msg}"
    
    async def _process_message_stream(
        self,
        message: str,
        max_iterations: int,
        start_time: float,
        cache_key: str
    ) -> AsyncGenerator[str, None]:
        """Process message with streaming response."""
        try:
            response_parts = []
            
            async for part in self._execute_llm_with_tools_stream(message, max_iterations):
                response_parts.append(part)
                yield part
            
            # Cache the complete response
            complete_response = "".join(response_parts)
            if complete_response and len(complete_response) > 10:
                self.response_cache[cache_key] = complete_response
                
        except Exception as e:
            yield f"❌ Streaming error: {str(e)}"
    
    async def _process_message_standard(
        self,
        message: str,
        max_iterations: int,
        start_time: float
    ) -> str:
        """Process message with standard response."""
        response = await self._execute_llm_with_tools(message, max_iterations)
        
        processing_time = time.time() - start_time
        logger.info(f"Message processed in {processing_time:.3f}s: {message[:50]}...")
        
        return response
    
    async def _execute_llm_with_tools_stream(
        self,
        message: str,
        max_iterations: int
    ) -> AsyncGenerator[str, None]:
        """Execute LLM with tools and stream the response."""
        # For now, fall back to standard processing and stream the result
        # Future: implement true streaming with tool calls
        response = await self._execute_llm_with_tools(message, max_iterations)
        
        # Stream the response word by word
        words = response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.03)  # Adjust typing speed
    
    async def _execute_llm_with_tools(
        self,
        message: str,
        max_iterations: int
    ) -> str:
        """Execute LLM with tool calling."""
        tools = [tool.to_openai_format() for tool in self.tools.values()]
        total_tool_calls = 0
        
        for iteration in range(max_iterations):
            try:
                # Prepare conversation for LLM
                conversation = [msg.to_dict() for msg in self.conversation_history]
                
                # Call LLM with retry logic
                llm_start_time = time.time()
                response = await self._call_llm_with_retry(conversation, tools)
                llm_duration = time.time() - llm_start_time
                
                # Track LLM metrics
                llm_requests_total.labels(
                    provider=self.llm_config.provider.value,
                    status='success'
                ).inc()
                
                # Process response
                assistant_message = response["choices"][0]["message"]
                content = assistant_message.get("content", "")
                tool_calls = assistant_message.get("tool_calls", [])
                
                # Add assistant message
                self.conversation_history.append(
                    ConversationMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                        metadata={
                            "llm_duration": llm_duration,
                            "iteration": iteration,
                            "tool_count": len(tool_calls)
                        }
                    )
                )
                
                # If no tool calls, return response
                if not tool_calls:
                    return content or "I've processed your request."
                
                # Execute tool calls
                tool_results = await self._execute_tool_calls(tool_calls)
                total_tool_calls += len(tool_calls)
                self.metrics["total_tool_calls"] += len(tool_calls)
                
                # Add tool results to conversation
                for tool_call, result in zip(tool_calls, tool_results):
                    function_name = result.tool_name or tool_call.get('function', {}).get('name', 'unknown')
                    tool_call_id = tool_call.get('id', f'call_{uuid.uuid4().hex[:8]}')
                    
                    tool_content = (
                        result.result if result.success 
                        else f"Error: {result.error}"
                    )
                    
                    self.conversation_history.append(
                        ConversationMessage(
                            role="tool",
                            content=str(tool_content)[:2000],  # Truncate long responses
                            tool_call_id=tool_call_id,
                            function_name=function_name,
                            metadata={
                                "execution_time": result.execution_time,
                                "cached": result.cached,
                                "success": result.success
                            }
                        )
                    )
                
                # Continue for synthesis if configured
                if not settings.llm.skip_final_synthesis and iteration == max_iterations - 1:
                    # Final synthesis call
                    final_response = await self._call_llm_with_retry(
                        [msg.to_dict() for msg in self.conversation_history],
                        tools=None  # No tools for synthesis
                    )
                    
                    final_content = final_response["choices"][0]["message"]["content"]
                    if final_content:
                        return final_content
                
            except Exception as e:
                logger.error(f"LLM execution error in iteration {iteration}: {e}")
                llm_requests_total.labels(
                    provider=self.llm_config.provider.value,
                    status='error'
                ).inc()
                
                if iteration == max_iterations - 1:
                    return f"I encountered an error processing your request: {str(e)}"
        
        # Fallback response
        return f"I've processed your request using {total_tool_calls} tools. The information has been retrieved successfully."
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _call_llm_with_retry(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Call LLM with retry logic and error handling."""
        if not self.provider:
            raise Exception("LLM provider not initialized")
        
        try:
            return await self.provider.chat_completion(
                messages=messages,
                tools=tools
            )
        except Exception as e:
            logger.warning(f"LLM call failed, retrying: {e}")
            raise
    
    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolCallResult]:
        """Execute tool calls with parallel processing."""
        async def execute_single_tool(tool_call: Dict[str, Any]) -> ToolCallResult:
            try:
                function = tool_call["function"]
                tool_name = function["name"]
                arguments = json.loads(function["arguments"])
                tool_input = arguments.get("tool_input", "")
                
                if tool_name not in self.tools:
                    return ToolCallResult(
                        request_id=str(uuid.uuid4()),
                        tool_name=tool_name,
                        result=None,
                        success=False,
                        error=f"Tool {tool_name} not found"
                    )
                
                tool = self.tools[tool_name]
                return await tool.call(
                    tool_input=tool_input,
                    user_context=f"session:{self.session_id}",
                    user_id=self.user_id
                )
                
            except Exception as e:
                return ToolCallResult(
                    request_id=str(uuid.uuid4()),
                    tool_name=tool_call.get("function", {}).get("name", "unknown"),
                    result=None,
                    success=False,
                    error=f"Tool execution error: {str(e)}"
                )
        
        # Execute tool calls concurrently
        tasks = [execute_single_tool(tool_call) for tool_call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    ToolCallResult(
                        request_id=str(uuid.uuid4()),
                        tool_name=tool_calls[i].get("function", {}).get("name", "unknown"),
                        result=None,
                        success=False,
                        error=f"Execution exception: {str(result)}"
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _update_response_time_metric(self, processing_time: float):
        """Update response time metrics."""
        alpha = 0.1  # Smoothing factor
        if self.metrics["avg_response_time"] == 0:
            self.metrics["avg_response_time"] = processing_time
        else:
            self.metrics["avg_response_time"] = (
                alpha * processing_time + 
                (1 - alpha) * self.metrics["avg_response_time"]
            )
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """Get comprehensive session metrics."""
        session_duration = (datetime.now() - self.metrics["session_start"]).total_seconds()
        
        # Tool metrics
        tool_metrics = {name: tool.get_metrics() for name, tool in self.tools.items()}
        
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "session_duration_seconds": session_duration,
            "conversation_length": len(self.conversation_history),
            "cache_size": len(self.response_cache),
            "tools_available": len(self.tools),
            "tool_metrics": tool_metrics,
            **self.metrics
        }

    # Backwards-compatible alias for older callers
    def get_metrics(self) -> Dict[str, Any]:
        """Compatibility wrapper returning session metrics (alias)."""
        return self.get_session_metrics()

    def get_last_tools_used(self, limit: int = 5) -> List[str]:
        """Return the last N tool names used in this session, most recent first."""
        tools = []
        # Walk conversation history backwards and collect tool message function_names
        for msg in reversed(self.conversation_history):
            if getattr(msg, 'role', '') == 'tool':
                name = getattr(msg, 'function_name', None) or msg.metadata.get('function_name') if isinstance(msg.metadata, dict) else None
                if name and name not in tools:
                    tools.append(name)
            if len(tools) >= limit:
                break
        return tools
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status."""
        health_score = 100.0
        issues = []
        
        # Check provider health
        if not self.provider:
            health_score -= 50
            issues.append("LLM provider not initialized")
        
        # Check error rate
        error_rate = (
            self.metrics["error_count"] / max(1, self.metrics["total_messages"])
        )
        if error_rate > 0.1:
            health_score -= 20
            issues.append(f"High error rate: {error_rate:.2%}")
        
        # Check response time
        if self.metrics["avg_response_time"] > 30.0:
            health_score -= 15
            issues.append("Slow response times")
        
        # Check memory usage (conversation history)
        if len(self.conversation_history) > self.max_history_length * 0.9:
            health_score -= 10
            issues.append("High memory usage")
        
        return {
            "session_id": self.session_id,
            "health_score": max(0, health_score),
            "status": "healthy" if health_score > 80 else "degraded" if health_score > 50 else "unhealthy",
            "issues": issues,
            "state": self.state.value,
            "timestamp": datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Graceful shutdown of the agent."""
        logger.info(f"Shutting down agent session: {self.session_id}")
        
        self.state = SessionState.TERMINATED
        
        # Cancel monitoring task
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Clear caches
        self.response_cache.clear()
        if hasattr(self, 'tools'):
            for tool in self.tools.values():
                if tool._cache:
                    tool._cache.clear()
        
        logger.info(f"Agent session shutdown complete: {self.session_id}")


# Global agent registry for backward compatibility
_global_agent: Optional[FinancialAgent] = None


async def get_financial_agent(
    llm_config: Optional[LLMConfig] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> FinancialAgent:
    """Get or create financial agent."""
    global _global_agent
    
    if _global_agent is None or _global_agent.state == SessionState.TERMINATED:
        _global_agent = FinancialAgent(
            llm_config=llm_config,
            session_id=session_id,
            user_id=user_id
        )
        await _global_agent.initialize()
        # Ensure background cleanup task is started now that an event loop is running
        try:
            session_manager.start_cleanup_task()
        except Exception:
            logger.debug("Failed to start session cleanup task during agent initialization")
    
    return _global_agent


# Start Prometheus metrics server
def start_metrics_server():
    """Start Prometheus metrics server."""
    if settings.monitoring.enable_metrics:
        try:
            start_http_server(settings.monitoring.metrics_port)
            logger.info(f"Metrics server started on port {settings.monitoring.metrics_port}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")


# Initialize metrics server
if settings.monitoring.enable_metrics:
    start_metrics_server()
