"""
Optimized Financial Analysis Agent
=================================

High-performance financial agent that uses persistent MCP connections
for significant performance improvements. This eliminates the overhead
of repeatedly starting MCP server processes.

Key optimizations:
- Persistent MCP server connections
- Connection pooling and reuse
- Optimized tool calls with caching
- Reduced database connection overhead
- Batch processing capabilities
- Response caching for frequent queries
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import hashlib
from functools import lru_cache

# Import our generic LLM providers
from .llm_providers import (
    LLMConfig, LLMProvider, ProviderType,
    create_provider, get_default_config
)

# Import persistent MCP client
from ..mcp.persistent_client import get_persistent_mcp_manager, PersistentMCPManager

# Import centralized logging
from ..logging_config import get_logger_with_context

logger = get_logger_with_context(__name__)

class ToolCallResult(BaseModel):
    """Result of a tool call."""
    tool_name: str
    result: str
    success: bool
    error: Optional[str] = None

class ConversationMessage(BaseModel):
    """Conversation message with tool support."""
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class OptimizedFinancialTool:
    """Represents an optimized financial analysis tool."""
    
    def __init__(self, name: str, description: str, func, cache_ttl: int = 0):
        self.name = name
        self.description = description
        self.func = func
        self.cache_ttl = cache_ttl  # Cache time-to-live in seconds
        self._cache = {}
        self._cache_times = {}
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI tool format for LLM providers."""
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
                            "description": "Input parameters for the financial tool"
                        }
                    },
                    "required": ["tool_input"]
                }
            }
        }
    
    def _get_cache_key(self, tool_input: str) -> str:
        """Generate cache key for tool input."""
        return hashlib.md5(f"{self.name}:{tool_input}".encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if self.cache_ttl <= 0:
            return False
        
        if cache_key not in self._cache_times:
            return False
        
        cache_time = self._cache_times[cache_key]
        return (time.time() - cache_time) < self.cache_ttl
    
    async def call(self, tool_input: str) -> str:
        """Call the tool with caching support."""
        cache_key = self._get_cache_key(tool_input)
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {self.name}")
            return self._cache[cache_key]
        
        # Call the actual function
        start_time = time.time()
        result = await self.func(tool_input)
        execution_time = time.time() - start_time
        
        # Cache the result if caching is enabled
        if self.cache_ttl > 0:
            self._cache[cache_key] = result
            self._cache_times[cache_key] = time.time()
            
            # Clean old cache entries
            self._cleanup_cache()
        
        logger.debug(f"Tool {self.name} executed in {execution_time:.3f}s")
        return result
    
    def _cleanup_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, cache_time in self._cache_times.items()
            if (current_time - cache_time) > self.cache_ttl
        ]
        
        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_times.pop(key, None)

# Global MCP manager instance
_mcp_manager: Optional[PersistentMCPManager] = None

async def _get_mcp_manager() -> PersistentMCPManager:
    """Get the persistent MCP manager instance."""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = await get_persistent_mcp_manager()
    return _mcp_manager

# Optimized financial analysis tools
async def get_account_summary_tool(tool_input: str) -> str:
    """Get account summary with current balance and transaction overview."""
    try:
        mcp_manager = await _get_mcp_manager()
        result = await mcp_manager.call_tool("financial-data-inr", "get_account_summary", {})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting account summary: {e}")
        return f"Error getting account summary: {str(e)}"

async def get_recent_transactions_tool(tool_input: str) -> str:
    """Get the most recent N transactions."""
    try:
        # Parse input to get number of transactions
        try:
            parsed_input = json.loads(tool_input) if tool_input.strip() else {}
            limit = parsed_input.get("limit", 10)
        except:
            # If input is just a number
            try:
                limit = int(tool_input.strip()) if tool_input.strip() else 10
            except:
                limit = 10
        
        mcp_manager = await _get_mcp_manager()
        result = await mcp_manager.call_tool("financial-data-inr", "get_recent_transactions", {"limit": limit})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting recent transactions: {e}")
        return f"Error getting recent transactions: {str(e)}"

async def search_transactions_tool(tool_input: str) -> str:
    """Search transactions by description pattern."""
    try:
        # Parse input to get search pattern and optional limit
        try:
            parsed_input = json.loads(tool_input) if tool_input.strip() else {}
            pattern = parsed_input.get("pattern", tool_input)
            limit = parsed_input.get("limit", 20)
        except:
            pattern = tool_input.strip()
            limit = 20
        
        if not pattern:
            return "Error: Search pattern is required"
        
        mcp_manager = await _get_mcp_manager()
        result = await mcp_manager.call_tool("financial-data-inr", "search_transactions", {
            "pattern": pattern,
            "limit": limit
        })
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error searching transactions: {e}")
        return f"Error searching transactions: {str(e)}"

async def get_transactions_by_date_range_tool(tool_input: str) -> str:
    """Get transactions within a specific date range."""
    try:
        parsed_input = json.loads(tool_input) if tool_input.strip() else {}
        start_date = parsed_input.get("start_date")
        end_date = parsed_input.get("end_date")
        
        if not start_date or not end_date:
            return "Error: Both start_date and end_date are required in YYYY-MM-DD format"
        
        mcp_manager = await _get_mcp_manager()
        result = await mcp_manager.call_tool("financial-data-inr", "get_transactions_by_date_range", {
            "start_date": start_date,
            "end_date": end_date
        })
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting transactions by date range: {e}")
        return f"Error getting transactions by date range: {str(e)}"

async def get_monthly_summary_tool(tool_input: str) -> str:
    """Get monthly spending summary."""
    try:
        # Parse input to get optional year and month
        try:
            parsed_input = json.loads(tool_input) if tool_input.strip() else {}
            year = parsed_input.get("year")
            month = parsed_input.get("month")
        except:
            year = None
            month = None
        
        arguments = {}
        if year:
            arguments["year"] = year
        if month:
            arguments["month"] = month
        
        mcp_manager = await _get_mcp_manager()
        result = await mcp_manager.call_tool("financial-data-inr", "get_monthly_summary", arguments)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting monthly summary: {e}")
        return f"Error getting monthly summary: {str(e)}"

async def get_spending_by_category_tool(tool_input: str) -> str:
    """Analyze spending by category."""
    try:
        # Parse input for optional time period
        try:
            parsed_input = json.loads(tool_input) if tool_input.strip() else {}
            days = parsed_input.get("days", 30)
        except:
            days = 30
        
        mcp_manager = await _get_mcp_manager()
        result = await mcp_manager.call_tool("financial-data-inr", "get_spending_by_category", {"days": days})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting spending by category: {e}")
        return f"Error getting spending by category: {str(e)}"

async def get_upi_transaction_analysis_tool(tool_input: str) -> str:
    """Analyze UPI transactions."""
    try:
        # Parse input for optional time period
        try:
            parsed_input = json.loads(tool_input) if tool_input.strip() else {}
            days = parsed_input.get("days", 30)
        except:
            days = 30
        
        mcp_manager = await _get_mcp_manager()
        result = await mcp_manager.call_tool("financial-data-inr", "get_upi_transaction_analysis", {"days": days})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting UPI analysis: {e}")
        return f"Error getting UPI analysis: {str(e)}"

async def find_recurring_payments_tool(tool_input: str) -> str:
    """Find recurring payments."""
    try:
        mcp_manager = await _get_mcp_manager()
        result = await mcp_manager.call_tool("financial-data-inr", "find_recurring_payments", {})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error finding recurring payments: {e}")
        return f"Error finding recurring payments: {str(e)}"

async def analyze_spending_trends_tool(tool_input: str) -> str:
    """Analyze spending trends over time."""
    try:
        # Parse input for optional time period
        try:
            parsed_input = json.loads(tool_input) if tool_input.strip() else {}
            months = parsed_input.get("months", 6)
        except:
            months = 6
        
        mcp_manager = await _get_mcp_manager()
        result = await mcp_manager.call_tool("financial-data-inr", "analyze_spending_trends", {"months": months})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error analyzing spending trends: {e}")
        return f"Error analyzing spending trends: {str(e)}"

async def get_balance_history_tool(tool_input: str) -> str:
    """Get account balance history."""
    try:
        # Parse input for optional time period
        try:
            parsed_input = json.loads(tool_input) if tool_input.strip() else {}
            days = parsed_input.get("days", 30)
        except:
            days = 30
        
        mcp_manager = await _get_mcp_manager()
        result = await mcp_manager.call_tool("financial-data-inr", "get_balance_history", {"days": days})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting balance history: {e}")
        return f"Error getting balance history: {str(e)}"

async def execute_custom_query_tool(tool_input: str) -> str:
    """Execute a custom SQL query (SELECT only)."""
    try:
        # Parse input to get SQL query
        try:
            parsed_input = json.loads(tool_input) if tool_input.strip() else {}
            sql_query = parsed_input.get("query", tool_input)
        except:
            sql_query = tool_input.strip()
        
        if not sql_query:
            return "Error: SQL query is required"
        
        mcp_manager = await _get_mcp_manager()
        result = await mcp_manager.call_tool("financial-data-inr", "execute_custom_query", {"sql_query": sql_query})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error executing custom query: {e}")
        return f"Error executing custom query: {str(e)}"

async def get_database_schema_tool(tool_input: str) -> str:
    """Get database schema information."""
    try:
        mcp_manager = await _get_mcp_manager()
        result = await mcp_manager.call_tool("financial-data-inr", "get_database_schema", {})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting database schema: {e}")
        return f"Error getting database schema: {str(e)}"

# Define optimized financial tools with caching
OPTIMIZED_FINANCIAL_TOOLS = [
    OptimizedFinancialTool(
        "get_account_summary",
        "Get current account balance and transaction summary. Provides overview of total transactions, date range, current balance in INR, and total debits/credits.",
        get_account_summary_tool,
        cache_ttl=30  # Cache for 30 seconds
    ),
    OptimizedFinancialTool(
        "get_recent_transactions", 
        "Get the most recent N transactions. Input: {\"limit\": 10} or just a number. Shows transaction details including amounts in INR.",
        get_recent_transactions_tool,
        cache_ttl=10  # Cache for 10 seconds
    ),
    OptimizedFinancialTool(
        "search_transactions",
        "Search transactions by description pattern. Input: {\"pattern\": \"search_term\", \"limit\": 20} or just the search term. All amounts shown in INR.",
        search_transactions_tool,
        cache_ttl=60  # Cache for 1 minute
    ),
    OptimizedFinancialTool(
        "get_transactions_by_date_range",
        "Get transactions within a date range. Input: {\"start_date\": \"YYYY-MM-DD\", \"end_date\": \"YYYY-MM-DD\"}. Returns transactions with INR amounts.",
        get_transactions_by_date_range_tool,
        cache_ttl=300  # Cache for 5 minutes
    ),
    OptimizedFinancialTool(
        "get_monthly_summary",
        "Get monthly spending summary. Input: {\"year\": 2024, \"month\": 1} (optional). Shows monthly totals in INR.",
        get_monthly_summary_tool,
        cache_ttl=300  # Cache for 5 minutes
    ),
    OptimizedFinancialTool(
        "get_spending_by_category",
        "Analyze spending by category for the last N days. Input: {\"days\": 30} or empty for 30 days. Categories based on transaction descriptions, amounts in INR.",
        get_spending_by_category_tool,
        cache_ttl=180  # Cache for 3 minutes
    ),
    OptimizedFinancialTool(
        "get_upi_transaction_analysis",
        "Analyze UPI payment transactions for the last N days. Input: {\"days\": 30} or empty for 30 days. Shows UPI-specific insights with INR amounts.",
        get_upi_transaction_analysis_tool,
        cache_ttl=180  # Cache for 3 minutes
    ),
    OptimizedFinancialTool(
        "find_recurring_payments",
        "Find recurring payments and subscriptions. No input required. Identifies potential recurring transactions with INR amounts.",
        find_recurring_payments_tool,
        cache_ttl=600  # Cache for 10 minutes
    ),
    OptimizedFinancialTool(
        "analyze_spending_trends",
        "Analyze spending trends over time. Input: {\"months\": 6} or empty for 6 months. Shows spending patterns with INR amounts.",
        analyze_spending_trends_tool,
        cache_ttl=600  # Cache for 10 minutes
    ),
    OptimizedFinancialTool(
        "get_balance_history",
        "Get account balance history for the last N days. Input: {\"days\": 30} or empty for 30 days. Shows balance trends in INR.",
        get_balance_history_tool,
        cache_ttl=300  # Cache for 5 minutes
    ),
    OptimizedFinancialTool(
        "execute_custom_query",
        "Execute a custom SQL query (SELECT only). Input: {\"query\": \"SELECT * FROM transactions LIMIT 5\"} or just the SQL. Results show INR amounts.",
        execute_custom_query_tool,
        cache_ttl=0  # No caching for custom queries
    ),
    OptimizedFinancialTool(
        "get_database_schema",
        "Get database schema and table structure information. No input required. Shows table definitions and sample data.",
        get_database_schema_tool,
        cache_ttl=3600  # Cache for 1 hour
    )
]

class OptimizedFinancialAgent:
    """Optimized financial analysis agent with persistent MCP connections."""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or get_default_config()
        self.provider: Optional[LLMProvider] = None
        self.tools = {tool.name: tool for tool in OPTIMIZED_FINANCIAL_TOOLS}
        self.conversation_history: List[ConversationMessage] = []
        self.session_id = str(uuid.uuid4())
        
        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "total_tool_calls": 0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "session_start": datetime.now()
        }
        
        logger.info(f"Initialized OptimizedFinancialAgent with session {self.session_id}")
    
    async def initialize(self):
        """Initialize the agent and ensure MCP connections."""
        try:
            # Initialize LLM provider
            self.provider = create_provider(self.llm_config)
            
            # Ensure MCP manager is initialized
            await _get_mcp_manager()
            
            logger.info(f"OptimizedFinancialAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OptimizedFinancialAgent: {e}")
            raise
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Get tools in OpenAI format for the LLM."""
        return [tool.to_openai_format() for tool in self.tools.values()]
    
    async def call_tool(self, tool_name: str, tool_input: str) -> ToolCallResult:
        """Call a specific tool with optimizations."""
        start_time = time.time()
        
        if tool_name not in self.tools:
            return ToolCallResult(
                tool_name=tool_name,
                result="",
                success=False,
                error=f"Tool '{tool_name}' not found"
            )
        
        try:
            tool = self.tools[tool_name]
            result = await tool.call(tool_input)
            
            execution_time = time.time() - start_time
            self._update_metrics(execution_time, True)
            
            logger.log_tool_call(
                tool_name, 
                {"tool_input": tool_input}, 
                result=result[:200] + "..." if len(result) > 200 else result,
                execution_time=execution_time,
                session_id=self.session_id
            )
            
            return ToolCallResult(
                tool_name=tool_name,
                result=result,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(execution_time, False)
            
            error_msg = str(e)
            logger.error(f"Tool call failed for {tool_name}: {error_msg}")
            
            return ToolCallResult(
                tool_name=tool_name,
                result="",
                success=False,
                error=error_msg
            )
    
    async def process_message(self, message: str, max_iterations: int = 5) -> str:
        """Process a user message with optimized tool calling."""
        start_time = time.time()
        
        if not self.provider:
            await self.initialize()
        
        if not self.provider:
            return "Error: Failed to initialize LLM provider"
        
        # Add user message to conversation
        self.conversation_history.append(
            ConversationMessage(role="user", content=message)
        )
        
        # Get available tools
        tools = self.get_openai_tools()
        total_tool_calls = 0
        
        try:
            for iteration in range(max_iterations):
                logger.debug(f"Processing iteration {iteration + 1}/{max_iterations}")
                
                # Call LLM with conversation history and tools
                response = await self.provider.chat_completion(
                    messages=[msg.dict() for msg in self.conversation_history],
                    tools=tools
                )
                
                # Extract response content and tool calls
                assistant_message = response["choices"][0]["message"]
                content = assistant_message.get("content", "")
                tool_calls = assistant_message.get("tool_calls", [])
                
                # Add assistant message to conversation
                self.conversation_history.append(
                    ConversationMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls
                    )
                )
                
                # If no tool calls, we're done
                if not tool_calls:
                    processing_time = time.time() - start_time
                    self._update_query_metrics(processing_time)
                    logger.info(f"Query processed in {processing_time:.3f}s without tool calls")
                    return content
                
                # Process tool calls in parallel for better performance
                tool_results = await self._process_tool_calls_parallel(tool_calls)
                total_tool_calls += len(tool_calls)
                
                # Add tool results to conversation
                for tool_call, result in zip(tool_calls, tool_results):
                    self.conversation_history.append(
                        ConversationMessage(
                            role="tool",
                            content=result.result if result.success else f"Error: {result.error}",
                            tool_call_id=tool_call["id"]
                        )
                    )
                
                # Check if all tool calls were successful
                all_successful = all(result.success for result in tool_results)
                if not all_successful:
                    logger.warning("Some tool calls failed, continuing with partial results")
            
            # Final LLM call to synthesize results
            final_response = await self.provider.chat_completion(
                messages=[msg.dict() for msg in self.conversation_history]
            )
            
            final_content = final_response["choices"][0]["message"]["content"]
            
            processing_time = time.time() - start_time
            self._update_query_metrics(processing_time)
            
            logger.info(f"Query processed in {processing_time:.3f}s with {total_tool_calls} tool calls")
            return final_content
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing message: {str(e)}"
            logger.error(f"{error_msg} (processing time: {processing_time:.3f}s)")
            return error_msg
    
    async def _process_tool_calls_parallel(self, tool_calls: List[Dict[str, Any]]) -> List[ToolCallResult]:
        """Process multiple tool calls in parallel for better performance."""
        async def process_single_tool_call(tool_call):
            try:
                function = tool_call["function"]
                tool_name = function["name"]
                arguments = json.loads(function["arguments"])
                tool_input = arguments.get("tool_input", "")
                
                return await self.call_tool(tool_name, tool_input)
                
            except Exception as e:
                return ToolCallResult(
                    tool_name=tool_call.get("function", {}).get("name", "unknown"),
                    result="",
                    success=False,
                    error=str(e)
                )
        
        # Process all tool calls concurrently
        tasks = [process_single_tool_call(tool_call) for tool_call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ToolCallResult(
                    tool_name=tool_calls[i].get("function", {}).get("name", "unknown"),
                    result="",
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _update_metrics(self, execution_time: float, success: bool):
        """Update tool call metrics."""
        self.metrics["total_tool_calls"] += 1
        
        # Update average response time (exponential moving average)
        alpha = 0.1
        if self.metrics["avg_response_time"] == 0:
            self.metrics["avg_response_time"] = execution_time
        else:
            self.metrics["avg_response_time"] = (
                alpha * execution_time + 
                (1 - alpha) * self.metrics["avg_response_time"]
            )
    
    def _update_query_metrics(self, processing_time: float):
        """Update query processing metrics."""
        self.metrics["total_queries"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        uptime = (datetime.now() - self.metrics["session_start"]).total_seconds()
        
        return {
            **self.metrics,
            "session_id": self.session_id,
            "uptime_seconds": uptime,
            "conversation_length": len(self.conversation_history),
            "tools_available": len(self.tools)
        }
    
    def clear_conversation(self):
        """Clear conversation history to free memory."""
        self.conversation_history.clear()
        logger.info(f"Cleared conversation history for session {self.session_id}")

# Global optimized agent instance
_optimized_agent: Optional[OptimizedFinancialAgent] = None

async def get_optimized_financial_agent(llm_config: Optional[LLMConfig] = None) -> OptimizedFinancialAgent:
    """Get or create the global optimized financial agent."""
    global _optimized_agent
    
    if _optimized_agent is None:
        _optimized_agent = OptimizedFinancialAgent(llm_config)
        await _optimized_agent.initialize()
    
    return _optimized_agent

# Backward compatibility - alias for the original function name
async def get_financial_agent(llm_config: Optional[LLMConfig] = None) -> OptimizedFinancialAgent:
    """Get financial agent (backward compatibility)."""
    return await get_optimized_financial_agent(llm_config)
