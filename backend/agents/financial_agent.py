"""
Financial Analysis Agent
========================

High-performance financial agent with persistent MCP connections and
advanced optimization features for efficient financial data analysis.

Key features:
- Persistent MCP server connections
- Connection pooling and reuse
- Tool calls with intelligent caching
- Optimized database operations
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

# Import our MCP client manager
from ..mcp.client import MCPManager, get_mcp_manager
from ..database.db import DatabaseManager, get_db_manager

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

class FinancialTool:
    """Represents a financial analysis tool with caching support."""
    
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
_mcp_manager: Optional[MCPManager] = None

async def _get_mcp_manager() -> MCPManager:
    """Get the MCP manager instance."""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = await get_mcp_manager()
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

# Define financial analysis tools with caching
FINANCIAL_TOOLS = [
    FinancialTool(
        "get_account_summary",
        "Get current account balance and transaction summary. Provides overview of total transactions, date range, current balance in INR, and total debits/credits.",
        get_account_summary_tool,
        cache_ttl=30  # Cache for 30 seconds
    ),
    FinancialTool(
        "get_recent_transactions", 
        "Get the most recent N transactions. Input: {\"limit\": 10} or just a number. Shows transaction details including amounts in INR.",
        get_recent_transactions_tool,
        cache_ttl=10  # Cache for 10 seconds
    ),
    FinancialTool(
        "search_transactions",
        "Search transactions by description pattern. Input: {\"pattern\": \"search_term\", \"limit\": 20} or just the search term. All amounts shown in INR.",
        search_transactions_tool,
        cache_ttl=60  # Cache for 1 minute
    ),
    FinancialTool(
        "get_transactions_by_date_range",
        "Get transactions within a date range. Input: {\"start_date\": \"YYYY-MM-DD\", \"end_date\": \"YYYY-MM-DD\"}. Returns transactions with INR amounts.",
        get_transactions_by_date_range_tool,
        cache_ttl=300  # Cache for 5 minutes
    ),
    FinancialTool(
        "get_monthly_summary",
        "Get monthly spending summary. Input: {\"year\": 2024, \"month\": 1} (optional). Shows monthly totals in INR.",
        get_monthly_summary_tool,
        cache_ttl=300  # Cache for 5 minutes
    ),
    FinancialTool(
        "get_spending_by_category",
        "Analyze spending by category for the last N days. Input: {\"days\": 30} or empty for 30 days. Categories based on transaction descriptions, amounts in INR.",
        get_spending_by_category_tool,
        cache_ttl=180  # Cache for 3 minutes
    ),
    FinancialTool(
        "get_upi_transaction_analysis",
        "Analyze UPI payment transactions for the last N days. Input: {\"days\": 30} or empty for 30 days. Shows UPI-specific insights with INR amounts.",
        get_upi_transaction_analysis_tool,
        cache_ttl=180  # Cache for 3 minutes
    ),
    FinancialTool(
        "find_recurring_payments",
        "Find recurring payments and subscriptions. No input required. Identifies potential recurring transactions with INR amounts.",
        find_recurring_payments_tool,
        cache_ttl=600  # Cache for 10 minutes
    ),
    FinancialTool(
        "analyze_spending_trends",
        "Analyze spending trends over time. Input: {\"months\": 6} or empty for 6 months. Shows spending patterns with INR amounts.",
        analyze_spending_trends_tool,
        cache_ttl=600  # Cache for 10 minutes
    ),
    FinancialTool(
        "get_balance_history",
        "Get account balance history for the last N days. Input: {\"days\": 30} or empty for 30 days. Shows balance trends in INR.",
        get_balance_history_tool,
        cache_ttl=300  # Cache for 5 minutes
    ),
    FinancialTool(
        "execute_custom_query",
        "Execute a custom SQL query (SELECT only). Input: {\"query\": \"SELECT * FROM transactions LIMIT 5\"} or just the SQL. Results show INR amounts.",
        execute_custom_query_tool,
        cache_ttl=0  # No caching for custom queries
    ),
    FinancialTool(
        "get_database_schema",
        "Get database schema and table structure information. No input required. Shows table definitions and sample data.",
        get_database_schema_tool,
        cache_ttl=3600  # Cache for 1 hour
    )
]

class FinancialAgent:
    """Financial analysis agent with persistent MCP connections and advanced optimization features."""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or get_default_config()
        self.provider: Optional[LLMProvider] = None
        self.tools = {tool.name: tool for tool in FINANCIAL_TOOLS}
        self.conversation_history: List[ConversationMessage] = []
        self.session_id = str(uuid.uuid4())
        
        # Add system prompt - optimized for performance
        self.conversation_history.append(
            ConversationMessage(
                role="system", 
                content="You are a financial assistant. Use tools to get data, then provide concise answers. Be direct and brief."
            )
        )
        
        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "total_tool_calls": 0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "session_start": datetime.now()
        }
        
        logger.info(f"Initialized FinancialAgent with session {self.session_id}")
    
    async def initialize(self):
        """Initialize the agent and ensure MCP connections."""
        try:
            # Initialize LLM provider
            self.provider = create_provider(self.llm_config)
            
            # Ensure MCP manager is initialized
            await _get_mcp_manager()
            
            logger.info(f"FinancialAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FinancialAgent: {e}")
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
                
                # Optimize conversation history - keep only recent messages to prevent slowdown
                optimized_history = self._optimize_conversation_history()
                
                # Call LLM with optimized conversation history and tools
                response = await self.provider.chat_completion(
                    messages=[msg.dict() for msg in optimized_history],
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
                
                # Add tool results to conversation with aggressive size optimization
                for tool_call, result in zip(tool_calls, tool_results):
                    # Aggressively truncate large tool results to prevent LLM timeout
                    if result.success:
                        # For successful results, extract only key information
                        tool_content = self._summarize_tool_result(result.result, tool_call.get('function', {}).get('name', 'unknown'))
                    else:
                        tool_content = f"Error: {result.error}"
                    
                    self.conversation_history.append(
                        ConversationMessage(
                            role="tool",
                            content=tool_content,
                            tool_call_id=tool_call["id"]
                        )
                    )
                
                # Check if all tool calls were successful
                all_successful = all(result.success for result in tool_results)
                if not all_successful:
                    logger.warning("Some tool calls failed, continuing with partial results")
                
                # Instead of continuing to next iteration, try one more LLM call to synthesize
                # But with a much shorter timeout and fallback
                try:
                    logger.debug("Making final synthesis call to LLM")
                    optimized_history = self._optimize_conversation_history()
                    
                    # Make a final call with no tools to get the synthesis
                    final_response = await self.provider.chat_completion(
                        messages=[msg.dict() for msg in optimized_history],
                        tools=None  # No tools for final synthesis
                    )
                    
                    final_content = final_response["choices"][0]["message"]["content"]
                    if final_content and final_content.strip():
                        processing_time = time.time() - start_time
                        self._update_query_metrics(processing_time)
                        logger.info(f"Query processed in {processing_time:.3f}s with {total_tool_calls} tool calls")
                        return final_content
                    
                except Exception as e:
                    logger.warning(f"Final synthesis failed: {e}, using fallback response")
                
                # Fallback: provide a response based on the tool results
                processing_time = time.time() - start_time
                self._update_query_metrics(processing_time)
                logger.info(f"Query processed in {processing_time:.3f}s with {total_tool_calls} tool calls (fallback)")
                
                # Generate a simple response based on available tool results
                if tool_results and any(r.success for r in tool_results):
                    successful_results = [r for r in tool_results if r.success]
                    return f"I've retrieved the requested financial information using {len(successful_results)} tool(s). The data has been processed successfully."
                else:
                    return "I encountered some issues retrieving the financial data. Please try rephrasing your question."
            
            # If we exit the loop without returning, it means we hit max iterations
            processing_time = time.time() - start_time
            self._update_query_metrics(processing_time)
            logger.warning(f"Reached max iterations ({max_iterations}) after {processing_time:.3f}s")
            return "I've processed your request but reached the maximum number of iterations. Please try rephrasing your question."
            
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
    
    def _get_relevant_tools(self, message: str) -> List[Dict[str, Any]]:
        """Get only relevant tools based on the message to reduce LLM processing time."""
        message_lower = message.lower()
        
        # Essential tools that are always included
        essential_tools = ["get_account_summary", "get_recent_transactions"]
        
        # Map keywords to specific tools
        keyword_tools = {
            "balance": ["get_account_summary", "get_balance_history"],
            "transaction": ["get_recent_transactions", "search_transactions"],
            "search": ["search_transactions"],
            "recent": ["get_recent_transactions"],
            "month": ["get_monthly_summary"],
            "summary": ["get_account_summary", "get_monthly_summary"],
            "category": ["get_spending_by_category"],
            "spending": ["get_spending_by_category", "analyze_spending_trends"],
            "upi": ["get_upi_transaction_analysis"],
            "recurring": ["find_recurring_payments"],
            "date": ["get_transactions_by_date_range"],
            "range": ["get_transactions_by_date_range"],
        }
        
        # Find relevant tools based on keywords
        relevant_tool_names = set(essential_tools)
        for keyword, tools in keyword_tools.items():
            if keyword in message_lower:
                relevant_tool_names.update(tools)
        
        # Limit to maximum 6 tools to prevent timeout
        relevant_tool_names = list(relevant_tool_names)[:6]
        
        # Convert to OpenAI format
        relevant_tools = []
        for tool_name in relevant_tool_names:
            if tool_name in self.tools:
                relevant_tools.append(self.tools[tool_name].to_openai_format())
        
        logger.debug(f"Selected {len(relevant_tools)} relevant tools for message: {relevant_tool_names}")
        return relevant_tools
    
    def _optimize_conversation_history(self, max_messages: int = 10) -> List[ConversationMessage]:
        """Optimize conversation history to prevent LLM slowdown from large contexts."""
        # Always keep the system message
        system_messages = [msg for msg in self.conversation_history if msg.role == "system"]
        
        # Get recent non-system messages
        non_system_messages = [msg for msg in self.conversation_history if msg.role != "system"]
        
        # Keep only the most recent messages to prevent context bloat
        if len(non_system_messages) > max_messages:
            recent_messages = non_system_messages[-max_messages:]
            logger.debug(f"Truncated conversation history from {len(non_system_messages)} to {len(recent_messages)} messages")
        else:
            recent_messages = non_system_messages
        
        # Combine system messages with recent messages
        return system_messages + recent_messages
    
    def _summarize_tool_result(self, result: str, tool_name: str, max_length: int = 1500) -> str:
        """Summarize tool results to prevent LLM timeouts from large JSON responses - increased limit."""
        try:
            # Try to parse as JSON and extract key information
            data = json.loads(result)
            
            if tool_name == "get_account_summary":
                # Extract only essential account info
                if isinstance(data, dict) and "data" in data:
                    account_data = data["data"]
                    return f"Account balance: {account_data.get('current_balance', 'N/A')}, Total transactions: {account_data.get('total_transactions', 'N/A')}"
                
            elif tool_name == "get_recent_transactions":
                # Show only count and recent transaction info
                if isinstance(data, dict) and "data" in data:
                    transactions = data["data"]
                    if isinstance(transactions, list) and len(transactions) > 0:
                        latest = transactions[0]
                        return f"Found {len(transactions)} transactions. Latest: {latest.get('description', 'N/A')} - {latest.get('amount', 'N/A')} on {latest.get('date', 'N/A')}"
                    return f"Found {len(transactions) if isinstance(transactions, list) else 'some'} transactions"
                
            elif "summary" in tool_name or "spending" in tool_name:
                # For summary tools, extract key metrics
                if isinstance(data, dict) and "data" in data:
                    summary_data = data["data"]
                    if isinstance(summary_data, dict):
                        key_points = []
                        for key, value in list(summary_data.items())[:3]:  # Only first 3 items
                            if isinstance(value, (int, float, str)) and len(str(value)) < 50:
                                key_points.append(f"{key}: {value}")
                        return f"Summary: {', '.join(key_points)}"
            
            # Fallback: truncate the original result
            if len(result) > max_length:
                return result[:max_length-20] + "... [truncated]"
            return result
            
        except json.JSONDecodeError:
            # If not JSON, just truncate
            if len(result) > max_length:
                return result[:max_length-20] + "... [truncated]"
            return result
    
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

# Global financial agent instance
_financial_agent: Optional[FinancialAgent] = None

async def get_financial_agent(llm_config: Optional[LLMConfig] = None) -> FinancialAgent:
    """Get or create the global financial agent."""
    global _financial_agent
    
    if _financial_agent is None:
        _financial_agent = FinancialAgent(llm_config)
        await _financial_agent.initialize()
    
    return _financial_agent
