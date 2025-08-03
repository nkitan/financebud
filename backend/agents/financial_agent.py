"""
Generic Financial Analysis Agent
===============================

This module implements a financial analysis agent that can work with any
OpenAI-compatible LLM provider (Ollama, OpenAI, Gemini, OpenRouter, etc.).

Key improvements:
- Provider-agnostic design using OpenAI-compatible APIs
- Easy switching between different LLM providers via configuration
- Standard tool calling implementation
- Environment-based configuration
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# Import our generic LLM providers
from .llm_providers import (
    LLMConfig, LLMProvider, ProviderType,
    create_provider, get_default_config
)

# FastMCP client for proper MCP protocol handling
from fastmcp import Client as FastMCPClient

logger = logging.getLogger(__name__)

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
    """Represents a financial analysis tool."""
    
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func
    
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

# Define financial analysis tools (same as before)
async def get_account_summary_tool(tool_input: str) -> str:
    """Get account summary with current balance and transaction overview."""
    try:
        client = FastMCPClient("mcp_server.py")
        async with client:
            result = await client.call_tool("get_account_summary", {})
            # Extract text content from FastMCP CallToolResult
            if hasattr(result, 'content') and result.content:
                for content_item in result.content:
                    # Check if it's a text content item
                    if hasattr(content_item, 'type') and content_item.type == 'text':
                        return getattr(content_item, 'text', str(content_item))
            return str(result)
    except Exception as e:
        logger.error(f"Error getting account summary: {e}")
        return f"Error getting account summary: {str(e)}"

async def search_transactions_tool(tool_input: str) -> str:
    """Search for transactions matching a pattern."""
    try:
        client = FastMCPClient("mcp_server.py")
        params = {"description_pattern": tool_input.strip()}
        async with client:
            result = await client.call_tool("search_transactions", params)
            # Extract text content from FastMCP CallToolResult
            if hasattr(result, 'content') and result.content:
                for content_item in result.content:
                    if hasattr(content_item, 'type') and content_item.type == 'text':
                        return getattr(content_item, 'text', str(content_item))
            return str(result)
    except Exception as e:
        logger.error(f"Error searching transactions: {e}")
        return f"Error searching transactions: {str(e)}"

async def get_transactions_by_date_range_tool(tool_input: str) -> str:
    """Get transactions within a specific date range."""
    try:
        client = FastMCPClient("mcp_server.py")
        
        # Parse date range
        params = {}
        if "start_date:" in tool_input and "end_date:" in tool_input:
            for part in tool_input.split():
                if part.startswith("start_date:"):
                    params["start_date"] = part.split(":", 1)[1]
                elif part.startswith("end_date:"):
                    params["end_date"] = part.split(":", 1)[1]
        elif " to " in tool_input:
            dates = tool_input.split(" to ")
            if len(dates) == 2:
                params["start_date"] = dates[0].strip()
                params["end_date"] = dates[1].strip()
        
        if "start_date" not in params or "end_date" not in params:
            return "Error: Please provide both start_date and end_date in format 'start_date:YYYY-MM-DD end_date:YYYY-MM-DD' or 'YYYY-MM-DD to YYYY-MM-DD'"
            
        async with client:
            result = await client.call_tool("get_transactions_by_date_range", params)
            # Extract text content from FastMCP CallToolResult
            if hasattr(result, 'content') and result.content:
                for content_item in result.content:
                    if hasattr(content_item, 'type') and content_item.type == 'text':
                        return getattr(content_item, 'text', str(content_item))
            return str(result)
    except Exception as e:
        logger.error(f"Error getting transactions by date range: {e}")
        return f"Error getting transactions by date range: {str(e)}"

async def get_monthly_summary_tool(tool_input: str) -> str:
    """Get monthly summary of spending and transactions."""
    try:
        client = FastMCPClient("mcp_server.py")
        
        # Parse year and month
        params = {}
        if "year:" in tool_input and "month:" in tool_input:
            for part in tool_input.split():
                if part.startswith("year:"):
                    params["year"] = int(part.split(":", 1)[1])
                elif part.startswith("month:"):
                    params["month"] = int(part.split(":", 1)[1])
        elif "-" in tool_input and len(tool_input.split("-")) == 2:
            year, month = tool_input.split("-")
            params["year"] = int(year)
            params["month"] = int(month)
        else:
            # Default to current month
            now = datetime.now()
            params["year"] = now.year
            params["month"] = now.month
            
        async with client:
            result = await client.call_tool("get_monthly_summary", params)
            # Extract text content from FastMCP CallToolResult
            if hasattr(result, 'content') and result.content:
                for content_item in result.content:
                    if hasattr(content_item, 'type') and content_item.type == 'text':
                        return getattr(content_item, 'text', str(content_item))
            return str(result)
    except Exception as e:
        return f"Error getting monthly summary: {str(e)}"

async def analyze_spending_trends_tool(tool_input: str) -> str:
    """Analyze spending trends over multiple months."""
    try:
        client = FastMCPClient("mcp_server.py")
        
        # Parse months parameter
        params = {"months_back": 6}  # Default - parameter name should match MCP server
        if "months:" in tool_input:
            try:
                months_str = tool_input.split("months:")[1].split()[0]
                params["months_back"] = int(months_str)
            except (IndexError, ValueError):
                pass
                
        async with client:
            result = await client.call_tool("analyze_spending_trends", params)
            # Extract text content from FastMCP CallToolResult
            if hasattr(result, 'content') and result.content:
                for content_item in result.content:
                    if hasattr(content_item, 'type') and content_item.type == 'text':
                        return getattr(content_item, 'text', str(content_item))
            return str(result)
    except Exception as e:
        return f"Error analyzing spending trends: {str(e)}"

async def get_upi_transaction_analysis_tool(tool_input: str) -> str:
    """Analyze UPI transactions and patterns."""
    try:
        client = FastMCPClient("mcp_server.py")
        async with client:
            result = await client.call_tool("get_upi_transaction_analysis", {})
            # Extract text content from FastMCP CallToolResult
            if hasattr(result, 'content') and result.content:
                for content_item in result.content:
                    if hasattr(content_item, 'type') and content_item.type == 'text':
                        return getattr(content_item, 'text', str(content_item))
            return str(result)
    except Exception as e:
        return f"Error analyzing UPI transactions: {str(e)}"

class GenericFinancialAgent:
    """Financial analysis agent using generic LLM providers."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_default_config()
        self.provider = create_provider(self.config)
        
        # Initialize financial tools
        self.tools = [
            FinancialTool("get_account_summary", "Get account summary with current balance and transaction overview", get_account_summary_tool),
            FinancialTool("search_transactions", "Search for transactions matching a pattern or description", search_transactions_tool),
            FinancialTool("get_transactions_by_date_range", "Get transactions within a specific date range", get_transactions_by_date_range_tool),
            FinancialTool("get_monthly_summary", "Get monthly summary of spending and transactions", get_monthly_summary_tool),
            FinancialTool("analyze_spending_trends", "Analyze spending trends over multiple months", analyze_spending_trends_tool),
            FinancialTool("get_upi_transaction_analysis", "Analyze UPI transactions and digital payment patterns", get_upi_transaction_analysis_tool),
        ]
        
        self.tool_map = {tool.name: tool.func for tool in self.tools}
        self.sessions: Dict[str, List[ConversationMessage]] = {}
        
                # System prompt for financial analysis
        self.system_prompt = f"""You are an expert financial analysis assistant with access to real banking and transaction data from Indian bank accounts.

Current LLM Provider: {self.config.provider.value} ({self.config.model})

Your capabilities include:
- Account balance and transaction summaries with amounts in Indian Rupees (₹)
- Transaction searching and filtering by various criteria
- Monthly and yearly spending analysis and trends
- UPI transaction analysis (India's digital payment system)
- Recurring payment identification and analysis
- Custom financial queries and detailed analytics

IMPORTANT: You have access to specialized financial tools. ALWAYS use these tools to get real data. Never attempt to write Python code or simulate data.

Guidelines for responses:
- Always format monetary amounts in Indian Rupees (₹) with proper comma separation
- Provide actionable insights and practical recommendations
- Use the available tools to get real data rather than making assumptions
- Be specific about Indian banking patterns (UPI, NEFT, IMPS, etc.)
- Suggest concrete steps for financial improvement

Tool Usage:
- For account balances/summaries: use get_account_summary tool
- For transaction searches: use search_transactions tool  
- For monthly analysis: use get_monthly_summary tool
- For trends over time: use analyze_spending_trends tool
- For UPI analysis: use get_upi_transaction_analysis tool
- For date ranges: use get_transactions_by_date_range tool

Always call the appropriate tool first to get current data, then provide analysis and insights based on the real data."""

    async def test_connection(self) -> bool:
        """Test if the LLM provider is available."""
        return await self.provider.test_connection()

    async def chat(self, message: str, session_id: str = "default") -> str:
        """
        Main chat interface using generic LLM providers.
        
        This implements the standard OpenAI tool calling flow:
        1. Send message with tools to LLM
        2. Parse tool_calls from response
        3. Execute tools and add results to conversation
        4. Send back to LLM for final response
        """
        try:
            # Initialize session if not exists
            if session_id not in self.sessions:
                self.sessions[session_id] = [
                    ConversationMessage(role="system", content=self.system_prompt)
                ]
            
            # Add user message to conversation
            self.sessions[session_id].append(
                ConversationMessage(role="user", content=message)
            )
            
            # Step 1: Send to LLM with tools
            response = await self._call_llm_with_tools(session_id)
            
            # Step 2: Check if LLM wants to use tools
            if self._has_tool_calls(response):
                # Step 3: Execute tools and add results
                await self._execute_tools(response, session_id)
                
                # Step 4: Get final response without tools
                final_response = await self._call_llm_without_tools(session_id)
                content = self._extract_content(final_response)
                
                # Add final response to conversation
                self.sessions[session_id].append(
                    ConversationMessage(role="assistant", content=content)
                )
                return content
            else:
                # Direct response without tools
                content = self._extract_content(response)
                self.sessions[session_id].append(
                    ConversationMessage(role="assistant", content=content)
                )
                return content
                
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"I encountered an error while processing your request: {str(e)}"

    async def _call_llm_with_tools(self, session_id: str) -> Dict[str, Any]:
        """Call LLM with tools available."""
        messages = self._format_messages_for_api(session_id)
        
        # Get the user's last message to select relevant tools
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"].lower()
                break
        
        # Select relevant tools based on user query to avoid overwhelming Ollama
        relevant_tools = self._select_relevant_tools(user_message)
        
        return await self.provider.chat_completion(messages, relevant_tools)
    
    def _select_relevant_tools(self, user_query: str) -> List[Dict[str, Any]]:
        """Select relevant tools based on user query to reduce token overhead."""
        all_tools = [tool.to_openai_format() for tool in self.tools]
        
        # Keywords to tool mapping
        tool_keywords = {
            "get_account_summary": ["balance", "summary", "account", "overview"],
            "search_transactions": ["search", "find", "transaction", "payment"],
            "get_transactions_by_date_range": ["date", "range", "between", "from", "to", "period"],
            "get_monthly_summary": ["month", "monthly", "spending"],
            "analyze_spending_trends": ["trend", "analysis", "pattern", "over time"],
            "get_upi_transaction_analysis": ["upi", "digital", "payment"]
        }
        
        # If query is short or general, return only the most common tools
        if len(user_query.split()) <= 3:
            primary_tools = ["get_account_summary", "search_transactions"]
            return [tool for tool in all_tools if tool["function"]["name"] in primary_tools]
        
        # Select tools based on keywords
        selected_tool_names = set()
        for tool_name, keywords in tool_keywords.items():
            if any(keyword in user_query for keyword in keywords):
                selected_tool_names.add(tool_name)
        
        # If no specific tools matched, return the most common ones
        if not selected_tool_names:
            selected_tool_names = {"get_account_summary", "search_transactions"}
        
        # Limit to maximum 3 tools to avoid overwhelming Ollama
        selected_tool_names = list(selected_tool_names)[:3]
        
        return [tool for tool in all_tools if tool["function"]["name"] in selected_tool_names]

    async def _call_llm_without_tools(self, session_id: str) -> Dict[str, Any]:
        """Call LLM without tools for final response."""
        messages = self._format_messages_for_api(session_id)
        
        return await self.provider.chat_completion(messages)

    def _format_messages_for_api(self, session_id: str) -> List[Dict[str, Any]]:
        """Format conversation messages for LLM API."""
        messages = []
        
        # Always include system message
        system_msg = None
        user_assistant_pairs = []
        
        for msg in self.sessions[session_id]:
            if msg.role == "system":
                system_msg = {"role": "system", "content": msg.content}
            elif msg.role == "user":
                user_assistant_pairs.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant" and not msg.tool_calls:
                # Only include assistant messages that are final responses, not tool calls
                if user_assistant_pairs and len(user_assistant_pairs) > 0:
                    user_assistant_pairs.append({"role": "assistant", "content": msg.content})
            # Skip tool call messages and tool response messages to avoid confusing the model
        
        # Build clean message history
        if system_msg:
            messages.append(system_msg)
        
        # Only include the last few user-assistant pairs to keep context manageable
        # This prevents the model from getting confused by complex tool calling history
        recent_pairs = user_assistant_pairs[-6:]  # Last 3 user-assistant pairs
        messages.extend(recent_pairs)
        
        return messages

    def _has_tool_calls(self, response: Dict[str, Any]) -> bool:
        """Check if LLM response contains tool calls."""
        try:
            message = response["choices"][0]["message"]
            return "tool_calls" in message and message["tool_calls"]
        except (KeyError, IndexError):
            return False

    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract content from LLM response."""
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            return "Sorry, I couldn't process that request properly."

    async def _execute_tools(self, response: Dict[str, Any], session_id: str):
        """Execute tools requested by LLM and add results to conversation."""
        try:
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls", [])
            
            # Add the assistant's tool call message to conversation
            self.sessions[session_id].append(
                ConversationMessage(
                    role="assistant",
                    content=message.get("content", ""),
                    tool_calls=tool_calls
                )
            )
            
            # Execute each tool call
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                tool_call_id = tool_call["id"]
                
                # Execute the tool
                if function_name in self.tool_map:
                    tool_input = arguments.get("tool_input", "")
                    result = await self.tool_map[function_name](tool_input)
                else:
                    result = f"Error: Unknown tool '{function_name}'"
                
                # Add tool result to conversation
                self.sessions[session_id].append(
                    ConversationMessage(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call_id
                    )
                )
                
        except Exception as e:
            logger.error(f"Error executing tools: {e}")
            # Add error message to conversation
            self.sessions[session_id].append(
                ConversationMessage(
                    role="tool",
                    content=f"Error executing tool: {str(e)}"
                )
            )

    async def get_health(self) -> Dict[str, Any]:
        """Get agent health status."""
        health = {
            "status": "healthy",
            "provider": self.config.provider.value,
            "model": self.config.model,
            "provider_available": await self.test_connection(),
            "tools_available": len(self.tools),
            "active_sessions": len(self.sessions)
        }
        
        # Test MCP server
        try:
            client = FastMCPClient("mcp_server.py")
            async with client:
                await client.call_tool("get_account_summary", {})
            health["mcp_server_available"] = True
        except Exception:
            health["mcp_server_available"] = False
        
        return health

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a session."""
        if session_id not in self.sessions:
            return []
        
        history = []
        for msg in self.sessions[session_id]:
            if msg.role != "system":  # Don't include system message in history
                history.append({
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": time.time()
                })
        return history

    def switch_provider(self, config: LLMConfig):
        """Switch to a different LLM provider."""
        self.config = config
        self.provider = create_provider(config)
        
        # Update system prompt to reflect new provider
        provider_info = f"Current LLM Provider: {self.config.provider.value} ({self.config.model})"
        self.system_prompt = self.system_prompt.replace(
            self.system_prompt.split('\n')[6],  # Line with provider info
            provider_info
        )
        
        # Update existing sessions with new system prompt
        for session_id in self.sessions:
            if self.sessions[session_id] and self.sessions[session_id][0].role == "system":
                self.sessions[session_id][0].content = self.system_prompt

# Global agent instance
_agent = None

async def get_financial_agent(config: Optional[LLMConfig] = None) -> GenericFinancialAgent:
    """Get or create the global financial agent instance."""
    global _agent
    if _agent is None or (config and config != _agent.config):
        _agent = GenericFinancialAgent(config)
    return _agent
