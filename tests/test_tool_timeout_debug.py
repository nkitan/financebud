#!/usr/bin/env python3
"""
Tool Timeout Debugging Tests
============================

This test suite is designed to identify and fix the tool execution timeout issues
that are causing the LLM to hang for 300+ seconds even with fast models like Gemini Flash.
"""

import asyncio
import time
import logging
import sys
import os
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.financial_agent import FinancialAgent, get_financial_agent
from backend.mcp.client import MCPManager, get_mcp_manager
from backend.database.db import get_db_manager
from backend.agents.llm_providers import get_default_config
from backend.logging_config import setup_logging, get_logger_with_context

# Configure logging
setup_logging(log_level="DEBUG", log_dir="logs")
logger = get_logger_with_context(__name__)

class ToolTimeoutTester:
    """Test suite for diagnosing tool execution timeouts."""
    
    def __init__(self):
        self.mcp_manager = None
        self.agent = None
        self.db_manager = None
    
    async def setup(self):
        """Initialize all components."""
        logger.info("🔧 Setting up test environment...")
        
        # Initialize database
        self.db_manager = get_db_manager()
        logger.info("✅ Database manager initialized")
        
        # Initialize MCP manager
        self.mcp_manager = await get_mcp_manager()
        await self.mcp_manager.initialize_default_servers()
        logger.info("✅ MCP manager initialized")
        
        # Initialize agent
        self.agent = await get_financial_agent()
        logger.info("✅ Financial agent initialized")
    
    async def test_mcp_tool_list(self):
        """Test getting the list of available tools."""
        logger.info("🧪 Testing MCP tool list...")
        start_time = time.time()
        
        try:
            tools = await asyncio.wait_for(
                self.mcp_manager.get_available_tools(),
                timeout=10.0
            )
            execution_time = time.time() - start_time
            logger.info(f"✅ Got {len(tools)} tools in {execution_time:.2f}s")
            
            # List first few tools
            for i, tool in enumerate(tools[:3]):
                logger.info(f"   Tool {i+1}: {tool.get('name', 'unknown')}")
            
            return True
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"❌ Tool list timed out after {execution_time:.2f}s")
            return False
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Tool list failed after {execution_time:.2f}s: {e}")
            return False
    
    async def test_simple_tool_call(self):
        """Test a simple tool call that should be fast."""
        logger.info("🧪 Testing simple tool call...")
        start_time = time.time()
        
        try:
            # Try to call get_account_summary which should be fast
            result = await asyncio.wait_for(
                self.mcp_manager.call_tool("get_account_summary", {}),
                timeout=15.0
            )
            execution_time = time.time() - start_time
            logger.info(f"✅ Simple tool call completed in {execution_time:.2f}s")
            logger.info(f"   Result type: {type(result)}")
            logger.info(f"   Result length: {len(str(result))}")
            return True
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"❌ Simple tool call timed out after {execution_time:.2f}s")
            return False
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Simple tool call failed after {execution_time:.2f}s: {e}")
            return False
    
    async def test_database_queries(self):
        """Test direct database queries."""
        logger.info("🧪 Testing database queries...")
        start_time = time.time()
        
        try:
            # Test a simple count query
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    self.db_manager.execute_query,
                    "SELECT COUNT(*) as count FROM transactions"
                ),
                timeout=5.0
            )
            execution_time = time.time() - start_time
            logger.info(f"✅ Database query completed in {execution_time:.2f}s")
            logger.info(f"   Transaction count: {result.data[0]['count']}")
            return True
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"❌ Database query timed out after {execution_time:.2f}s")
            return False
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Database query failed after {execution_time:.2f}s: {e}")
            return False
    
    async def test_mcp_health(self):
        """Test MCP server health checks."""
        logger.info("🧪 Testing MCP health checks...")
        start_time = time.time()
        
        try:
            health = await asyncio.wait_for(
                self.mcp_manager.health_check(),
                timeout=10.0
            )
            execution_time = time.time() - start_time
            logger.info(f"✅ MCP health check completed in {execution_time:.2f}s")
            
            # Log server status
            servers = health.get("servers", {})
            for server_name, status in servers.items():
                logger.info(f"   Server {server_name}: {status}")
            
            return True
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"❌ MCP health check timed out after {execution_time:.2f}s")
            return False
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ MCP health check failed after {execution_time:.2f}s: {e}")
            return False
    
    async def test_agent_without_llm(self):
        """Test agent tool processing without LLM calls."""
        logger.info("🧪 Testing agent tool processing (no LLM)...")
        start_time = time.time()
        
        try:
            # Get tools available to agent
            tools = await asyncio.wait_for(
                self.agent.get_available_tools(),
                timeout=10.0
            )
            execution_time = time.time() - start_time
            logger.info(f"✅ Agent tools retrieved in {execution_time:.2f}s")
            logger.info(f"   Available tools: {len(tools)}")
            return True
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"❌ Agent tools retrieval timed out after {execution_time:.2f}s")
            return False
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Agent tools retrieval failed after {execution_time:.2f}s: {e}")
            return False
    
    async def test_conversation_history_optimization(self):
        """Test conversation history management."""
        logger.info("🧪 Testing conversation history optimization...")
        start_time = time.time()
        
        try:
            # Add many messages to test history optimization
            for i in range(15):
                self.agent.conversation_history.append({
                    "role": "user",
                    "content": f"Test message {i}"
                })
                self.agent.conversation_history.append({
                    "role": "assistant", 
                    "content": f"Test response {i}"
                })
            
            # Test optimization
            original_length = len(self.agent.conversation_history)
            self.agent._optimize_conversation_history()
            optimized_length = len(self.agent.conversation_history)
            
            execution_time = time.time() - start_time
            logger.info(f"✅ History optimization completed in {execution_time:.2f}s")
            logger.info(f"   Original length: {original_length}")
            logger.info(f"   Optimized length: {optimized_length}")
            
            return True
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ History optimization failed after {execution_time:.2f}s: {e}")
            return False
    
    async def test_parallel_tool_calls(self):
        """Test parallel tool execution."""
        logger.info("🧪 Testing parallel tool calls...")
        start_time = time.time()
        
        try:
            # Create multiple tool call tasks
            tasks = []
            for i in range(3):
                task = asyncio.create_task(
                    self.mcp_manager.call_tool("get_account_summary", {})
                )
                tasks.append(task)
            
            # Wait for all with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=20.0
            )
            
            execution_time = time.time() - start_time
            logger.info(f"✅ Parallel tool calls completed in {execution_time:.2f}s")
            
            # Check results
            successful = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(f"   Successful calls: {successful}/{len(results)}")
            
            return True
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"❌ Parallel tool calls timed out after {execution_time:.2f}s")
            return False
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Parallel tool calls failed after {execution_time:.2f}s: {e}")
            return False
    
    async def test_tool_result_summarization(self):
        """Test tool result summarization."""
        logger.info("🧪 Testing tool result summarization...")
        start_time = time.time()
        
        try:
            # Create a large fake tool result
            large_result = "x" * 10000  # 10KB of data
            
            # Test summarization
            summarized = self.agent._summarize_tool_result(
                tool_name="test_tool",
                result=large_result
            )
            
            execution_time = time.time() - start_time
            logger.info(f"✅ Tool result summarization completed in {execution_time:.2f}s")
            logger.info(f"   Original length: {len(large_result)}")
            logger.info(f"   Summarized length: {len(summarized)}")
            
            return True
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Tool result summarization failed after {execution_time:.2f}s: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all diagnostic tests."""
        logger.info("🚀 Starting comprehensive tool timeout diagnosis...")
        
        tests = [
            ("Database Queries", self.test_database_queries),
            ("MCP Health Check", self.test_mcp_health),
            ("MCP Tool List", self.test_mcp_tool_list),
            ("Simple Tool Call", self.test_simple_tool_call),
            ("Agent Tools (No LLM)", self.test_agent_without_llm),
            ("Conversation History", self.test_conversation_history_optimization),
            ("Tool Result Summarization", self.test_tool_result_summarization),
            ("Parallel Tool Calls", self.test_parallel_tool_calls),
        ]
        
        results = {}
        total_start = time.time()
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                result = await test_func()
                results[test_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"{test_name}: {status}")
            except Exception as e:
                results[test_name] = False
                logger.error(f"{test_name}: ❌ ERROR - {e}")
        
        total_time = time.time() - total_start
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total execution time: {total_time:.2f}s")
        
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        logger.info(f"Tests passed: {passed}/{total}")
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        # Recommendations
        logger.info(f"\n{'='*60}")
        logger.info("RECOMMENDATIONS")
        logger.info(f"{'='*60}")
        
        failed_tests = [name for name, result in results.items() if not result]
        
        if not failed_tests:
            logger.info("✅ All tests passed! The timeout issue may be in LLM provider interaction.")
        else:
            logger.info("❌ Failed tests indicate bottlenecks:")
            for test in failed_tests:
                if "Database" in test:
                    logger.info("  - Database queries are slow - check SQLite connection")
                elif "MCP" in test:
                    logger.info("  - MCP server communication issues - check server processes")
                elif "Tool" in test:
                    logger.info("  - Tool execution problems - check MCP client implementation")
                elif "Parallel" in test:
                    logger.info("  - Concurrency issues - check async/await patterns")
        
        return results
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("🧹 Cleaning up test environment...")
        
        if self.mcp_manager:
            await self.mcp_manager.shutdown()
        
        if self.db_manager:
            self.db_manager.close()
        
        logger.info("✅ Cleanup completed")

async def main():
    """Main test runner."""
    tester = ToolTimeoutTester()
    
    try:
        await tester.setup()
        results = await tester.run_all_tests()
        
        # Exit with error code if any tests failed
        failed_count = sum(1 for r in results.values() if not r)
        if failed_count > 0:
            logger.error(f"❌ {failed_count} tests failed - investigation needed")
            sys.exit(1)
        else:
            logger.info("✅ All tests passed!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Test runner failed: {e}")
        sys.exit(1)
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
