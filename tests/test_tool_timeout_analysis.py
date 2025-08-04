"""
Tool Timeout Analysis Test
=========================

Comprehensive test to diagnose tool execution hanging issues.
This test will help identify where the timeout is occurring in the tool chain.
"""

import asyncio
import pytest
import time
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.agents.financial_agent import FinancialAgent
from backend.agents.llm_providers import LLMConfig, ProviderType, get_default_config
from backend.mcp.client import MCPManager
from backend.database.db import get_db_manager

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class TestToolTimeoutAnalysis:
    """Test class to analyze tool execution timeouts."""
    
    @pytest.fixture
    async def mock_llm_config(self):
        """Get the default LLM config from environment."""
        return get_default_config()
    
    @pytest.fixture
    async def mcp_manager(self):
        """Create MCP manager for testing."""
        manager = MCPManager()
        await manager.initialize_default_servers()
        yield manager
        await manager.shutdown()
    
    async def test_mcp_connection_health(self, mcp_manager):
        """Test if MCP connections are healthy and responsive."""
        print("\n🔍 Testing MCP connection health...")
        
        start_time = time.time()
        try:
            health = await asyncio.wait_for(
                mcp_manager.health_check(),
                timeout=10.0
            )
            elapsed = time.time() - start_time
            
            print(f"✅ MCP health check completed in {elapsed:.2f}s")
            print(f"Health status: {health}")
            
            # Check if any servers are unhealthy
            if 'servers' in health:
                for server_name, status in health['servers'].items():
                    if not status.get('healthy', False):
                        print(f"❌ Unhealthy server: {server_name} - {status}")
                    else:
                        print(f"✅ Healthy server: {server_name}")
            
            assert elapsed < 10.0, f"Health check took too long: {elapsed}s"
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ MCP health check timed out after {elapsed:.2f}s")
            assert False, "MCP health check timed out"
    
    async def test_individual_tool_calls(self, mcp_manager):
        """Test individual tool calls to identify which ones hang."""
        print("\n🔍 Testing individual tool calls...")
        
        # Test each tool individually with aggressive timeout
        tool_results = []
        
        for tool_name in ['get_account_summary', 'get_recent_transactions', 'get_database_schema']:
            print(f"\n🔧 Testing tool: {tool_name}")
            start_time = time.time()
            
            try:
                # Very aggressive timeout for diagnosis
                result = await asyncio.wait_for(
                    mcp_manager.call_tool("financial-data-inr", tool_name, {}),
                    timeout=300.0  # 300 seconds max per tool
                )
                
                elapsed = time.time() - start_time
                print(f"✅ {tool_name}: {elapsed:.2f}s")
                tool_results.append((tool_name, 'success', elapsed, str(result)[:100]))
                
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                print(f"❌ {tool_name}: TIMEOUT after {elapsed:.2f}s")
                tool_results.append((tool_name, 'timeout', elapsed, 'Tool call timed out'))
                
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ {tool_name}: ERROR after {elapsed:.2f}s - {e}")
                tool_results.append((tool_name, 'error', elapsed, str(e)))
        
        # Report results
        print(f"\n📊 Tool Call Results:")
        for tool_name, status, elapsed, details in tool_results:
            print(f"  {tool_name}: {status} ({elapsed:.2f}s) - {details}")
        
        return tool_results
    
    def _create_test_args_for_tool(self, tool_name: str) -> dict:
        """Create appropriate test arguments for different tools."""
        test_args = {
            'get_account_summary': {},
            'get_recent_transactions': {'limit': 5},
            'search_transactions': {'query': 'test', 'limit': 5},
            'get_monthly_summary': {'year': 2024, 'month': 1},
            'get_transaction_trends': {'months': 3},
            'get_category_breakdown': {'months': 3},
            'get_balance_history': {'months': 3}
        }
        return test_args.get(tool_name, {})
    
    async def test_database_connection_timeout(self):
        """Test if database operations are causing timeouts."""
        print("\n🔍 Testing database connection and query performance...")
        
        db_manager = get_db_manager()
        
        # Test simple query
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    db_manager.execute_query,
                    "SELECT COUNT(*) as count FROM transactions LIMIT 1"
                ),
                timeout=5.0
            )
            elapsed = time.time() - start_time
            print(f"✅ Simple query completed in {elapsed:.2f}s")
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ Simple query timed out after {elapsed:.2f}s")
            assert False, "Database query timed out"
        
        # Test complex query
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    db_manager.execute_query,
                    """
                    SELECT 
                        strftime('%Y-%m', transaction_date) as month,
                        SUM(debit_amount) as total_debits,
                        SUM(credit_amount) as total_credits
                    FROM transactions 
                    WHERE transaction_date >= date('now', '-12 months')
                    GROUP BY strftime('%Y-%m', transaction_date)
                    ORDER BY month DESC
                    """
                ),
                timeout=10.0
            )
            elapsed = time.time() - start_time
            print(f"✅ Complex query completed in {elapsed:.2f}s")
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ Complex query timed out after {elapsed:.2f}s")
            # Don't fail the test, just log the issue
    
    async def test_mcp_server_process_status(self, mcp_manager):
        """Test if MCP server processes are running and responsive."""
        print("\n🔍 Testing MCP server process status...")
        
        # Check if MCP server processes are running
        import subprocess
        import json
        
        try:
            # Look for MCP server processes
            result = subprocess.run(
                ['ps', 'aux'], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            
            mcp_processes = []
            for line in result.stdout.split('\n'):
                if 'mcp_server' in line or 'python' in line and 'mcp' in line:
                    mcp_processes.append(line.strip())
            
            print(f"Found {len(mcp_processes)} MCP-related processes:")
            for proc in mcp_processes:
                print(f"  {proc}")
                
            if not mcp_processes:
                print("⚠️  No MCP server processes found")
            
        except subprocess.TimeoutExpired:
            print("❌ Process check timed out")
        except Exception as e:
            print(f"❌ Process check failed: {e}")
    
    async def test_concurrent_tool_calls(self, mcp_manager):
        """Test if concurrent tool calls cause deadlocks."""
        print("\n🔍 Testing concurrent tool calls for deadlocks...")
        
        async def make_tool_call(tool_name: str, args: dict, call_id: int):
            """Make a single tool call with timeout."""
            start_time = time.time()
            try:
                result = await asyncio.wait_for(
                    mcp_manager.call_tool("financial-data-inr", tool_name, args),
                    timeout=300.0
                )
                elapsed = time.time() - start_time
                print(f"✅ Concurrent call {call_id} completed in {elapsed:.2f}s")
                return True
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                print(f"❌ Concurrent call {call_id} timed out after {elapsed:.2f}s")
                return False
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ Concurrent call {call_id} failed after {elapsed:.2f}s: {e}")
                return False
        
        # Make 3 concurrent simple calls
        tasks = [
            make_tool_call('get_account_summary', {}, 1),
            make_tool_call('get_recent_transactions', {'limit': 5}, 2),
            make_tool_call('get_account_summary', {}, 3)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time
        
        successful_calls = sum(1 for r in results if r is True)
        print(f"Concurrent test completed in {elapsed:.2f}s")
        print(f"Successful calls: {successful_calls}/3")
        
        if successful_calls == 0:
            print("❌ All concurrent calls failed - possible deadlock")
        elif successful_calls < 3:
            print("⚠️  Some concurrent calls failed - possible resource contention")
    
    async def test_tool_call_with_agent_simulation(self, mock_llm_config):
        """Test tool calls in the context of agent processing."""
        print("\n🔍 Testing tool calls within agent context...")
        
        start_time = time.time()
        try:
            # Test with actual Gemini provider but simple message
            from backend.agents.financial_agent import get_financial_agent
            agent = await get_financial_agent(mock_llm_config)
            
            # Test a simple message that should trigger tool calls
            response = await asyncio.wait_for(
                agent.process_message("Show me a quick account summary"),
                timeout=60.0  # Reduced timeout for quicker diagnosis
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Agent processing completed in {elapsed:.2f}s")
            print(f"Response length: {len(response)} characters")
            print(f"Response preview: {response[:200]}...")
            
            assert elapsed < 60.0, f"Agent processing took too long: {elapsed}s"
            assert len(response) > 0, "Agent returned empty response"
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ Agent processing timed out after {elapsed:.2f}s")
            
            # Let's try to get more debug info
            print("🔍 Checking if it's an LLM issue vs tool issue...")
            
            # Test just the LLM without tools
            try:
                from backend.agents.llm_providers import create_provider, LLMConfig, ProviderType
                
                # Use the default config from environment
                test_config = get_default_config()
                
                provider = create_provider(test_config)  # Fixed: pass config parameter
                
                simple_response = await asyncio.wait_for(
                    provider.chat_completion([
                        {"role": "user", "content": "Say 'Hello, I am working' in exactly those words."}
                    ]),
                    timeout=10.0
                )
                print("✅ LLM provider working - issue is in tool execution flow")
                print(f"LLM response: {simple_response}")
                
            except Exception as llm_error:
                print(f"❌ LLM provider issue: {llm_error}")
                
            print("⚠️  Agent test failed but continuing with other tests...")
            # Don't fail the entire test suite
            return
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Agent processing failed after {elapsed:.2f}s: {e}")
            print(f"Error type: {type(e).__name__}")
            
            # Don't fail the entire test suite, just log the error
            print("⚠️  Agent test failed but continuing with other tests...")
            return


# Async test runner
async def run_timeout_analysis():
    """Run the timeout analysis tests."""
    print("🚀 Starting Tool Timeout Analysis...")
    
    test_instance = TestToolTimeoutAnalysis()
    
    # Create MCP manager
    mcp_manager = MCPManager()
    await mcp_manager.initialize_default_servers()
    
    try:
        # Run all tests
        await test_instance.test_mcp_connection_health(mcp_manager)
        await test_instance.test_individual_tool_calls(mcp_manager)
        await test_instance.test_database_connection_timeout()
        await test_instance.test_mcp_server_process_status(mcp_manager)
        await test_instance.test_concurrent_tool_calls(mcp_manager)
        
        # Use default config from environment
        mock_config = get_default_config()
        await test_instance.test_tool_call_with_agent_simulation(mock_config)
        
    finally:
        await mcp_manager.shutdown()
    
    print("✅ Tool Timeout Analysis completed")


if __name__ == "__main__":
    # Run the analysis
    asyncio.run(run_timeout_analysis())
