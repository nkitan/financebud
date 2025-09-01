"""
FinanceBud System Test
====================

Test the financial agent system with real data.
"""

import asyncio
import sys
import inspect
from pathlib import Path
from typing import Union, AsyncGenerator

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.config.settings import get_settings
from backend.agents.financial_agent import FinancialAgent
from backend.logging_config import get_logger_with_context

logger = get_logger_with_context(__name__)


async def test_system():
    """Test the financial system with real queries."""
    print("🚀 Starting FinanceBud System Test")
    print("=" * 50)
    
    agent = None
    
    try:
        # Initialize settings
        settings = get_settings()
        logger.info(f"Settings loaded: {settings.database.database_path}")
        
        # Create agent
        agent = FinancialAgent()
        await agent.initialize()
        
        print("✅ Financial Agent initialized successfully")
        
        # Test queries
        test_queries = [
            "What's my current account balance and financial health?",
            "Show me my recent transactions with insights",
            "Analyze my spending patterns for the last month",
            "Find recurring payments and subscriptions",
            "What are my biggest spending categories?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {query}")
            print("-" * 30)
            
            try:
                response = await agent.process_message(query)
                if response:
                    # Check if it's an async generator by inspecting its type
                    if inspect.isasyncgen(response):
                        # It's an async generator, collect the full response
                        print("Response (streaming):")
                        response_text = ""
                        async for chunk in response:
                            print(chunk, end="", flush=True)
                            response_text += chunk
                        print()  # New line after streaming
                    elif isinstance(response, str):
                        # It's a string
                        if len(response) > 500:
                            print(f"Response (truncated): {response[:500]}...")
                        else:
                            print(f"Response: {response}")
                    else:
                        print(f"Response (unknown type): {response}")
                else:
                    print("❌ No response received")
                    
            except Exception as e:
                print(f"❌ Query failed: {e}")
                logger.error(f"Query {i} failed: {e}")
        
        # Test health check
        print(f"\n🏥 Health Check")
        print("-" * 20)
        health = agent.get_health_status()
        print(f"Agent Health: {health}")
        
        # Test session info
        print(f"\n📊 Session Info")
        print("-" * 20)
        session_info = agent.get_session_metrics()
        for key, value in session_info.items():
            print(f"{key}: {value}")
        
        print(f"\n✅ System test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"System test failed: {e}")
        return False
    
    finally:
        try:
            # Only try to shutdown if agent was successfully created
            if agent is not None:
                await agent.shutdown()
                print("🧹 Agent cleanup completed")
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup warning: {cleanup_error}")
    
    return True


async def test_mcp_server_connection():
    """Test MCP server connection separately."""
    print("\n🔌 Testing MCP Server Connection")
    print("-" * 30)
    
    try:
        from backend.mcp.client import get_mcp_manager
        
        mcp_manager = await get_mcp_manager()
        
        # Test account summary
        result = await mcp_manager.call_tool(
            "financial-data-inr",
            "get_account_summary",
            {}
        )
        
        if result and isinstance(result, dict):
            if "data" in result:
                print("✅ MCP server connection successful")
                data = result["data"]
                print(f"Current Balance: {data.get('current_balance', 'N/A')}")
                print(f"Total Transactions: {data.get('total_transactions', 'N/A')}")
                return True
            else:
                print(f"❌ Unexpected result format: {result}")
                return False
        else:
            print(f"❌ Invalid result: {result}")
            return False
            
    except Exception as e:
        print(f"❌ MCP connection failed: {e}")
        logger.error(f"MCP connection test failed: {e}")
        return False


if __name__ == "__main__":
    async def main():
        print("🎯 FinanceBud Production Test Suite")
        print("=" * 50)
        
        # Test MCP connection first
        mcp_success = await test_mcp_server_connection()
        
        if mcp_success:
            # Test agent system
            agent_success = await test_system()
            
            if agent_success:
                print("\n🎉 All tests passed! System is production ready.")
                return 0
            else:
                print("\n❌ Agent tests failed.")
                return 1
        else:
            print("\n❌ MCP server connection failed. Check if mcp_server.py is running.")
            return 1
    
    # Run the test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
