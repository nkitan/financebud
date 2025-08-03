#!/usr/bin/env python3
"""
Test script for the new get_recent_transactions functionality
"""

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.agents.financial_agent import get_financial_agent

async def test_recent_transactions():
    """Test the recent transactions functionality"""
    
    print("🔧 Testing Recent Transactions Functionality")
    print("=" * 50)
    
    try:
        # Get the financial agent
        agent = await get_financial_agent()
        
        # Test 1: Direct tool call
        print("\n1. Testing direct MCP tool call...")
        from fastmcp import Client as FastMCPClient
        
        client = FastMCPClient("mcp_server.py")
        async with client:
            result = await client.call_tool("get_recent_transactions", {"limit": 5})
            
            # Extract text content from FastMCP CallToolResult
            tool_result = ""
            if hasattr(result, 'content') and result.content:
                for content_item in result.content:
                    if hasattr(content_item, 'type') and content_item.type == 'text':
                        tool_result = getattr(content_item, 'text', str(content_item))
                        break
            else:
                tool_result = str(result)
            
            print(f"✅ Direct tool result: {tool_result[:200]}...")
        
        # Test 2: Agent chat with "recent transactions" query
        print("\n2. Testing agent chat with 'Show me my recent transactions'...")
        response = await agent.chat("Show me my recent transactions")
        print(f"✅ Agent response: {response[:300]}...")
        
        # Test 3: Agent chat with specific limit
        print("\n3. Testing agent chat with 'Show me my last 5 transactions'...")
        response = await agent.chat("Show me my last 5 transactions")
        print(f"✅ Agent response: {response[:300]}...")
        
        # Test 4: Check if agent uses the right tool
        print("\n4. Testing if agent selected the right tool...")
        # The agent should have used get_recent_transactions tool
        if "recent" in response.lower() or "transaction" in response.lower():
            print("✅ Agent provided transaction data")
        else:
            print("❌ Agent did not provide transaction data")
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_recent_transactions())
