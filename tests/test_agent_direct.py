#!/usr/bin/env python3
"""
Financial Agent Direct Test
===========================

Test the financial agent directly to isolate any timeout issues.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from backend.agents.financial_agent import get_financial_agent
from backend.mcp.client import get_mcp_manager

async def test_financial_agent():
    """Test the financial agent directly."""
    print("🧪 Testing Financial Agent Direct...")
    
    try:
        # Initialize MCP manager first
        print("🔧 Initializing MCP manager...")
        mcp_manager = await get_mcp_manager()
        await mcp_manager.initialize_default_servers()
        print("✅ MCP manager initialized")
        
        # Initialize financial agent
        print("🤖 Initializing financial agent...")
        agent = await get_financial_agent()
        print("✅ Financial agent initialized")
        
        # Test simple message without tools
        print("💬 Testing simple message...")
        simple_response = await agent.process_message("Hello, just say hi back")
        print(f"✅ Simple response: {simple_response[:100]}...")
        
        # Test financial query that should use tools
        print("💰 Testing financial query...")
        financial_response = await agent.process_message("What is my current account balance?")
        print(f"✅ Financial response: {financial_response[:200]}...")
        
        # Test with a specific tool
        print("🔧 Testing specific tool call...")
        tool_response = await agent.call_tool("get_account_summary", "{}")
        print(f"✅ Tool response success: {tool_response.success}")
        if tool_response.success:
            print(f"   Result: {tool_response.result[:100]}...")
        else:
            print(f"   Error: {tool_response.error}")
        
        print("✅ All agent tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_financial_agent())
