#!/usr/bin/env python3
"""
Test script to verify MCP tools work correctly
"""

import asyncio
import sys
import os

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.agents.financial_agent import get_account_summary_tool, search_transactions_tool

async def test_mcp_tools():
    """Test the MCP tools functionality."""
    print("🔧 Testing MCP Tools...")
    
    try:
        # Test account summary
        print("\n1. Testing get_account_summary_tool...")
        summary_result = await get_account_summary_tool("")
        print("✅ Account summary result:")
        print(summary_result[:500] + "..." if len(summary_result) > 500 else summary_result)
        
        # Test transaction search
        print("\n2. Testing search_transactions_tool...")
        search_result = await search_transactions_tool("UPI")
        print("✅ Transaction search result:")
        print(search_result[:500] + "..." if len(search_result) > 500 else search_result)
        
        print("\n🎉 All MCP tools are working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing MCP tools: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_tools())
