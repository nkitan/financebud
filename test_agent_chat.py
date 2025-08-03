#!/usr/bin/env python3
"""
Test the complete agent chat flow to debug hallucination issue
"""

import asyncio
import sys
import os

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.agents.financial_agent import get_financial_agent, get_account_summary_tool

async def test_agent_chat():
    """Test the agent's complete chat flow."""
    print("🔧 Testing Agent Chat Flow...")
    
    try:
        # First test the tool directly
        print("\n1. Testing get_account_summary_tool directly...")
        direct_result = await get_account_summary_tool("")
        print("✅ Direct tool result:")
        print(direct_result[:300] + "..." if len(direct_result) > 300 else direct_result)
        
        # Now test through the agent
        print("\n2. Testing through agent chat...")
        agent = await get_financial_agent()
        
        # Test with a simple query that should trigger the account summary tool
        chat_result = await agent.chat("Show me my account summary", "test-session")
        print("✅ Agent chat result:")
        print(chat_result)
        
        # Check if the agent has sessions
        print(f"\n3. Agent session info:")
        print(f"   - Sessions: {list(agent.sessions.keys())}")
        if "test-session" in agent.sessions:
            session = agent.sessions["test-session"]
            print(f"   - Messages in session: {len(session)}")
            for i, msg in enumerate(session):
                print(f"   - Message {i}: role={msg.role}, content_length={len(msg.content) if msg.content else 0}")
        
        print("\n🎉 Test completed!")
        
    except Exception as e:
        print(f"❌ Error testing agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent_chat())
