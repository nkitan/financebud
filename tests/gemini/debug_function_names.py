#!/usr/bin/env python3
"""
Debug Gemini Function Name Issue
===============================

This test will trace exactly what's happening with function names in tool responses.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.financial_agent import FinancialAgent, ConversationMessage
from backend.agents.llm_providers import LLMConfig, ProviderType
from dotenv import load_dotenv

load_dotenv()

async def debug_function_names():
    """Debug exactly what's happening with function names."""
    print("🔍 Debugging Gemini function name issue...")
    
    config = LLMConfig(
        provider=ProviderType.GEMINI,
        base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        api_key=os.getenv("GEMINI_API_KEY", "dummy"),
        max_tokens=500,
        timeout=60
    )
    
    agent = FinancialAgent(config)
    await agent.initialize()
    
    print("✅ Agent initialized")
    
    # Test one query that was failing and trace the conversation
    test_query = "What is my current balance?"
    
    print(f"\n💬 Testing: '{test_query}'")
    
    # Add the user message manually to trace
    agent.conversation_history.append(
        ConversationMessage(role="user", content=test_query)
    )
    
    # Get tools and make first call
    tools = agent.get_openai_tools()
    optimized_history = agent._optimize_conversation_history()
    
    print(f"\n📋 Conversation before first call:")
    for i, msg in enumerate(optimized_history):
        msg_dict = msg.to_dict()
        print(f"  {i}: role={msg_dict['role']}, content_len={len(msg_dict.get('content', ''))}, has_tool_calls={bool(msg_dict.get('tool_calls'))}")
        if msg_dict.get('tool_calls'):
            for tc in msg_dict['tool_calls']:
                print(f"    Tool call: {tc.get('function', {}).get('name', 'NO_NAME')} id={tc.get('id', 'NO_ID')}")
    
    # Make first call to LLM
    try:
        response = await agent.provider.chat_completion(
            messages=[msg.to_dict() for msg in optimized_history],
            tools=tools
        )
        
        assistant_message = response["choices"][0]["message"]
        content = assistant_message.get("content", "")
        tool_calls = assistant_message.get("tool_calls", [])
        
        print(f"\n🤖 LLM Response:")
        print(f"  Content: {content[:100]}...")
        print(f"  Tool calls: {len(tool_calls)}")
        
        for i, tc in enumerate(tool_calls):
            print(f"    Tool call {i}: id='{tc.get('id', 'MISSING')}', name='{tc.get('function', {}).get('name', 'MISSING')}'")
        
        if tool_calls:
            # Add assistant message
            agent.conversation_history.append(
                ConversationMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls
                )
            )
            
            # Process tool calls
            tool_results = await agent._process_tool_calls_parallel(tool_calls)
            
            print(f"\n🔧 Tool Results:")
            for i, result in enumerate(tool_results):
                print(f"  Tool {i}: success={result.success}, tool_name='{result.tool_name}'")
            
            # Add tool responses and trace them
            print(f"\n📝 Adding tool responses:")
            for tool_call, result in zip(tool_calls, tool_results):
                function_name = tool_call.get('function', {}).get('name')
                tool_call_id = tool_call.get('id')
                
                print(f"  Original: function_name='{function_name}', tool_call_id='{tool_call_id}'")
                
                # Apply the same logic as in the agent
                if not function_name or str(function_name).strip() == '':
                    function_name = 'unknown_function'
                    print(f"  Fixed function_name to: '{function_name}'")
                
                if not tool_call_id or str(tool_call_id).strip() == '':
                    tool_call_id = 'unknown_call_id'
                    print(f"  Fixed tool_call_id to: '{tool_call_id}'")
                
                tool_content = f"Tool result: {result.result[:100]}..." if result.success else f"Error: {result.error}"
                
                tool_message = ConversationMessage(
                    role="tool",
                    content=tool_content,
                    tool_call_id=tool_call_id,
                    function_name=function_name
                )
                
                # Check what to_dict produces
                tool_dict = tool_message.to_dict()
                print(f"  Final message dict: role='{tool_dict['role']}', name='{tool_dict.get('name', 'MISSING')}', tool_call_id='{tool_dict.get('tool_call_id', 'MISSING')}'")
                
                agent.conversation_history.append(tool_message)
            
            # Try second call to see what happens
            print(f"\n📋 Conversation before second call:")
            optimized_history = agent._optimize_conversation_history()
            for i, msg in enumerate(optimized_history):
                msg_dict = msg.to_dict()
                print(f"  {i}: role={msg_dict['role']}")
                if msg_dict['role'] == 'tool':
                    print(f"    name='{msg_dict.get('name', 'MISSING')}', tool_call_id='{msg_dict.get('tool_call_id', 'MISSING')}'")
            
            # Make second call
            print(f"\n🔄 Making second call to LLM...")
            try:
                final_response = await agent.provider.chat_completion(
                    messages=[msg.to_dict() for msg in optimized_history],
                    tools=None
                )
                print("✅ Second call successful")
            except Exception as e:
                print(f"❌ Second call failed: {e}")
        
    except Exception as e:
        print(f"❌ First call failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_function_names())
