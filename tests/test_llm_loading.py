#!/usr/bin/env python3
"""
Direct LLM Provider Test
========================

Test the LLM provider directly to isolate any issues.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from backend.agents.llm_providers import get_default_config, create_provider

async def test_llm_provider():
    """Test the LLM provider directly."""
    print("🧪 Testing LLM Provider Direct Connection...")
    
    try:
        # Get configuration
        config = get_default_config()
        print(f"✅ Configuration loaded:")
        print(f"   Provider: {config.provider}")
        print(f"   Model: {config.model}")
        print(f"   Base URL: {config.base_url}")
        print(f"   Timeout: {config.timeout}s")
        
        # Create provider
        provider = create_provider(config)
        print(f"✅ Provider created: {type(provider).__name__}")
        
        # Test connection
        print("🔍 Testing connection...")
        connection_ok = await provider.test_connection()
        print(f"✅ Connection test: {'PASSED' if connection_ok else 'FAILED'}")
        
        if not connection_ok:
            print("❌ Connection test failed, aborting chat test")
            return
        
        # Test simple chat
        print("💬 Testing simple chat completion...")
        messages = [
            {"role": "user", "content": "Hello, respond with just 'test successful'"}
        ]
        
        response = await provider.chat_completion(messages)
        print(f"✅ Chat response received:")
        
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            print(f"   Content: {content}")
        else:
            print(f"   Raw response: {response}")
        
        # Test with tools
        print("🔧 Testing chat with tools...")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"}
                        },
                        "required": ["input"]
                    }
                }
            }
        ]
        
        messages = [
            {"role": "user", "content": "Use the test_tool with input 'hello'"}
        ]
        
        response = await provider.chat_completion(messages, tools=tools)
        print(f"✅ Tool response received:")
        
        if 'choices' in response and len(response['choices']) > 0:
            message = response['choices'][0]['message']
            content = message.get('content', '')
            tool_calls = message.get('tool_calls', [])
            
            print(f"   Content: {content}")
            print(f"   Tool calls: {len(tool_calls)}")
            
            if tool_calls:
                for tool_call in tool_calls:
                    print(f"     - {tool_call.get('function', {}).get('name', 'unknown')}")
        else:
            print(f"   Raw response: {response}")
        
        print("✅ All LLM tests completed successfully!")
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_llm_provider())
