#!/usr/bin/env python3
"""
Test tool calling support for different Ollama models
"""

import asyncio
import aiohttp
import json

async def test_tool_support(model_name: str):
    """Test if a model supports tool calling."""
    print(f"\n🔍 Testing tool support for {model_name}")
    print("=" * 60)
    
    # Simple tool definition
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in New York? Use the weather tool."
        }
    ]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 100,
        "temperature": 0.1
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Check if the response contains tool calls
                    choice = result.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    tool_calls = message.get("tool_calls")
                    
                    if tool_calls:
                        print("✅ SUPPORTS TOOL CALLING")
                        print(f"📞 Tool calls detected: {len(tool_calls)}")
                        for i, tool_call in enumerate(tool_calls):
                            print(f"   {i+1}. {tool_call.get('function', {}).get('name', 'unknown')}")
                        return True
                    else:
                        print("❌ NO TOOL CALLING SUPPORT")
                        print(f"📝 Response: {message.get('content', 'No content')[:100]}...")
                        return False
                else:
                    error_text = await response.text()
                    print(f"❌ HTTP ERROR: {response.status}")
                    print(f"📝 Error: {error_text[:200]}...")
                    return False
                    
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

async def main():
    """Test multiple models for tool support."""
    print("🧪 Testing Ollama Models for Tool Calling Support")
    print("=" * 60)
    
    models_to_test = [
        "llama3.1",  # Known to support tools well
        "deepseek-r1:8b",  # This should support tools
        "gemma-3-12b-it-Q4_K_M:latest",
        "Qwen3-30B-A3B-Instruct-2507-Q4_K_M:latest", 
        "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M:latest"
    ]
    
    results = {}
    
    for model in models_to_test:
        try:
            supports_tools = await test_tool_support(model)
            results[model] = supports_tools
        except Exception as e:
            print(f"❌ Failed to test {model}: {e}")
            results[model] = False
    
    print("\n📊 SUMMARY")
    print("=" * 60)
    for model, supports in results.items():
        status = "✅ SUPPORTED" if supports else "❌ NOT SUPPORTED"
        print(f"{model}: {status}")
    
    # Recommend the best model
    supported_models = [model for model, supports in results.items() if supports]
    if supported_models:
        print(f"\n🎯 RECOMMENDED: Use {supported_models[0]} for FinanceBud")
    else:
        print("\n⚠️  WARNING: None of the tested models support tool calling!")
        print("   Consider using a different model like llama3.1 or qwen2.5")

if __name__ == "__main__":
    asyncio.run(main())
