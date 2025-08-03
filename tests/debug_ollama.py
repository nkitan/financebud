#!/usr/bin/env python3
"""
Debug script to test Ollama API directly
"""
import asyncio
import aiohttp
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_ollama_direct():
    """Test Ollama API directly to debug the issue."""
    
    chat_url = "http://localhost:11434/v1/chat/completions"
    
    # Simple test without tools
    simple_payload = {
        "model": "llama3.1",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 10,
        "temperature": 0.7,
        "stream": False
    }
    
    logger.info("Testing simple chat completion...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                chat_url,
                json=simple_payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                logger.info(f"Response status: {response.status}")
                logger.info(f"Response headers: {dict(response.headers)}")
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Success! Response: {json.dumps(result, indent=2)}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Error status {response.status}: {error_text}")
                    return None
    except Exception as e:
        logger.error(f"Exception occurred: {type(e).__name__}: {e}")
        logger.error(f"Exception args: {e.args}")
        return None

async def test_ollama_with_tools():
    """Test Ollama API with tools to see if that's the issue."""
    
    chat_url = "http://localhost:11434/v1/chat/completions"
    
    # Test with tools
    tools_payload = {
        "model": "llama3.1", 
        "messages": [{"role": "user", "content": "What's my account balance?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_account_summary",
                "description": "Get account summary with current balance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_input": {
                            "type": "string",
                            "description": "Input parameters"
                        }
                    },
                    "required": ["tool_input"]
                }
            }
        }],
        "tool_choice": "auto",
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }
    
    logger.info("Testing chat completion with tools...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                chat_url,
                json=tools_payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                logger.info(f"Response status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Success with tools! Response: {json.dumps(result, indent=2)}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Error with tools status {response.status}: {error_text}")
                    return None
    except Exception as e:
        logger.error(f"Exception with tools: {type(e).__name__}: {e}")
        logger.error(f"Exception args: {e.args}")
        return None

async def main():
    print("🧪 Testing Ollama API directly...")
    
    # Test 1: Simple completion
    result1 = await test_ollama_direct()
    
    # Test 2: With tools
    result2 = await test_ollama_with_tools()
    
    if result1:
        print("✅ Simple API call works!")
    else:
        print("❌ Simple API call failed!")
        
    if result2:
        print("✅ API call with tools works!")
    else:
        print("❌ API call with tools failed!")

if __name__ == "__main__":
    asyncio.run(main())
