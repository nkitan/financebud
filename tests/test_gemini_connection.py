#!/usr/bin/env python3
"""
Gemini API Connection Test
=========================

Test the Gemini API connection directly to diagnose LLM timeout issues.
"""

import asyncio
import aiohttp
import json
import time
import sys
import os
from dotenv import load_dotenv

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()


async def test_gemini_connection():
    """Test Gemini API connection directly."""
    print("🔍 Testing Gemini API connection...")
    
    # Get configuration from environment
    base_url = os.getenv('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta/openai')
    model = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        return False
    
    print(f"📊 Configuration:")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    print(f"  API Key: {api_key[:10]}...")
    
    # Prepare the request
    chat_url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Hello! Just respond with 'Hi' to test the connection."}
        ],
        "max_tokens": 50,
        "temperature": 0.1
    }
    
    # Test connection with different timeouts
    timeouts = [5, 10, 30, 60]
    
    for timeout in timeouts:
        print(f"\n🔧 Testing with {timeout}s timeout...")
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    chat_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    elapsed = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ Success in {elapsed:.2f}s")
                        print(f"Response: {result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ HTTP {response.status} after {elapsed:.2f}s")
                        print(f"Error: {error_text[:200]}")
                        
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ Timeout after {elapsed:.2f}s")
            
        except aiohttp.ClientConnectorError as e:
            elapsed = time.time() - start_time
            print(f"❌ Connection error after {elapsed:.2f}s: {e}")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Error after {elapsed:.2f}s: {e}")
    
    return False


async def test_alternative_endpoints():
    """Test alternative Gemini endpoints."""
    print("\n🔍 Testing alternative Gemini endpoints...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return
    
    # Alternative endpoints to try
    endpoints = [
        {
            "name": "Direct Gemini API",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "endpoint": "/models/gemini-2.0-flash-exp:generateContent",
            "method": "POST",
            "headers": {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            },
            "payload": {
                "contents": [
                    {
                        "parts": [
                            {"text": "Hello! Just respond with 'Hi' to test the connection."}
                        ]
                    }
                ]
            }
        },
        {
            "name": "OpenAI-compatible (v1beta)",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "endpoint": "/chat/completions",
            "method": "POST",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            "payload": {
                "model": "gemini-2.0-flash-exp",
                "messages": [
                    {"role": "user", "content": "Hello! Just respond with 'Hi' to test the connection."}
                ],
                "max_tokens": 50
            }
        }
    ]
    
    for config in endpoints:
        print(f"\n🧪 Testing {config['name']}...")
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                url = config['base_url'] + config['endpoint']
                
                async with session.request(
                    config['method'],
                    url,
                    json=config['payload'],
                    headers=config['headers'],
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    elapsed = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ {config['name']} success in {elapsed:.2f}s")
                        print(f"Response: {json.dumps(result, indent=2)[:200]}...")
                    else:
                        error_text = await response.text()
                        print(f"❌ {config['name']} HTTP {response.status} after {elapsed:.2f}s")
                        print(f"Error: {error_text[:200]}")
                        
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ {config['name']} error after {elapsed:.2f}s: {e}")


if __name__ == "__main__":
    asyncio.run(test_gemini_connection())
    asyncio.run(test_alternative_endpoints())
