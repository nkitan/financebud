#!/usr/bin/env python3
"""
Test multiple Gemini tool calls to verify the fix
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.agents.financial_agent import FinancialAgent
from backend.agents.llm_providers import LLMConfig, ProviderType
from dotenv import load_dotenv

load_dotenv()

async def test_gemini_tool_calls():
    """Test multiple tool calls that should work."""
    print("🔍 Testing Gemini tool calls...")
    
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
    
    # Test queries that should work (no SQL issues)
    test_queries = [
        "Show me my account summary",
        "Get my recent 5 transactions", 
        "What's my current balance?",
        "Search for UPI transactions",
        "Show me transactions from last month"
    ]
    
    success_count = 0
    for query in test_queries:
        print(f"\n💬 Testing: '{query}'")
        try:
            response = await agent.process_message(query)
            if "Gemini returned status 400" in response or "Name cannot be empty" in response:
                print(f"❌ Gemini API error: {response[:200]}...")
            else:
                print(f"✅ Success: {response[:150]}...")
                success_count += 1
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print(f"\n📊 Results: {success_count}/{len(test_queries)} queries successful")
    
    if success_count == len(test_queries):
        print("🎉 All tool calls working! Gemini function name fix is successful.")
        return True
    else:
        print("💥 Some tool calls still failing.")
        return False

if __name__ == "__main__":
    asyncio.run(test_gemini_tool_calls())
