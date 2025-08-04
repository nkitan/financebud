#!/usr/bin/env python3
"""
Test the Fixed Gemini Provider
=============================
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.financial_agent import FinancialAgent
from backend.agents.llm_providers import LLMConfig, ProviderType
from dotenv import load_dotenv

load_dotenv()

async def test_fixed_gemini():
    """Test that the Gemini fix works."""
    print("🔍 Testing fixed Gemini provider...")
    
    config = LLMConfig(
        provider=ProviderType.GEMINI,
        base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        api_key=os.getenv("GEMINI_API_KEY", "dummy"),
        max_tokens=500,
        timeout=60
    )
    
    # Test the financial agent with fixed provider
    agent = FinancialAgent(config)
    await agent.initialize()
    
    print("✅ Agent initialized")
    
    # Test a query that would trigger tool calls
    test_queries = [
        "Show me my account summary",
        "What are my recent transactions?",
        "Give me a quick financial overview"
    ]
    
    for query in test_queries:
        print(f"\n💬 Testing: '{query}'")
        try:
            response = await agent.process_message(query)
            print(f"✅ Success! Response: {response[:200]}...")
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    
    return True

async def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ No GEMINI_API_KEY found")
        return
    
    success = await test_fixed_gemini()
    
    if success:
        print("\n🎉 All tests passed! Gemini fix is working.")
    else:
        print("\n💥 Some tests failed.")

if __name__ == "__main__":
    asyncio.run(main())
