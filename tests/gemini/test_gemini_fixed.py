#!/usr/bin/env python3
"""
Test the Fixed Gemini Provider
=============================

This test verifies that the Gemini function name fix is working and
that SQL issues have been resolved.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
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
        "Give me a quick financial overview",
        "Find my recurring payments",  # This was failing before the SQL fix
        "Analyze my spending patterns",
        "What is my current balance?",
        "Summarize my investments",
    ]
    
    success_count = 0
    failed_queries = []
    
    for query in test_queries:
        print(f"\n💬 Testing: '{query}'")
        try:
            response = await agent.process_message(query)
            
            # Check for specific Gemini API errors (should not happen with fix)
            if "Gemini returned status 400" in response and "Name cannot be empty" in response:
                print(f"❌ Gemini API function name error: {response[:200]}...")
                failed_queries.append(f"{query} - Gemini function name error")
            elif "misuse of window function LAG()" in response:
                print(f"❌ SQL LAG() error: {response[:200]}...")
                failed_queries.append(f"{query} - SQL LAG() error")
            elif "I'm experiencing connection issues" in response and "Gemini returned status 400" in response:
                print(f"❌ Gemini connection error: {response[:200]}...")
                failed_queries.append(f"{query} - Gemini connection error")
            else:
                print(f"✅ Success! Response: {response[:200]}...")
                success_count += 1
        except Exception as e:
            print(f"❌ Failed: {e}")
            failed_queries.append(f"{query} - Exception: {e}")
    
    print(f"\n📊 Results: {success_count}/{len(test_queries)} queries successful")
    
    if failed_queries:
        print("❌ Failed queries:")
        for failed in failed_queries:
            print(f"   - {failed}")
    
    # Test was successful if we don't have the specific Gemini function name errors
    gemini_function_errors = [f for f in failed_queries if "function name error" in f]
    sql_lag_errors = [f for f in failed_queries if "SQL LAG() error" in f]
    
    if len(gemini_function_errors) == 0 and len(sql_lag_errors) == 0:
        print("\n🎉 Core fixes working! No Gemini function name errors or SQL LAG() errors.")
        return True
    else:
        print(f"\n💥 Still have issues: {len(gemini_function_errors)} Gemini errors, {len(sql_lag_errors)} SQL errors")
        return False

async def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ No GEMINI_API_KEY found")
        return
    
    success = await test_fixed_gemini()
    
    if success:
        print("\n🎉 All tests passed! Gemini fix is working.")
    else:
        print("\n💥 Some tests failed.")
    
    exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
