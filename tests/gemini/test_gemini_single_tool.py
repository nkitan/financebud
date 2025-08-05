#!/usr/bin/env python3
"""
Test a single tool call with detailed error logging
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

async def test_single_tool():
    """Test a single tool that was failing."""
    print("🔍 Testing single tool call...")
    
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
    
    # Test the specific tool that was failing
    try:
        print("🔧 Testing find_recurring_payments tool directly...")
        from backend.agents.financial_agent import find_recurring_payments_tool
        result = await find_recurring_payments_tool("")
        print(f"Direct tool result: {result[:200]}...")
    except Exception as e:
        print(f"❌ Direct tool call failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test through the agent
    try:
        print("\n🤖 Testing through agent...")
        response = await agent.process_message("Find my recurring payments")
        print(f"Agent response: {response[:200]}...")
    except Exception as e:
        print(f"❌ Agent call failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_tool())
