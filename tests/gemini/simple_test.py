#!/usr/bin/env python3
"""
Simple Gemini Test
==================

Test a single query with better error handling.
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

async def simple_test():
    """Test a simple query that should work."""
    print("🔍 Simple Gemini test...")
    
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
    
    # Test a simple query that should work without complex tool calls
    try:
        response = await agent.process_message("Show me my account balance")
        print(f"✅ Response: {response[:200]}...")
        
        # Check for specific error
        if "Name cannot be empty" in response:
            print("❌ Still getting function name errors!")
            return False
        else:
            print("✅ No function name errors detected")
            return True
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(simple_test())
