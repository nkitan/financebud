#!/usr/bin/env python3
"""
Test improved agent with better prompting
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.agents.financial_agent_generic import GenericFinancialAgent
from backend.agents.llm_providers import get_default_config

async def test_improved_agent():
    """Test the agent with improved prompting."""
    
    config = get_default_config()
    agent = GenericFinancialAgent(config)
    
    # Test 1: Simple account query
    print("🧪 Test 1: Account summary")
    print("-" * 40)
    
    response1 = await agent.chat("What's my account balance?", "test1")
    print(f"Response 1: {response1[:200]}...")
    print(f"Length: {len(response1)}")
    
    # Check if it contains tool-generated data vs code
    if "₹" in response1 and "balance" in response1.lower():
        print("✅ Contains financial data")
    elif "<|python_tag|>" in response1 or "from tools import" in response1:
        print("❌ Generated code instead of using tools")
    else:
        print("⚠️ Unclear response type")
    
    print()
    
    # Test 2: Transaction search
    print("🧪 Test 2: Transaction search")
    print("-" * 40)
    
    response2 = await agent.chat("Find transactions containing Swiggy", "test2")
    print(f"Response 2: {response2[:200]}...")
    print(f"Length: {len(response2)}")
    
    # Check for code generation
    if "<|python_tag|>" in response2 or "from tools import" in response2:
        print("❌ Generated code instead of using tools")
    elif "swiggy" in response2.lower() or "transaction" in response2.lower():
        print("✅ Contains transaction data")
    else:
        print("⚠️ Unclear response type")

if __name__ == "__main__":
    print("🧪 Testing Improved Agent Prompting")
    print("=" * 50)
    
    asyncio.run(test_improved_agent())
