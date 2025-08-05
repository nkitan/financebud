#!/usr/bin/env python3
"""
Verify Environment and Test Basic Functionality
==============================================
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from backend.agents.financial_agent import FinancialAgent, get_account_summary_tool
    from backend.agents.llm_providers import LLMConfig, ProviderType
    from dotenv import load_dotenv
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

load_dotenv()

async def test_environment():
    """Test that the environment is set up correctly."""
    print("🔍 Testing environment...")
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "dummy":
        print("❌ No valid GEMINI_API_KEY found")
        return False
    
    print("✅ API key found")
    
    # Test tool directly first
    try:
        print("🔧 Testing tool directly...")
        result = await get_account_summary_tool("")
        print(f"✅ Direct tool test successful: {result[:100]}...")
    except Exception as e:
        print(f"❌ Direct tool test failed: {e}")
        return False
    
    # Test agent
    try:
        print("🤖 Testing agent...")
        config = LLMConfig(
            provider=ProviderType.GEMINI,
            base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            api_key=api_key,
            max_tokens=200,
            timeout=30
        )
        
        agent = FinancialAgent(config)
        await agent.initialize()
        print("✅ Agent initialized")
        
        # Simple test
        response = await agent.process_message("Hello")
        print(f"✅ Agent response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_environment())
    print(f"\n{'✅ Environment test passed' if success else '❌ Environment test failed'}")
    sys.exit(0 if success else 1)
