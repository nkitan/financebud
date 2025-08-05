#!/usr/bin/env python3
"""
Quick LLM Test
=============

Test just the LLM provider to see if it's working.
"""

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


import asyncio
import time
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


async def test_llm_quick():
    """Quick LLM test."""
    print("🔍 Quick LLM test...")
    
    try:
        from backend.agents.llm_providers import create_provider, LLMConfig, ProviderType
        
        print("✅ Imports successful")
        
        # Use Gemini config
        config = LLMConfig(
            provider=ProviderType.GEMINI,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-2.5-flash",
            timeout=10,
            max_tokens=50,
            temperature=0.1
        )
        
        print("✅ Config created")
        
        provider = create_provider(config)
        print("✅ Provider created")
        
        start_time = time.time()
        response = await asyncio.wait_for(
            provider.chat_completion([
                {"role": "user", "content": "Say hello"}
            ]),
            timeout=15.0
        )
        elapsed = time.time() - start_time
        
        print(f"✅ LLM response in {elapsed:.2f}s: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_llm_quick())
