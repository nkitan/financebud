#!/usr/bin/env python3
"""
Example: Using Multiple LLM Providers with FinanceBud
====================================================

This script demonstrates how to easily switch between different LLM providers
(Ollama, OpenAI, Gemini, OpenRouter) with the same financial agent.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.llm_providers import LLMConfig, ProviderType
from backend.agents.financial_agent_generic import GenericFinancialAgent

async def test_provider(config: LLMConfig, test_message: str = "Give me a quick account summary"):
    """Test a specific provider configuration."""
    print(f"\n🔍 Testing {config.provider.value} ({config.model})")
    print("=" * 50)
    
    try:
        agent = GenericFinancialAgent(config)
        
        # Test connection
        connection_ok = await agent.test_connection()
        print(f"📡 Connection: {'✅ OK' if connection_ok else '❌ Failed'}")
        
        if connection_ok:
            # Test chat
            print(f"💬 Testing chat with: '{test_message}'")
            response = await agent.chat(test_message)
            print(f"🤖 Response: {response[:200]}...")
            
            # Health check
            health = await agent.get_health()
            print(f"🏥 Health: {health}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def main():
    """Demonstrate switching between different LLM providers."""
    
    # Example 1: Ollama (Local)
    ollama_config = LLMConfig(
        provider=ProviderType.OLLAMA,
        base_url="http://localhost:11434",
        model="llama3.1"
    )
    await test_provider(ollama_config)
    
    # Example 2: OpenAI (if API key is available)
    if os.getenv("OPENAI_API_KEY"):
        openai_config = LLMConfig(
            provider=ProviderType.OPENAI,
            base_url="https://api.openai.com",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        await test_provider(openai_config)
    else:
        print("\n⚠️  Skipping OpenAI test (no API key)")
    
    # Example 3: OpenRouter (if API key is available)
    if os.getenv("OPENROUTER_API_KEY"):
        openrouter_config = LLMConfig(
            provider=ProviderType.OPENROUTER,
            base_url="https://openrouter.ai/api",
            model="anthropic/claude-3-sonnet",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        await test_provider(openrouter_config)
    else:
        print("\n⚠️  Skipping OpenRouter test (no API key)")
    
    # Example 4: Switching providers dynamically
    print("\n🔄 Dynamic Provider Switching")
    print("=" * 50)
    
    agent = GenericFinancialAgent(ollama_config)
    print(f"Started with: {agent.config.provider.value}")
    
    # Switch to a different provider (example)
    if os.getenv("OPENAI_API_KEY"):
        openai_config = LLMConfig(
            provider=ProviderType.OPENAI,
            base_url="https://api.openai.com",
            model="gpt-3.5-turbo",  # Cheaper for testing
            api_key=os.getenv("OPENAI_API_KEY")
        )
        agent.switch_provider(openai_config)
        print(f"Switched to: {agent.config.provider.value}")
    
    print("\n✅ Provider testing complete!")

if __name__ == "__main__":
    print("🚀 FinanceBud Multi-Provider Test")
    print("=" * 50)
    print("Make sure you have:")
    print("1. Ollama running locally (ollama serve)")
    print("2. API keys in environment variables (optional)")
    print("3. MCP server running for financial data")
    
    asyncio.run(main())
