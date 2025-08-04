#!/usr/bin/env python3
"""
Simple Agent Test
================

Test the financial agent with minimal setup to identify timeout issues.
"""

import asyncio
import time
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


async def test_agent_simple():
    """Test the agent with a simple non-tool message first."""
    print("🔍 Testing agent with simple message (no tools)...")
    
    start_time = time.time()
    try:
        from backend.agents.financial_agent import get_financial_agent
        agent = await get_financial_agent()
        
        # Test a simple message that should NOT trigger tools
        response = await asyncio.wait_for(
            agent.process_message("Hello, how are you today?"),
            timeout=30.0
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Simple agent response in {elapsed:.2f}s")
        print(f"Response: {response[:200]}...")
        
        return True
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"❌ Simple agent message timed out after {elapsed:.2f}s")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Simple agent failed after {elapsed:.2f}s: {e}")
        return False


async def test_agent_with_tool_trigger():
    """Test the agent with a message that should trigger tools."""
    print("\n🔍 Testing agent with tool-triggering message...")
    
    start_time = time.time()
    try:
        from backend.agents.financial_agent import get_financial_agent
        agent = await get_financial_agent()
        
        # Test a message that should trigger tools
        response = await asyncio.wait_for(
            agent.process_message("What is my current account balance?"),
            timeout=60.0
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Tool-based agent response in {elapsed:.2f}s")
        print(f"Response: {response[:200]}...")
        
        return True
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"❌ Tool-based agent message timed out after {elapsed:.2f}s")
        
        # Let's check what might be hanging
        print("🔍 Investigating timeout cause...")
        
        # Test LLM provider directly
        print("Testing LLM provider...")
        try:
            from backend.agents.llm_providers import create_provider, LLMConfig, ProviderType
            
            test_config = LLMConfig(
                provider=ProviderType.GEMINI,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                model="gemini-2.5-flash",
                timeout=10,
                max_tokens=100,
                temperature=0.1
            )
            
            provider = create_provider(test_config)
            
            llm_response = await asyncio.wait_for(
                provider.chat_completion([
                    {"role": "user", "content": "Say hello"}
                ]),
                timeout=10.0
            )
            print("✅ LLM provider works")
            
        except Exception as llm_error:
            print(f"❌ LLM provider failed: {llm_error}")
        
        # Test MCP client directly
        print("Testing MCP client...")
        try:
            from backend.mcp.client import get_mcp_manager
            
            mcp_manager = await get_mcp_manager()
            
            mcp_response = await asyncio.wait_for(
                mcp_manager.call_tool("financial-data-inr", "get_account_summary", {}),
                timeout=10.0
            )
            print("✅ MCP client works")
            
        except Exception as mcp_error:
            print(f"❌ MCP client failed: {mcp_error}")
        
        return False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Tool-based agent failed after {elapsed:.2f}s: {e}")
        return False


async def test_llm_provider_direct():
    """Test the LLM provider directly."""
    print("\n🔍 Testing LLM provider directly...")
    
    start_time = time.time()
    try:
        from backend.agents.llm_providers import create_provider, LLMConfig, ProviderType
        
        # Use the same config as the agent would
        test_config = LLMConfig(
            provider=ProviderType.GEMINI,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-2.5-flash",
            timeout=30,
            max_tokens=500,
            temperature=0.7
        )
        
        provider = create_provider(test_config)
        
        # Test basic chat completion
        response = await asyncio.wait_for(
            provider.chat_completion([
                {"role": "user", "content": "What is 2+2? Answer briefly."}
            ]),
            timeout=15.0
        )
        
        elapsed = time.time() - start_time
        print(f"✅ LLM provider response in {elapsed:.2f}s")
        print(f"Response: {response}")
        
        return True
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"❌ LLM provider timed out after {elapsed:.2f}s")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ LLM provider failed after {elapsed:.2f}s: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Simple Agent Tests...")
    
    # Test 1: LLM provider directly
    llm_ok = await test_llm_provider_direct()
    
    # Test 2: Simple agent message (no tools)
    if llm_ok:
        simple_ok = await test_agent_simple()
        
        # Test 3: Agent with tools (only if simple works)
        if simple_ok:
            await test_agent_with_tool_trigger()
        else:
            print("⚠️  Skipping tool test because simple agent failed")
    else:
        print("⚠️  Skipping agent tests because LLM provider failed")
    
    print("✅ Simple Agent Tests completed")


if __name__ == "__main__":
    asyncio.run(main())
