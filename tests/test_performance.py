#!/usr/bin/env python3
"""
Quick Performance Test
======================

Test the optimized financial agent for performance.
"""

import asyncio
import sys
import os
import time

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

async def quick_test():
    """Quick test of optimized agent."""
    print("🚀 Testing optimized configuration...")
    
    try:
        from backend.agents.llm_providers import get_default_config
        config = get_default_config()
        print(f"✅ Timeout: {config.timeout}s, Max tokens: {config.max_tokens}")
        
        # Test simple message processing time
        from backend.agents.financial_agent import get_financial_agent
        from backend.mcp.client import get_mcp_manager
        
        print("🔧 Initializing services...")
        mcp_manager = await get_mcp_manager()
        await mcp_manager.initialize_default_servers()
        
        agent = await get_financial_agent()
        print("✅ Agent ready")
        
        # Test a simple query
        print("💬 Testing simple balance query...")
        start_time = time.time()
        
        try:
            response = await asyncio.wait_for(
                agent.process_message("What is my account balance?"),
                timeout=310.0  # Should complete within 5 minutes + buffer
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Response received in {elapsed:.2f}s")
            print(f"📄 Response preview: {response[:100]}...")
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ Still timed out after {elapsed:.2f}s")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    print("🎯 Test completed successfully!" if success else "❌ Test failed")
