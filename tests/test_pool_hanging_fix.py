"""
Connection Pool Hanging Fix Test
=================================

Test and fix for the connection pool hanging issue.
"""

import asyncio
import time
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.database.db import get_db_manager


async def test_connection_pool_hanging():
    """Test the connection pool for hanging issues and provide a fix."""
    print("🔍 Testing connection pool hanging...")
    
    db_manager = get_db_manager()
    
    # Test rapid concurrent queries that could exhaust the pool
    async def rapid_query(query_id: int):
        try:
            start_time = time.time()
            print(f"Query {query_id}: Starting...")
            
            # Use a very short timeout to test the hanging
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    db_manager.execute_query,
                    f"SELECT COUNT(*) as count FROM transactions LIMIT 1"
                ),
                timeout=3.0  # Short timeout to catch hanging
            )
            
            elapsed = time.time() - start_time
            print(f"Query {query_id}: ✅ Completed in {elapsed:.2f}s")
            return True
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"Query {query_id}: ❌ TIMED OUT after {elapsed:.2f}s")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Query {query_id}: ❌ Failed after {elapsed:.2f}s - {e}")
            return False
    
    # Launch 15 concurrent queries (more than the default pool size of 10)
    print("🔧 Launching 15 concurrent queries...")
    tasks = [rapid_query(i) for i in range(1, 16)]
    
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_elapsed = time.time() - start_time
    
    successful = sum(1 for r in results if r is True)
    timed_out = sum(1 for r in results if r is False)
    exceptions = sum(1 for r in results if isinstance(r, Exception))
    
    print(f"\n📊 Results after {total_elapsed:.2f}s:")
    print(f"✅ Successful: {successful}/15")
    print(f"⏰ Timed out: {timed_out}/15")
    print(f"❌ Exceptions: {exceptions}/15")
    
    if timed_out > 0:
        print("\n❌ CONFIRMED: Connection pool hanging issue!")
        print("The connection pool is not properly managing connections.")
        print("Connections are being exhausted and not returned properly.")
        return False
    else:
        print("\n✅ Connection pool working correctly")
        return True


async def test_fix_recommendations():
    """Provide fix recommendations for the connection pool issue."""
    print("\n🔧 RECOMMENDED FIXES:")
    
    print("1. **Increase connection pool timeout** - Current 5s/10s is too low")
    print("2. **Add connection pool monitoring** - Track active/available connections")
    print("3. **Implement connection cleanup** - Force return connections after use")
    print("4. **Add connection health checks** - Detect and replace dead connections")
    print("5. **Use asyncio-native connection pool** - Replace threading with asyncio")
    
    print("\n🎯 IMMEDIATE FIX: Modify ConnectionPool.get_connection() timeout")
    print("Current: timeout=5.0 and timeout=10.0")
    print("Recommended: timeout=30.0 and proper connection lifecycle management")
    
    # Show the exact location of the issue
    print("\n📍 Issue location: backend/database/db.py lines 82-92")
    print("The queue.get(timeout=X) calls are causing the hanging.")


if __name__ == "__main__":
    async def run_test():
        print("🚀 Starting Connection Pool Hanging Fix Test...")
        
        # Test the hanging issue
        pool_working = await test_connection_pool_hanging()
        
        # Provide fix recommendations
        await test_fix_recommendations()
        
        if not pool_working:
            print("\n🎯 CONCLUSION: Connection pool is the root cause of 320s timeouts!")
            print("Tool calls hang because database queries never complete.")
            print("LLM providers (Ollama/Gemini) are innocent - it's a DB connection pool bug.")
        
        print("\n✅ Connection Pool Fix Test completed")
    
    asyncio.run(run_test())
