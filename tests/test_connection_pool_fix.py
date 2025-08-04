"""
Database Manager Fix Test
=========================

Test to verify the connection pool deadlock fix.
"""

import asyncio
import time
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.database.db import get_db_manager


async def test_connection_pool_fix():
    """Test the connection pool deadlock fix."""
    print("🚀 Testing connection pool deadlock fix...")
    
    # Test 1: Simple query should work
    print("\n🔍 Test 1: Simple query...")
    db_manager = get_db_manager()
    
    start_time = time.time()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                db_manager.execute_query,
                "SELECT COUNT(*) as count FROM transactions"
            ),
            timeout=2.0  # Very short timeout
        )
        elapsed = time.time() - start_time
        print(f"✅ Simple query: {elapsed:.2f}s - {result.data[0]['count']} transactions")
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"❌ Simple query timed out after {elapsed:.2f}s")
        return False
    
    # Test 2: Multiple concurrent queries
    print("\n🔍 Test 2: Concurrent queries...")
    
    async def make_query(query_id: int):
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    db_manager.execute_query,
                    f"SELECT COUNT(*) as count FROM transactions LIMIT 1"
                ),
                timeout=2.0
            )
            elapsed = time.time() - start_time
            print(f"  Query {query_id}: ✅ {elapsed:.2f}s")
            return True
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"  Query {query_id}: ❌ TIMEOUT {elapsed:.2f}s")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  Query {query_id}: ❌ ERROR {elapsed:.2f}s - {e}")
            return False
    
    # Run 5 concurrent queries
    tasks = [make_query(i) for i in range(1, 6)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful = sum(1 for r in results if r is True)
    print(f"Concurrent test: {successful}/5 successful")
    
    return successful >= 4  # Allow 1 failure


if __name__ == "__main__":
    asyncio.run(test_connection_pool_fix())
