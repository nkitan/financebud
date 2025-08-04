"""
Database Hanging Debug Test
===========================

Focused test to diagnose why database operations are hanging.
This test specifically targets the SQLite database issues.
"""

import asyncio
import time
import sys
import os
import sqlite3
import threading

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.database.db import get_db_manager


async def test_database_connection_pool():
    """Test database connection pool for deadlocks."""
    print("🔍 Testing database connection pool...")
    
    db_manager = get_db_manager()
    
    # Test multiple concurrent database calls
    async def db_query(query_id: int):
        try:
            start_time = time.time()
            print(f"Query {query_id}: Starting...")
            
            result = await asyncio.to_thread(
                db_manager.execute_query,
                "SELECT COUNT(*) as count FROM transactions"
            )
            
            elapsed = time.time() - start_time
            print(f"Query {query_id}: ✅ Completed in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Query {query_id}: ❌ Failed after {elapsed:.2f}s - {e}")
            return False
    
    # Run 3 concurrent queries
    print("🔧 Running concurrent database queries...")
    tasks = [db_query(i) for i in range(1, 4)]
    
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=15.0
        )
        
        successful = sum(1 for r in results if r is True)
        print(f"Concurrent test: {successful}/3 successful")
        
        if successful == 0:
            print("❌ All database queries failed!")
            return False
        elif successful < 3:
            print("⚠️ Some database queries failed!")
            return False
        else:
            print("✅ All database queries successful")
            return True
            
    except asyncio.TimeoutError:
        print("❌ Database concurrent test timed out!")
        return False


async def test_database_lock_detection():
    """Check for database locks and WAL mode issues."""
    print("🔍 Testing database locks...")
    
    db_path = "/home/notroot/Work/financebud/financial_data.db"
    
    try:
        # Test direct SQLite connection
        print("🔧 Testing direct SQLite connection...")
        start_time = time.time()
        
        def direct_db_test():
            try:
                conn = sqlite3.connect(db_path, timeout=5.0)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM transactions")
                result = cursor.fetchone()
                conn.close()
                return result[0]
            except Exception as e:
                print(f"Direct DB error: {e}")
                return None
        
        count = await asyncio.to_thread(direct_db_test)
        elapsed = time.time() - start_time
        
        if count is not None:
            print(f"✅ Direct DB access successful in {elapsed:.2f}s, count: {count}")
        else:
            print(f"❌ Direct DB access failed after {elapsed:.2f}s")
            return False
        
        # Check database file permissions
        print("🔧 Checking database file permissions...")
        import stat
        file_stat = os.stat(db_path)
        permissions = stat.filemode(file_stat.st_mode)
        print(f"DB file permissions: {permissions}")
        
        # Check for lock files
        lock_files = [
            db_path + "-wal",
            db_path + "-shm",
            db_path + "-journal"
        ]
        
        for lock_file in lock_files:
            if os.path.exists(lock_file):
                size = os.path.getsize(lock_file)
                print(f"Lock file exists: {lock_file} ({size} bytes)")
            else:
                print(f"No lock file: {lock_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database lock test failed: {e}")
        return False


async def test_database_manager_state():
    """Check the database manager internal state."""
    print("🔍 Testing database manager state...")
    
    db_manager = get_db_manager()
    
    # Check manager attributes
    attrs = ['connection_pool', 'cache', '_lock', '_db_path']
    for attr in attrs:
        if hasattr(db_manager, attr):
            value = getattr(db_manager, attr)
            print(f"Manager.{attr}: {type(value)} = {str(value)[:100]}")
        else:
            print(f"Manager.{attr}: NOT FOUND")
    
    # Test if manager is in a bad state
    try:
        print("🔧 Testing manager query with short timeout...")
        start_time = time.time()
        
        result = await asyncio.wait_for(
            asyncio.to_thread(
                db_manager.execute_query,
                "SELECT 1 as test"
            ),
            timeout=2.0
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Short timeout query completed in {elapsed:.2f}s")
        return True
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"❌ Short timeout query failed after {elapsed:.2f}s")
        
        # Check if manager is deadlocked
        print("🔧 Checking for deadlock...")
        
        # Try to access manager state
        try:
            if hasattr(db_manager, '_lock'):
                print(f"Manager lock state: {db_manager._lock}")
        except Exception as e:
            print(f"Cannot access manager lock: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Manager state test failed: {e}")
        return False


async def test_database_threading_issues():
    """Test for threading issues in database access."""
    print("🔍 Testing database threading issues...")
    
    import threading
    
    # Check current thread
    current_thread = threading.current_thread()
    print(f"Current thread: {current_thread.name} (ID: {current_thread.ident})")
    
    # Test from different thread contexts
    def thread_db_test(thread_id):
        try:
            thread = threading.current_thread()
            print(f"Thread {thread_id}: {thread.name} (ID: {thread.ident})")
            
            # Try to use the db_manager from this thread
            db_manager = get_db_manager()
            result = db_manager.execute_query("SELECT 1 as test")
            print(f"Thread {thread_id}: ✅ Success")
            return True
            
        except Exception as e:
            print(f"Thread {thread_id}: ❌ Failed - {e}")
            return False
    
    # Test from asyncio thread pool
    try:
        result = await asyncio.to_thread(thread_db_test, "asyncio_pool")
        if result:
            print("✅ asyncio.to_thread database access works")
        else:
            print("❌ asyncio.to_thread database access failed")
            
        return result
        
    except Exception as e:
        print(f"❌ Threading test failed: {e}")
        return False


async def run_database_debug():
    """Run all database debug tests."""
    print("🚀 Starting Database Hanging Debug Tests...")
    
    # Test 1: Database lock detection
    lock_ok = await test_database_lock_detection()
    
    # Test 2: Database manager state
    manager_ok = await test_database_manager_state()
    
    # Test 3: Threading issues
    threading_ok = await test_database_threading_issues()
    
    # Test 4: Connection pool
    if lock_ok and manager_ok:
        pool_ok = await test_database_connection_pool()
    else:
        print("⚠️ Skipping connection pool test due to earlier failures")
        pool_ok = False
    
    print("\n🔍 DIAGNOSIS SUMMARY:")
    print(f"Database file access: {'✅' if lock_ok else '❌'}")
    print(f"Database manager state: {'✅' if manager_ok else '❌'}")
    print(f"Threading compatibility: {'✅' if threading_ok else '❌'}")
    print(f"Connection pool: {'✅' if pool_ok else '❌'}")
    
    if not any([lock_ok, manager_ok, threading_ok, pool_ok]):
        print("\n❌ CRITICAL: All database tests failed!")
        print("Possible causes:")
        print("1. Database file is corrupted")
        print("2. Database is locked by another process")
        print("3. Database manager is in deadlock state")
        print("4. Threading/async incompatibility")
    elif not pool_ok:
        print("\n⚠️ Database works but connection pool has issues")
    else:
        print("\n✅ Database appears to be working correctly")
    
    print("\n✅ Database Debug Tests completed")


if __name__ == "__main__":
    asyncio.run(run_database_debug())
