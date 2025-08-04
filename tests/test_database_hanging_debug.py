"""
Database Hanging Debug Test
============================

Focused test to diagnose why database queries are hanging.
This is the root cause of the tool timeout issues.
"""

import asyncio
import time
import sys
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.database.db import get_db_manager


class TestDatabaseHanging:
    """Test class to diagnose database hanging issues."""
    
    async def test_direct_sqlite_connection(self):
        """Test direct SQLite connection to rule out basic issues."""
        print("\n🔍 Testing direct SQLite connection...")
        
        db_path = "/home/notroot/Work/financebud/financial_data.db"
        
        start_time = time.time()
        try:
            # Test direct SQLite connection with timeout
            conn = sqlite3.connect(db_path, timeout=5.0)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM transactions")
            result = cursor.fetchone()
            
            elapsed = time.time() - start_time
            print(f"✅ Direct SQLite query completed in {elapsed:.2f}s: {result[0]} transactions")
            
            conn.close()
            return True
            
        except sqlite3.OperationalError as e:
            elapsed = time.time() - start_time
            print(f"❌ Direct SQLite query failed after {elapsed:.2f}s: {e}")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Direct SQLite query error after {elapsed:.2f}s: {e}")
            return False
    
    async def test_database_manager_hanging(self):
        """Test the OptimizedDatabaseManager for hanging issues."""
        print("\n🔍 Testing OptimizedDatabaseManager...")
        
        db_manager = get_db_manager()
        
        start_time = time.time()
        try:
            # Test with very short timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    db_manager.execute_query,
                    "SELECT COUNT(*) as count FROM transactions"
                ),
                timeout=3.0
            )
            
            elapsed = time.time() - start_time
            print(f"✅ DatabaseManager query completed in {elapsed:.2f}s")
            print(f"Result: {result.data if hasattr(result, 'data') else result}")
            return True
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ DatabaseManager query TIMED OUT after {elapsed:.2f}s")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ DatabaseManager query failed after {elapsed:.2f}s: {e}")
            return False
    
    async def test_database_locks_and_wal_mode(self):
        """Test for database locks and WAL mode issues."""
        print("\n🔍 Testing database locks and WAL mode...")
        
        db_path = "/home/notroot/Work/financebud/financial_data.db"
        
        try:
            conn = sqlite3.connect(db_path, timeout=2.0)
            cursor = conn.cursor()
            
            # Check WAL mode
            cursor.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            print(f"Journal mode: {journal_mode}")
            
            # Check for locks
            cursor.execute("PRAGMA database_list")
            databases = cursor.fetchall()
            print(f"Databases: {databases}")
            
            # Check if database is busy
            cursor.execute("PRAGMA busy_timeout")
            busy_timeout = cursor.fetchone()[0]
            print(f"Busy timeout: {busy_timeout}ms")
            
            # Check for active connections
            cursor.execute("PRAGMA wal_checkpoint(PASSIVE)")
            checkpoint_result = cursor.fetchall()
            print(f"WAL checkpoint result: {checkpoint_result}")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Database lock/WAL check failed: {e}")
            return False
    
    async def test_concurrent_database_access(self):
        """Test concurrent database access for deadlocks."""
        print("\n🔍 Testing concurrent database access...")
        
        async def make_query(query_id: int):
            """Make a database query with timing."""
            start_time = time.time()
            try:
                db_manager = get_db_manager()
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        db_manager.execute_query,
                        "SELECT COUNT(*) FROM transactions"
                    ),
                    timeout=5.0
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
        
        # Run 3 concurrent queries
        start_time = time.time()
        tasks = [make_query(i) for i in range(1, 4)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed = time.time() - start_time
        successful = sum(1 for r in results if r is True)
        print(f"Concurrent test completed in {elapsed:.2f}s: {successful}/3 successful")
        
        return successful == 3
    
    async def test_database_file_system_issues(self):
        """Test for file system issues with the database."""
        print("\n🔍 Testing database file system issues...")
        
        db_path = "/home/notroot/Work/financebud/financial_data.db"
        
        try:
            # Check if database file exists and is readable
            if not os.path.exists(db_path):
                print(f"❌ Database file does not exist: {db_path}")
                return False
            
            file_size = os.path.getsize(db_path)
            print(f"Database file size: {file_size:,} bytes")
            
            # Check file permissions
            readable = os.access(db_path, os.R_OK)
            writable = os.access(db_path, os.W_OK)
            print(f"File permissions - Readable: {readable}, Writable: {writable}")
            
            # Check for related files (WAL, SHM)
            wal_file = f"{db_path}-wal"
            shm_file = f"{db_path}-shm"
            
            wal_exists = os.path.exists(wal_file)
            shm_exists = os.path.exists(shm_file)
            print(f"WAL file exists: {wal_exists}")
            print(f"SHM file exists: {shm_exists}")
            
            if wal_exists:
                wal_size = os.path.getsize(wal_file)
                print(f"WAL file size: {wal_size:,} bytes")
            
            if shm_exists:
                shm_size = os.path.getsize(shm_file)
                print(f"SHM file size: {shm_size:,} bytes")
            
            return True
            
        except Exception as e:
            print(f"❌ File system check failed: {e}")
            return False
    
    async def test_database_integrity(self):
        """Test database integrity to check for corruption."""
        print("\n🔍 Testing database integrity...")
        
        db_path = "/home/notroot/Work/financebud/financial_data.db"
        
        try:
            conn = sqlite3.connect(db_path, timeout=10.0)
            cursor = conn.cursor()
            
            # Quick integrity check
            start_time = time.time()
            cursor.execute("PRAGMA quick_check")
            result = cursor.fetchone()[0]
            elapsed = time.time() - start_time
            
            print(f"Integrity check result: {result} ({elapsed:.2f}s)")
            
            if result == "ok":
                print("✅ Database integrity is good")
                success = True
            else:
                print(f"❌ Database integrity issue: {result}")
                success = False
            
            conn.close()
            return success
            
        except Exception as e:
            print(f"❌ Integrity check failed: {e}")
            return False
    
    def test_thread_safety_issues(self):
        """Test for thread safety issues with the database manager."""
        print("\n🔍 Testing thread safety issues...")
        
        def make_threaded_query(thread_id: int) -> bool:
            """Make a query from a separate thread."""
            start_time = time.time()
            try:
                db_manager = get_db_manager()
                result = db_manager.execute_query("SELECT COUNT(*) FROM transactions")
                elapsed = time.time() - start_time
                print(f"  Thread {thread_id}: ✅ {elapsed:.2f}s")
                return True
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"  Thread {thread_id}: ❌ {elapsed:.2f}s - {e}")
                return False
        
        # Use thread pool executor
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_threaded_query, i) for i in range(1, 4)]
            results = [future.result(timeout=10) for future in futures]
        
        successful = sum(results)
        print(f"Thread test: {successful}/3 successful")
        return successful == 3


# Test runner
async def run_database_hanging_debug():
    """Run the database hanging debug tests."""
    print("🚀 Starting Database Hanging Debug...")
    
    test_instance = TestDatabaseHanging()
    results = {}
    
    try:
        # Test 1: Direct SQLite
        results['direct_sqlite'] = await test_instance.test_direct_sqlite_connection()
        
        # Test 2: Database Manager
        results['database_manager'] = await test_instance.test_database_manager_hanging()
        
        # Test 3: Locks and WAL mode
        results['locks_wal'] = await test_instance.test_database_locks_and_wal_mode()
        
        # Test 4: Concurrent access
        results['concurrent'] = await test_instance.test_concurrent_database_access()
        
        # Test 5: File system issues
        results['filesystem'] = await test_instance.test_database_file_system_issues()
        
        # Test 6: Database integrity
        results['integrity'] = await test_instance.test_database_integrity()
        
        # Test 7: Thread safety (synchronous)
        results['thread_safety'] = test_instance.test_thread_safety_issues()
        
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        raise
    
    # Summary
    print(f"\n📊 Database Hanging Debug Summary:")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    failed_tests = [name for name, success in results.items() if not success]
    if failed_tests:
        print(f"\n🚨 Failed tests: {', '.join(failed_tests)}")
        print("These are likely the root cause of the hanging issue!")
    else:
        print("\n✅ All database tests passed - the issue might be elsewhere")
    
    print("✅ Database Hanging Debug completed")


if __name__ == "__main__":
    # Run the debug tests
    asyncio.run(run_database_hanging_debug())
