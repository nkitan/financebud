"""
Real Tool Execution Pattern Test
=================================

Test the actual pattern that tools use when making database calls.
This simulates the get_account_summary tool execution.
"""

import asyncio
import time
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.database.db import get_db_manager


async def simulate_get_account_summary():
    """Simulate the exact pattern of get_account_summary tool."""
    print("🔍 Simulating get_account_summary tool pattern...")
    
    db_manager = get_db_manager()
    
    try:
        start_time = time.time()
        
        # This is the exact sequence from get_account_summary
        print("🔧 Step 1: Get total transactions count...")
        total_result = await asyncio.to_thread(
            db_manager.execute_query,
            "SELECT COUNT(*) as count FROM transactions"
        )
        
        print("🔧 Step 2: Get date range...")
        date_range_result = await asyncio.to_thread(
            db_manager.execute_query,
            "SELECT MIN(transaction_date) as earliest_date, MAX(transaction_date) as latest_date FROM transactions"
        )
        
        print("🔧 Step 3: Get latest balance...")
        latest_balance_result = await asyncio.to_thread(
            db_manager.execute_query,
            "SELECT balance FROM transactions ORDER BY transaction_date DESC, transaction_id DESC LIMIT 1"
        )
        
        print("🔧 Step 4: Get totals...")
        totals_result = await asyncio.to_thread(
            db_manager.execute_query,
            "SELECT SUM(debit_amount) as total_debits, SUM(credit_amount) as total_credits FROM transactions WHERE debit_amount IS NOT NULL OR credit_amount IS NOT NULL"
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Account summary simulation completed in {elapsed:.2f}s")
        
        # Print results
        print(f"Total transactions: {total_result.data[0]['count']}")
        print(f"Date range: {date_range_result.data[0]}")
        print(f"Latest balance: {latest_balance_result.data[0]['balance'] if latest_balance_result.data else 0}")
        print(f"Totals: {totals_result.data[0]}")
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Account summary simulation failed after {elapsed:.2f}s: {e}")
        return False


async def simulate_mcp_tool_call():
    """Simulate the exact MCP tool call pattern that hangs."""
    print("🔍 Simulating MCP tool call pattern...")
    
    from backend.mcp.client import MCPManager
    
    manager = MCPManager()
    
    try:
        # Initialize manager
        print("🔧 Initializing MCP manager...")
        await manager.initialize_default_servers()
        
        # Get connection for debugging
        if 'financial-data-inr' in manager.connections:
            connection = manager.connections['financial-data-inr']
            print(f"Connection state: {connection.state}")
            print(f"Process running: {connection.process and connection.process.returncode is None}")
        
        # Make the actual tool call that hangs
        print("🔧 Making tool call...")
        start_time = time.time()
        
        result = await asyncio.wait_for(
            manager.call_tool('financial-data-inr', 'get_account_summary', {}),
            timeout=10.0  # Short timeout to catch hanging
        )
        
        elapsed = time.time() - start_time
        print(f"✅ MCP tool call completed in {elapsed:.2f}s")
        print(f"Result: {str(result)[:200]}...")
        return True
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"❌ MCP tool call TIMED OUT after {elapsed:.2f}s")
        
        # Debug connection state
        if 'financial-data-inr' in manager.connections:
            connection = manager.connections['financial-data-inr']
            print(f"Connection state during timeout: {connection.state}")
            if hasattr(connection, 'pending_requests'):
                print(f"Pending requests: {connection.pending_requests}")
            if hasattr(connection, 'active_requests'):
                print(f"Active requests: {connection.active_requests}")
        
        return False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ MCP tool call failed after {elapsed:.2f}s: {e}")
        return False
        
    finally:
        await manager.shutdown()


async def test_database_under_mcp_load():
    """Test database behavior when accessed through MCP servers."""
    print("🔍 Testing database under MCP load...")
    
    # Start multiple MCP managers and make concurrent tool calls
    async def mcp_worker(worker_id: int):
        manager = MCPManager()
        try:
            await manager.initialize_default_servers()
            
            result = await asyncio.wait_for(
                manager.call_tool('financial-data-inr', 'get_account_summary', {}),
                timeout=15.0
            )
            
            print(f"Worker {worker_id}: ✅ Success")
            return True
            
        except asyncio.TimeoutError:
            print(f"Worker {worker_id}: ❌ Timed out")
            return False
        except Exception as e:
            print(f"Worker {worker_id}: ❌ Failed - {e}")
            return False
        finally:
            await manager.shutdown()
    
    # Run 3 concurrent MCP workers
    print("🔧 Running 3 concurrent MCP workers...")
    start_time = time.time()
    
    tasks = [mcp_worker(i) for i in range(1, 4)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r is True)
    
    print(f"MCP load test completed in {elapsed:.2f}s")
    print(f"Successful workers: {successful}/3")
    
    if successful == 0:
        print("❌ All MCP workers failed - this confirms the hanging issue!")
        return False
    elif successful < 3:
        print("⚠️ Some MCP workers failed - intermittent hanging issue")
        return False
    else:
        print("✅ All MCP workers succeeded")
        return True


async def run_real_pattern_test():
    """Run all real pattern tests."""
    print("🚀 Starting Real Tool Execution Pattern Test...")
    
    # Test 1: Direct simulation of account summary
    print("\n" + "="*50)
    direct_ok = await simulate_get_account_summary()
    
    # Test 2: MCP tool call simulation
    print("\n" + "="*50)
    mcp_ok = await simulate_mcp_tool_call()
    
    # Test 3: MCP load test
    print("\n" + "="*50)
    load_ok = await test_database_under_mcp_load()
    
    print("\n" + "="*50)
    print("🔍 FINAL DIAGNOSIS:")
    print(f"Direct DB access: {'✅' if direct_ok else '❌'}")
    print(f"MCP tool call: {'✅' if mcp_ok else '❌'}")
    print(f"MCP load test: {'✅' if load_ok else '❌'}")
    
    if direct_ok and not mcp_ok:
        print("\n🎯 CONCLUSION: Issue is in MCP communication layer!")
        print("Database works fine, but MCP tool calls hang.")
        print("Problem likely in: backend/mcp/client.py tool execution")
    elif not direct_ok:
        print("\n🎯 CONCLUSION: Issue is in database layer!")
        print("Database operations themselves are hanging.")
    elif not load_ok:
        print("\n🎯 CONCLUSION: Issue appears under concurrent MCP load!")
        print("Single MCP calls work, but multiple concurrent calls fail.")
    else:
        print("\n🎯 CONCLUSION: System appears to be working correctly!")
        print("The hanging issue may be intermittent or timing-related.")
    
    print("\n✅ Real Tool Execution Pattern Test completed")


if __name__ == "__main__":
    asyncio.run(run_real_pattern_test())
