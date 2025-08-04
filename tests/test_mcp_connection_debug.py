"""
MCP Connection State Debug Test
===============================

Focused test to diagnose the "invalid state" error and connection hanging issues.
This test specifically looks at MCP client connection management and state transitions.
"""

import asyncio
import pytest
import time
import sys
import os
import json
import subprocess
from unittest.mock import Mock, patch, AsyncMock

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.mcp.client import MCPManager, MCPConnection, MCPServerConfig, ConnectionState


class TestMCPConnectionState:
    """Test class to diagnose MCP connection state issues."""
    
    async def test_mcp_client_state_transitions(self):
        """Test MCP connection state transitions to find invalid states."""
        print("\n🔍 Testing MCP connection state transitions...")
        
        # Test individual MCP connection
        config = MCPServerConfig(
            name="test-connection",
            command="/home/notroot/Work/financebud/venv/bin/python",
            args=["/home/notroot/Work/financebud/mcp_server.py"],
            timeout=30.0
        )
        connection = MCPConnection(config)
        
        print(f"Initial state: {connection.state}")
        
        # Test connection
        try:
            await asyncio.wait_for(client.connect(), timeout=10.0)
            print(f"After connect: {client.process}")
            
            # Test if client is responsive
            if hasattr(client, 'process') and client.process:
                print(f"Process state: {client.process.returncode}")
            
            # Test disconnection
            await client.disconnect()
            print(f"After disconnect: {client.process}")
            
        except Exception as e:
            print(f"❌ MCP client state test failed: {e}")
            raise
    
    async def test_mcp_server_startup_sequence(self):
        """Test the MCP server startup sequence for hanging issues."""
        print("\n🔍 Testing MCP server startup sequence...")
        
        # Test starting the financial data server specifically
        server_cmd = [
            "python", "mcp_server.py"
        ]
        
        start_time = time.time()
        try:
            # Start the process with timeout
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *server_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd="/home/notroot/Work/financebud"
                ),
                timeout=10.0
            )
            
            elapsed = time.time() - start_time
            print(f"✅ MCP server started in {elapsed:.2f}s, PID: {process.pid}")
            
            # Give it time to initialize
            await asyncio.sleep(2)
            
            # Check if it's still running
            if process.returncode is None:
                print("✅ MCP server is running")
            else:
                print(f"❌ MCP server exited with code: {process.returncode}")
                stdout, stderr = await process.communicate()
                print(f"STDOUT: {stdout.decode()}")
                print(f"STDERR: {stderr.decode()}")
            
            # Clean up
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except:
                process.kill()
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ MCP server startup timed out after {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ MCP server startup failed after {elapsed:.2f}s: {e}")
    
    async def test_mcp_tool_call_state_tracking(self):
        """Test tool call state tracking to find where hangs occur."""
        print("\n🔍 Testing MCP tool call state tracking...")
        
        manager = MCPManager()
        
        try:
            # Initialize with timeout
            init_start = time.time()
            await asyncio.wait_for(
                manager.initialize_default_servers(),
                timeout=15.0
            )
            init_elapsed = time.time() - init_start
            print(f"✅ MCP manager initialized in {init_elapsed:.2f}s")
            
            # Test tool call with detailed state tracking
            tool_start = time.time()
            
            # Add state tracking to the tool call
            async def tracked_tool_call():
                print("🔧 Starting tool call...")
                try:
                    result = await manager.call_tool('get_account_summary', {})
                    print("🔧 Tool call completed")
                    return result
                except Exception as e:
                    print(f"🔧 Tool call failed: {e}")
                    raise
            
            result = await asyncio.wait_for(
                tracked_tool_call(),
                timeout=20.0
            )
            
            tool_elapsed = time.time() - tool_start
            print(f"✅ Tool call completed in {tool_elapsed:.2f}s")
            print(f"Result type: {type(result)}")
            
        except asyncio.TimeoutError:
            elapsed = time.time() - tool_start if 'tool_start' in locals() else 0
            print(f"❌ Tool call timed out after {elapsed:.2f}s")
            
            # Check manager state
            print(f"Manager clients: {len(manager.clients) if hasattr(manager, 'clients') else 'unknown'}")
            
        except Exception as e:
            elapsed = time.time() - tool_start if 'tool_start' in locals() else 0
            print(f"❌ Tool call failed after {elapsed:.2f}s: {e}")
            
        finally:
            await manager.shutdown()
    
    async def test_concurrent_mcp_connections(self):
        """Test concurrent MCP connections for deadlock detection."""
        print("\n🔍 Testing concurrent MCP connections...")
        
        async def create_manager_and_call(manager_id: int):
            """Create a manager and make a tool call."""
            manager = MCPManager()
            try:
                print(f"Manager {manager_id}: Initializing...")
                await asyncio.wait_for(
                    manager.initialize_default_servers(),
                    timeout=10.0
                )
                print(f"Manager {manager_id}: Making tool call...")
                result = await asyncio.wait_for(
                    manager.call_tool('get_account_summary', {}),
                    timeout=15.0
                )
                print(f"Manager {manager_id}: ✅ Success")
                return True
            except Exception as e:
                print(f"Manager {manager_id}: ❌ Failed - {e}")
                return False
            finally:
                await manager.shutdown()
        
        # Test 2 concurrent managers
        start_time = time.time()
        tasks = [
            create_manager_and_call(1),
            create_manager_and_call(2)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time
        
        successful = sum(1 for r in results if r is True)
        print(f"Concurrent test completed in {elapsed:.2f}s")
        print(f"Successful managers: {successful}/2")
        
        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                print(f"Manager {i} exception: {result}")
    
    async def test_mcp_process_monitoring(self):
        """Test MCP process monitoring and cleanup."""
        print("\n🔍 Testing MCP process monitoring...")
        
        # Check for existing MCP processes
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'mcp_server'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            existing_pids = result.stdout.strip().split('\n') if result.stdout.strip() else []
            print(f"Existing MCP processes: {len(existing_pids)}")
            
            for pid in existing_pids:
                if pid:
                    print(f"  PID: {pid}")
            
            # Check if any are zombie processes
            if existing_pids:
                ps_result = subprocess.run(
                    ['ps', '-p', ','.join(existing_pids), '-o', 'pid,ppid,state,comm'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                print("Process states:")
                print(ps_result.stdout)
                
        except subprocess.TimeoutExpired:
            print("❌ Process check timed out")
        except Exception as e:
            print(f"❌ Process check failed: {e}")
    
    async def test_database_lock_detection(self):
        """Test for database locks that might cause tool hanging."""
        print("\n🔍 Testing database lock detection...")
        
        from backend.database.db import get_db_manager
        
        db_manager = get_db_manager()
        
        # Test rapid consecutive queries (potential lock issue)
        async def make_query(query_id: int):
            try:
                start_time = time.time()
                result = await asyncio.to_thread(
                    db_manager.execute_query,
                    "SELECT COUNT(*) as count FROM transactions"
                )
                elapsed = time.time() - start_time
                print(f"Query {query_id}: ✅ {elapsed:.2f}s")
                return True
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"Query {query_id}: ❌ {elapsed:.2f}s - {e}")
                return False
        
        # Run 5 concurrent queries
        tasks = [make_query(i) for i in range(1, 6)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if r is True)
        print(f"Database concurrent test: {successful}/5 successful")


# Test runner
async def run_mcp_connection_debug():
    """Run the MCP connection state debug tests."""
    print("🚀 Starting MCP Connection State Debug...")
    
    test_instance = TestMCPConnectionState()
    
    try:
        await test_instance.test_mcp_client_state_transitions()
        await test_instance.test_mcp_server_startup_sequence()
        await test_instance.test_mcp_tool_call_state_tracking()
        await test_instance.test_concurrent_mcp_connections()
        await test_instance.test_mcp_process_monitoring()
        await test_instance.test_database_lock_detection()
        
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        raise
    
    print("✅ MCP Connection State Debug completed")


if __name__ == "__main__":
    # Run the debug tests
    asyncio.run(run_mcp_connection_debug())
