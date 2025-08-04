"""
Tool Hanging Debug Test
=======================

Focused test to diagnose why tool calls are hanging indefinitely.
This test specifically targets the MCP tool execution pipeline.
"""

import asyncio
import time
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.mcp.client import MCPManager


async def test_simple_tool_call():
    """Test the simplest possible tool call to isolate hanging."""
    print("🔍 Testing simple tool call...")
    
    manager = MCPManager()
    
    try:
        # Initialize with timeout
        print("🔧 Initializing manager...")
        start_time = time.time()
        await asyncio.wait_for(
            manager.initialize_default_servers(),
            timeout=20.0
        )
        init_elapsed = time.time() - start_time
        print(f"✅ Manager initialized in {init_elapsed:.2f}s")
        
        # Test the simplest tool call
        print("🔧 Making tool call...")
        tool_start = time.time()
        
        try:
            # Use the correct call signature: server_name, tool_name, arguments
            result = await asyncio.wait_for(
                manager.call_tool('financial-data-inr', 'get_account_summary', {}),
                timeout=10.0  # Very short timeout to catch hanging quickly
            )
            tool_elapsed = time.time() - tool_start
            print(f"✅ Tool call completed in {tool_elapsed:.2f}s")
            print(f"Result: {str(result)[:200]}...")
            return True
            
        except asyncio.TimeoutError:
            tool_elapsed = time.time() - tool_start
            print(f"❌ Tool call TIMED OUT after {tool_elapsed:.2f}s")
            
            # Check health status when hanging
            print("🔍 Checking health status during hang...")
            health = await manager.health_check()
            print(f"Health: {health}")
            
            return False
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            print(f"❌ Manager initialization timed out after {elapsed:.2f}s")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        print("🔧 Shutting down manager...")
        await manager.shutdown()


async def test_mcp_process_state():
    """Check the actual MCP process state."""
    print("🔍 Testing MCP process state...")
    
    import subprocess
    
    try:
        # Check if MCP server processes are running
        result = subprocess.run(
            ['pgrep', '-af', 'mcp_server'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.stdout.strip():
            print("🔧 MCP server processes found:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
        else:
            print("🔧 No MCP server processes found")
            
        # Check for zombie processes
        zombie_result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        zombie_lines = [line for line in zombie_result.stdout.split('\n') if 'Z' in line and 'mcp' in line.lower()]
        if zombie_lines:
            print("⚠️ Zombie MCP processes found:")
            for line in zombie_lines:
                print(f"  {line}")
        else:
            print("✅ No zombie MCP processes")
            
    except subprocess.TimeoutExpired:
        print("❌ Process check timed out")
    except Exception as e:
        print(f"❌ Process check failed: {e}")


async def test_database_direct_access():
    """Test if database access is the bottleneck."""
    print("🔍 Testing direct database access...")
    
    from backend.database.db import get_db_manager
    
    db_manager = get_db_manager()
    
    try:
        start_time = time.time()
        result = await asyncio.wait_for(
            asyncio.to_thread(
                db_manager.execute_query,
                "SELECT COUNT(*) as count FROM transactions"
            ),
            timeout=5.0
        )
        elapsed = time.time() - start_time
        print(f"✅ Direct DB access completed in {elapsed:.2f}s")
        print(f"Result: {result.data}")
        return True
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"❌ Direct DB access timed out after {elapsed:.2f}s")
        return False
        
    except Exception as e:
        print(f"❌ Direct DB access failed: {e}")
        return False


async def test_mcp_server_direct():
    """Test starting MCP server directly to check for startup issues."""
    print("🔍 Testing direct MCP server startup...")
    
    try:
        start_time = time.time()
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "/home/notroot/Work/financebud/venv/bin/python",
                "/home/notroot/Work/financebud/mcp_server.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/home/notroot/Work/financebud"
            ),
            timeout=15.0
        )
        
        startup_elapsed = time.time() - start_time
        print(f"✅ MCP server started in {startup_elapsed:.2f}s, PID: {process.pid}")
        
        # Wait a bit for initialization
        await asyncio.sleep(2)
        
        # Check if still running
        if process.returncode is None:
            print("✅ MCP server is running")
            
            # Try to send a simple message
            test_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"}
                }
            }
            
            try:
                process.stdin.write((json.dumps(test_message) + '\n').encode())
                await process.stdin.drain()
                
                # Try to read response with timeout
                response_data = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=5.0
                )
                
                if response_data:
                    print(f"✅ MCP server responded: {response_data.decode().strip()[:100]}...")
                else:
                    print("❌ No response from MCP server")
                    
            except asyncio.TimeoutError:
                print("❌ MCP server did not respond within timeout")
            except Exception as e:
                print(f"❌ Error communicating with MCP server: {e}")
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
        print("❌ MCP server startup timed out")
    except Exception as e:
        print(f"❌ MCP server startup failed: {e}")


async def run_hanging_debug():
    """Run all hanging debug tests."""
    print("🚀 Starting Tool Hanging Debug Tests...")
    
    # Test 1: Direct database access
    db_success = await test_database_direct_access()
    
    # Test 2: MCP process state
    await test_mcp_process_state()
    
    # Test 3: Direct MCP server startup
    await test_mcp_server_direct()
    
    # Test 4: Simple tool call (the main test)
    if db_success:
        tool_success = await test_simple_tool_call()
        
        if not tool_success:
            print("\n❌ DIAGNOSIS: Tool call is hanging!")
            print("Possible causes:")
            print("1. MCP server is not responding to requests")
            print("2. Database connection pool is exhausted")
            print("3. Deadlock in the communication protocol")
            print("4. Infinite loop in tool execution")
        else:
            print("\n✅ Tool call completed successfully")
    else:
        print("\n❌ DIAGNOSIS: Database access is the bottleneck!")
    
    print("\n✅ Tool Hanging Debug Tests completed")


if __name__ == "__main__":
    import json
    asyncio.run(run_hanging_debug())
