#!/usr/bin/env python3
"""
Direct MCP Server Test
=====================

Test the MCP server directly to see where it hangs.
"""

import json
import subprocess
import sys
import os
import asyncio
import time

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


async def test_mcp_server_direct():
    """Test MCP server directly via subprocess."""
    print("🔍 Testing MCP server direct communication...")
    
    # Start MCP server process
    process = subprocess.Popen(
        [sys.executable, '/home/notroot/Work/financebud/mcp_server.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd='/home/notroot/Work/financebud'
    )
    
    print("📡 MCP server process started, waiting for initialization...")
    await asyncio.sleep(2)  # Wait for server to start
    
    try:
        # Check if process started correctly
        if not process.stdin or not process.stdout:
            print("❌ Failed to start MCP server process")
            return False
            
        # Send initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        print("📤 Sending initialization request...")
        init_json = json.dumps(init_request) + "\n"
        process.stdin.write(init_json)
        process.stdin.flush()
        
        # Read initialization response
        print("📥 Reading initialization response...")
        start_time = time.time()
        try:
            response_line = await asyncio.wait_for(
                asyncio.to_thread(process.stdout.readline),
                timeout=5.0
            )
            elapsed = time.time() - start_time
            print(f"✅ Initialization response received in {elapsed:.2f}s")
            
            if response_line:
                response = json.loads(response_line.strip())
                print(f"Response: {response}")
            else:
                print("❌ Empty response from server")
                return False
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ Initialization timed out after {elapsed:.2f}s")
            return False
        
        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        print("📤 Sending initialized notification...")
        init_notify_json = json.dumps(initialized_notification) + "\n"
        process.stdin.write(init_notify_json)
        process.stdin.flush()
        
        # Wait a moment
        await asyncio.sleep(0.5)
        
        # Send tool call request
        tool_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "find_recurring_payments",
                "arguments": {"days_back": 1200}
            }
        }
        
        print("📤 Sending tool call request (find_recurring_payments with days_back=1200)...")
        tool_json = json.dumps(tool_request) + "\n"
        process.stdin.write(tool_json)
        process.stdin.flush()
        
        # Read tool response with timeout
        print("📥 Reading tool response...")
        start_time = time.time()
        try:
            response_line = await asyncio.wait_for(
                asyncio.to_thread(process.stdout.readline),
                timeout=15.0  # 15 second timeout
            )
            elapsed = time.time() - start_time
            print(f"✅ Tool response received in {elapsed:.2f}s")
            
            if response_line:
                response = json.loads(response_line.strip())
                print(f"Response: {json.dumps(response, indent=2)[:200]}...")
                return True
            else:
                print("❌ Empty tool response from server")
                return False
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ Tool call timed out after {elapsed:.2f}s")
            
            # Check if process is still running
            if process.poll() is None:
                print("📊 Process still running - tool execution hanging")
            else:
                print(f"💀 Process died with return code: {process.poll()}")
                if process.stderr:
                    stderr_output = process.stderr.read()
                    if stderr_output:
                        print(f"Error output: {stderr_output}")
            
            return False
            
    finally:
        # Clean up process
        if process.poll() is None:
            print("🧹 Terminating MCP server process...")
            process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(process.wait),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                print("🔪 Force killing MCP server process...")
                process.kill()


if __name__ == "__main__":
    asyncio.run(test_mcp_server_direct())
