"""
Production-Ready MCP Client Manager
===================================

Manages connections to multiple MCP servers and provides a unified interface
for LangGraph agents to interact with different MCP servers. Includes robust
error handling, connection management, and monitoring capabilities.
"""

import asyncio
import json
import logging
import subprocess
import signal
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class MCPServerConnection:
    """Represents a robust connection to a single MCP server."""
    
    def __init__(self, name: str, command: str, args: Optional[List[str]] = None, 
                 env: Optional[Dict[str, str]] = None, working_dir: Optional[str] = None):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.working_dir = working_dir
        self.process: Optional[subprocess.Popen] = None
        self.tools: List[Dict[str, Any]] = []
        self.status = "disconnected"
        self.last_connected: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.connection_attempts = 0
        self.max_retries = 3
        self.retry_delay = 2.0
        self.request_id = 1
        
    async def connect(self) -> bool:
        """Connect to the MCP server with retry logic."""
        for attempt in range(self.max_retries):
            self.connection_attempts += 1
            
            try:
                logger.info(f"Connecting to MCP server: {self.name} (attempt {attempt + 1}/{self.max_retries})")
                
                # Ensure command exists
                if not self._validate_command():
                    self.status = "error"
                    self.last_error = f"Command not found: {self.command}"
                    return False
                
                # Start the MCP server process
                await self._start_process()
                
                # Initialize the MCP connection
                await self._initialize_connection()
                
                # Fetch available tools
                await self._fetch_tools()
                
                self.status = "connected"
                self.last_connected = datetime.now()
                self.last_error = None
                
                logger.info(f"✅ Successfully connected to MCP server: {self.name} with {len(self.tools)} tools")
                return True
                
            except Exception as e:
                error_msg = f"Failed to connect to MCP server {self.name}: {str(e)}"
                logger.error(error_msg)
                self.last_error = error_msg
                
                # Clean up failed connection
                await self._cleanup_process()
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying connection to {self.name} in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.status = "error"
                    return False
        
        return False
    
    def _validate_command(self) -> bool:
        """Validate that the command exists and is executable."""
        if self.command == "python":
            # Check if the Python script exists
            if self.args and len(self.args) > 0:
                script_path = Path(self.args[0])
                return script_path.exists() and script_path.is_file()
            return True
        
        # For other commands, try to find them in PATH
        try:
            import shutil
            return shutil.which(self.command) is not None
        except Exception:
            return False
    
    async def _start_process(self):
        """Start the MCP server process."""
        env = {**self.env} if self.env else None
        
        self.process = subprocess.Popen(
            [self.command] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=self.working_dir,
            preexec_fn=None if hasattr(signal, 'SIGKILL') else None
        )
        
        # Give the process a moment to start
        await asyncio.sleep(0.5)
        
        # Check if process is still running
        if self.process.poll() is not None:
            stderr_output = self.process.stderr.read() if self.process.stderr else "No error output"
            raise Exception(f"Process exited immediately. Error: {stderr_output}")
    
    async def _initialize_connection(self):
        """Initialize the MCP connection with proper handshake."""
        init_request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "financebud-backend",
                    "version": "1.0.0"
                }
            }
        }
        
        # Send initialization
        await self._send_request(init_request)
        
        # Read and validate response
        response = await self._read_response(timeout=10.0)
        
        if "error" in response:
            raise Exception(f"Initialization failed: {response['error']}")
        
        if "result" not in response or "capabilities" not in response["result"]:
            raise Exception("Invalid initialization response")
        
        logger.debug(f"MCP server {self.name} initialized successfully")
    
    async def _fetch_tools(self):
        """Fetch available tools from the MCP server."""
        tools_request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "tools/list"
        }
        
        await self._send_request(tools_request)
        response = await self._read_response(timeout=5.0)
        
        if "error" in response:
            logger.warning(f"Failed to fetch tools from {self.name}: {response['error']}")
            # Try alternative method
            await self._fetch_tools_alternative()
            return
        
        if "result" in response and "tools" in response["result"]:
            self.tools = response["result"]["tools"]
            logger.debug(f"Fetched {len(self.tools)} tools from {self.name}")
        else:
            self.tools = []
            logger.warning(f"No tools found in response from {self.name}")
    
    async def _fetch_tools_alternative(self):
        """Alternative method to fetch tools."""
        try:
            # Try without the params field entirely
            tools_request = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/list",
                "params": None
            }
            
            await self._send_request(tools_request)
            response = await self._read_response(timeout=5.0)
            
            if "result" in response and "tools" in response["result"]:
                self.tools = response["result"]["tools"]
                logger.info(f"Successfully fetched {len(self.tools)} tools from {self.name} using alternative method")
            else:
                # If still no luck, manually define the expected tools
                logger.warning(f"Could not fetch tools from {self.name}, using manual tool definitions")
                self.tools = [
                    {"name": "get_account_summary", "description": "Get account balance and transaction summary"},
                    {"name": "search_transactions", "description": "Search transactions by description pattern"},
                    {"name": "get_transactions_by_date_range", "description": "Get transactions within date range"},
                    {"name": "get_monthly_summary", "description": "Get monthly spending summary"},
                    {"name": "get_spending_by_category", "description": "Analyze spending by category"},
                    {"name": "get_upi_transaction_analysis", "description": "Analyze UPI transactions"},
                    {"name": "find_recurring_payments", "description": "Find recurring payments"},
                    {"name": "analyze_spending_trends", "description": "Analyze spending trends"},
                    {"name": "get_balance_history", "description": "Get account balance history"},
                    {"name": "execute_custom_query", "description": "Execute custom SQL query"}
                ]
        except Exception as e:
            logger.error(f"Alternative tool fetch failed: {e}")
            self.tools = []
    
    async def _send_request(self, request: Dict[str, Any]):
        """Send a JSON-RPC request to the MCP server."""
        if not self.process or not self.process.stdin:
            raise Exception("Process not available for sending request")
        
        request_json = json.dumps(request) + "\n"
        
        try:
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise Exception(f"Failed to send request: {e}")
    
    async def _read_response(self, timeout: float = 30.0) -> Dict[str, Any]:
        """Read a JSON-RPC response from the MCP server."""
        if not self.process or not self.process.stdout:
            raise Exception("Process not available for reading response")
        
        try:
            # Use asyncio to read with timeout
            line = await asyncio.wait_for(
                asyncio.create_task(self._read_line()),
                timeout=timeout
            )
            
            if not line:
                raise Exception("Empty response from server")
            
            return json.loads(line.strip())
            
        except asyncio.TimeoutError:
            raise Exception(f"Timeout waiting for response from {self.name}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {e}")
    
    async def _read_line(self) -> str:
        """Read a line from the process stdout."""
        if not self.process or not self.process.stdout:
            raise Exception("Process stdout not available")
        
        # Use executor to avoid blocking
        loop = asyncio.get_event_loop()
        line = await loop.run_in_executor(None, self.process.stdout.readline)
        return line
    
    def _get_next_id(self) -> int:
        """Get the next request ID."""
        request_id = self.request_id
        self.request_id += 1
        return request_id
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool on this MCP server."""
        if self.status != "connected":
            raise Exception(f"Server {self.name} is not connected")
        
        # Try different tool call formats
        tool_requests = [
            # Standard MCP format
            {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            },
            # Alternative format without nested arguments
            {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name
                }
            },
            # FastMCP specific format
            {
                "jsonrpc": "2.0",
                "id": self._get_next_id(),
                "method": tool_name,
                "params": arguments
            }
        ]
        
        for i, tool_request in enumerate(tool_requests):
            try:
                await self._send_request(tool_request)
                response = await self._read_response()
                
                if "error" not in response:
                    logger.debug(f"Tool call successful using format {i+1}")
                    return response.get("result", {})
                else:
                    logger.debug(f"Tool call format {i+1} failed: {response['error']}")
                    
            except Exception as e:
                logger.debug(f"Tool call format {i+1} exception: {e}")
                continue
        
        # If all formats failed
        raise Exception(f"All tool call formats failed for {tool_name}")
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        self.status = "disconnecting"
        await self._cleanup_process()
        self.status = "disconnected"
        logger.info(f"Disconnected from MCP server: {self.name}")
    
    async def _cleanup_process(self):
        """Clean up the server process."""
        if self.process:
            try:
                if self.process.poll() is None:  # Process is still running
                    self.process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        await asyncio.wait_for(
                            asyncio.create_task(self._wait_for_process()),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        # Force kill if it doesn't terminate gracefully
                        if hasattr(self.process, 'kill'):
                            self.process.kill()
                        
                self.process = None
                
            except Exception as e:
                logger.warning(f"Error cleaning up process for {self.name}: {e}")
    
    async def _wait_for_process(self):
        """Wait for the process to terminate."""
        if self.process:
            while self.process.poll() is None:
                await asyncio.sleep(0.1)
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information."""
        return {
            "name": self.name,
            "status": self.status,
            "last_connected": self.last_connected.isoformat() if self.last_connected else None,
            "last_error": self.last_error,
            "connection_attempts": self.connection_attempts,
            "tools_count": len(self.tools),
            "process_running": self.process is not None and self.process.poll() is None
        }

class MCPClientManager:
    """Manages multiple MCP server connections."""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConnection] = {}
        self.default_servers_initialized = False
        
    async def add_server(self, name: str, command: str, args: Optional[List[str]] = None,
                        env: Optional[Dict[str, str]] = None, working_dir: Optional[str] = None) -> bool:
        """Add and connect to a new MCP server."""
        if name in self.servers:
            logger.warning(f"Server {name} already exists, disconnecting old connection")
            await self.disconnect_server(name)
        
        server = MCPServerConnection(name, command, args, env, working_dir)
        self.servers[name] = server
        
        return await server.connect()
    
    async def disconnect_server(self, name: str):
        """Disconnect from a specific server."""
        if name in self.servers:
            await self.servers[name].disconnect()
            del self.servers[name]
    
    async def reconnect_server(self, name: str) -> bool:
        """Reconnect to a specific server."""
        if name in self.servers:
            return await self.servers[name].connect()
        return False
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on a specific server."""
        if server_name not in self.servers:
            raise Exception(f"Server {server_name} not found")
        
        server = self.servers[server_name]
        if server.status != "connected":
            # Try to reconnect
            logger.info(f"Server {server_name} not connected, attempting reconnection...")
            if not await server.connect():
                raise Exception(f"Server {server_name} is not available")
        
        return await server.call_tool(tool_name, arguments)
    
    async def get_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available tools from all connected servers."""
        all_tools = {}
        for name, server in self.servers.items():
            if server.status == "connected":
                all_tools[name] = server.tools
            else:
                all_tools[name] = []
        return all_tools
    
    async def initialize_default_servers(self):
        """Initialize default MCP servers."""
        if self.default_servers_initialized:
            return
        
        # Financial data server with virtual environment Python
        financial_server_path = "/home/notroot/Work/financebud/mcp_server.py"
        python_path = "/home/notroot/Work/financebud/.venv/bin/python"
        
        if Path(financial_server_path).exists():
            success = await self.add_server(
                name="financial-data-inr",
                command=python_path,
                args=[financial_server_path],
                working_dir="/home/notroot/Work/financebud"
            )
            if success:
                logger.info("✅ Financial data MCP server connected successfully")
            else:
                logger.error("❌ Failed to connect to financial data MCP server")
        else:
            logger.error(f"❌ Financial data server script not found at {financial_server_path}")
        
        self.default_servers_initialized = True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of all servers."""
        health = {
            "total_servers": len(self.servers),
            "connected_servers": 0,
            "servers": {}
        }
        
        for name, server in self.servers.items():
            server_status = server.get_status()
            health["servers"][name] = server_status
            if server_status["status"] == "connected":
                health["connected_servers"] += 1
        
        health["overall_status"] = "healthy" if health["connected_servers"] > 0 else "unhealthy"
        return health
    
    async def close_all(self):
        """Close all server connections."""
        for name, server in self.servers.items():
            try:
                await server.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")
        
        self.servers.clear()
        logger.info("All MCP servers disconnected")
    
    def get_server_status(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific server."""
        if server_name in self.servers:
            return self.servers[server_name].get_status()
        return None
    
    def list_servers(self) -> List[str]:
        """List all server names."""
        return list(self.servers.keys())
