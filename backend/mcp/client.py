"""
MCP Client Manager
==================

High-performance MCP client that maintains persistent connections to MCP servers
throughout the application lifecycle. This eliminates the overhead of repeatedly
starting and stopping MCP server processes.

Key features:
- Persistent server connections
- Connection pooling and reuse
- Health monitoring with auto-reconnection
- Request pipelining and batching
- Connection state caching
- Optimized tool discovery
"""

import asyncio
import json
import logging
import os
import subprocess
import signal
import sys
import time
import weakref
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

# Import centralized logging
from ..logging_config import get_logger_with_context

logger = get_logger_with_context(__name__)

# Global timeout settings
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_HEALTH_CHECK_INTERVAL = 30.0
DEFAULT_RETRY_DELAY = 2.0

class ConnectionState(Enum):
    """Connection states for MCP servers."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None
    max_retries: int = 3
    retry_delay: float = DEFAULT_RETRY_DELAY
    health_check_interval: float = DEFAULT_HEALTH_CHECK_INTERVAL
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    auto_reconnect: bool = True

@dataclass
class ToolCallRequest:
    """Represents a tool call request."""
    tool_name: str
    arguments: Dict[str, Any]
    request_id: str
    timeout: float = DEFAULT_REQUEST_TIMEOUT
    callback: Optional[Callable] = None

class MCPConnection:
    """A high-performance connection to a single MCP server with advanced features."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.process: Optional[subprocess.Popen] = None
        self.tools: Dict[str, Dict[str, Any]] = {}  # Tool name -> tool definition
        self.last_connected: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.connection_attempts = 0
        self.request_id = 1
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.response_futures: Dict[int, asyncio.Future] = {}
        self.health_check_task: Optional[asyncio.Task] = None
        self.request_handler_task: Optional[asyncio.Task] = None
        self.read_task: Optional[asyncio.Task] = None
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "last_request_time": None,
            "uptime_start": None
        }
        self._lock = asyncio.Lock()
        
    async def start(self) -> bool:
        """Start the persistent connection and background tasks."""
        async with self._lock:
            if self.state not in (ConnectionState.DISCONNECTED, ConnectionState.ERROR):
                logger.warning(f"Server {self.config.name} is already starting/started")
                return self.state == ConnectionState.CONNECTED
            
            self.state = ConnectionState.CONNECTING
            
            try:
                # Start the server process
                await self._start_process()
                
                # Initialize the connection
                await self._initialize_connection()
                
                # Fetch tools
                await self._fetch_tools()
                
                # Start background tasks
                self._start_background_tasks()
                
                self.state = ConnectionState.CONNECTED
                self.last_connected = datetime.now()
                self.stats["uptime_start"] = datetime.now()
                self.last_error = None
                
                logger.info(f"✅ Persistent connection established to {self.config.name} with {len(self.tools)} tools")
                return True
                
            except Exception as e:
                error_msg = f"Failed to start persistent connection to {self.config.name}: {e}"
                logger.error(error_msg)
                self.last_error = error_msg
                self.state = ConnectionState.ERROR
                await self._cleanup_process()
                return False
    
    async def _start_process(self):
        """Start the MCP server process with optimized settings."""
        env = {**self.config.env} if self.config.env else None
        
        # Add Python path optimization for faster startup
        if env is None:
            env = {}
        env.update({
            "PYTHONUNBUFFERED": "1",  # Disable Python output buffering
            "PYTHONDONTWRITEBYTECODE": "1"  # Skip .pyc file creation
        })
        
        self.process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self.config.working_dir
        )
        
        # Wait for process to stabilize
        await asyncio.sleep(0.2)
        
        # Check if process is still running
        if self.process.returncode is not None:
            stderr_output = await self.process.stderr.read() if self.process.stderr else b""
            raise Exception(f"Process exited immediately. Error: {stderr_output.decode('utf-8', errors='ignore')}")
    
    async def _initialize_connection(self):
        """Initialize the MCP connection with optimized handshake."""
        init_request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "clientInfo": {
                    "name": "financebud-persistent-client",
                    "version": "2.0.0"
                }
            }
        }
        
        # Send initialization with shorter timeout for faster failure detection
        await self._send_request_direct(init_request)
        response = await self._read_response_direct(timeout=DEFAULT_REQUEST_TIMEOUT)
        
        if "error" in response:
            raise Exception(f"Initialization failed: {response['error']}")
        
        if "result" not in response or "capabilities" not in response["result"]:
            raise Exception("Invalid initialization response")
        
        # Send the initialized notification (required by MCP protocol)
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        await self._send_request_direct(initialized_notification)
        
        logger.debug(f"MCP server {self.config.name} initialized successfully")
    
    async def _fetch_tools(self):
        """Fetch and cache available tools."""
        tools_request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "tools/list",
            "params": {}  # MCP requires params field, even if empty
        }
        
        await self._send_request_direct(tools_request)
        response = await self._read_response_direct(timeout=DEFAULT_REQUEST_TIMEOUT)
        
        if "error" in response:
            logger.warning(f"Failed to fetch tools from {self.config.name}: {response['error']}")
            # Use fallback tool definitions
            self._setup_fallback_tools()
            return
        
        if "result" in response and "tools" in response["result"]:
            # Convert tool list to dictionary for faster lookup
            self.tools = {
                tool["name"]: tool for tool in response["result"]["tools"]
            }
            logger.debug(f"Cached {len(self.tools)} tools from {self.config.name}")
        else:
            logger.warning(f"No tools found in response from {self.config.name}")
            self._setup_fallback_tools()
    
    def _setup_fallback_tools(self):
        """Setup fallback tool definitions if discovery fails."""
        fallback_tools = [
            {"name": "get_account_summary", "description": "Get account balance and transaction summary"},
            {"name": "get_recent_transactions", "description": "Get recent transactions"},
            {"name": "search_transactions", "description": "Search transactions by pattern"},
            {"name": "get_transactions_by_date_range", "description": "Get transactions within date range"},
            {"name": "get_monthly_summary", "description": "Get monthly spending summary"},
            {"name": "get_spending_by_category", "description": "Analyze spending by category"},
            {"name": "get_upi_transaction_analysis", "description": "Analyze UPI transactions"},
            {"name": "find_recurring_payments", "description": "Find recurring payments"},
            {"name": "analyze_spending_trends", "description": "Analyze spending trends"},
            {"name": "get_balance_history", "description": "Get account balance history"},
            {"name": "execute_custom_query", "description": "Execute custom SQL query"},
            {"name": "get_database_schema", "description": "Get database schema information"}
        ]
        
        self.tools = {tool["name"]: tool for tool in fallback_tools}
        logger.info(f"Using fallback tool definitions for {self.config.name}: {len(self.tools)} tools")
    
    def _start_background_tasks(self):
        """Start background tasks for handling requests and health checks."""
        # Start request handler
        self.request_handler_task = asyncio.create_task(self._request_handler())
        
        # Start response reader
        self.read_task = asyncio.create_task(self._response_reader())
        
        # Start health check if enabled
        if self.config.auto_reconnect:
            self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _request_handler(self):
        """Handle outgoing requests in a queue."""
        while self.state != ConnectionState.SHUTTING_DOWN:
            try:
                # Wait for requests with timeout to allow periodic checks
                request = await asyncio.wait_for(
                    self.request_queue.get(), 
                    timeout=1.0
                )
                
                if request is None:  # Shutdown signal
                    break
                
                await self._process_request(request)
                
            except asyncio.TimeoutError:
                continue  # Normal timeout, check state and continue
            except Exception as e:
                logger.error(f"Request handler error for {self.config.name}: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_request(self, request: ToolCallRequest):
        """Process a single tool call request."""
        start_time = time.time()
        request_id = self._get_next_id()
        
        try:
            # Prepare the JSON-RPC request
            rpc_request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": request.tool_name,
                    "arguments": request.arguments
                }
            }
            
            # Create a future for the response
            response_future = asyncio.Future()
            self.response_futures[request_id] = response_future
            
            # Send the request
            await self._send_request_direct(rpc_request)
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(response_future, timeout=request.timeout)
                
                execution_time = time.time() - start_time
                self._update_stats(True, execution_time)
                
                if request.callback:
                    await request.callback(response, None)
                
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                self._update_stats(False, execution_time)
                error = f"Request timeout after {request.timeout}s"
                
                if request.callback:
                    await request.callback(None, error)
                
            finally:
                # Clean up the future
                self.response_futures.pop(request_id, None)
                
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_stats(False, execution_time)
            
            if request.callback:
                await request.callback(None, str(e))
    
    async def _response_reader(self):
        """Read responses from the MCP server."""
        while self.state != ConnectionState.SHUTTING_DOWN:
            try:
                if not self.process or not self.process.stdout:
                    await asyncio.sleep(1.0)
                    continue
                
                # Read a line with timeout (should be longer than typical tool execution time)
                line = await asyncio.wait_for(
                    self._read_line_async(),
                    timeout=DEFAULT_REQUEST_TIMEOUT  # Increased timeout to handle database queries
                )
                
                if not line:
                    continue
                
                try:
                    response = json.loads(line.strip())
                    request_id = response.get("id")
                    
                    if request_id in self.response_futures:
                        future = self.response_futures[request_id]
                        if not future.done():
                            future.set_result(response)
                
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON response from {self.config.name}: {line[:100]}")
                
            except asyncio.TimeoutError:
                # If no response received in 30 seconds, continue
                # This indicates the server might be unresponsive
                continue
            except Exception as e:
                logger.error(f"Response reader error for {self.config.name}: {e}")
                await asyncio.sleep(1.0)
    
    async def _health_check_loop(self):
        """Periodic health check and reconnection."""
        while self.state != ConnectionState.SHUTTING_DOWN:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                if self.state == ConnectionState.CONNECTED:
                    # Check if process is still alive
                    if not self.process or self.process.returncode is not None:
                        logger.warning(f"MCP server process for {self.config.name} died, reconnecting...")
                        await self._reconnect()
                
            except Exception as e:
                logger.error(f"Health check error for {self.config.name}: {e}")
    
    async def _reconnect(self):
        """Reconnect to the MCP server."""
        async with self._lock:
            if self.state == ConnectionState.SHUTTING_DOWN:
                return
            
            self.state = ConnectionState.RECONNECTING
            logger.info(f"Reconnecting to {self.config.name}...")
            
            # Clean up current connection
            await self._cleanup_process()
            
            # Attempt reconnection
            for attempt in range(self.config.max_retries):
                try:
                    await self._start_process()
                    await self._initialize_connection()
                    await self._fetch_tools()
                    
                    self.state = ConnectionState.CONNECTED
                    self.last_connected = datetime.now()
                    self.last_error = None
                    
                    logger.info(f"✅ Reconnected to {self.config.name}")
                    return
                    
                except Exception as e:
                    logger.warning(f"Reconnection attempt {attempt + 1} failed for {self.config.name}: {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
            
            self.state = ConnectionState.ERROR
            self.last_error = f"Failed to reconnect after {self.config.max_retries} attempts"
            logger.error(f"❌ Failed to reconnect to {self.config.name}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: float = DEFAULT_REQUEST_TIMEOUT) -> Any:
        """Call a tool asynchronously with optimized handling."""
        if self.state != ConnectionState.CONNECTED:
            if self.config.auto_reconnect and self.state != ConnectionState.RECONNECTING:
                logger.info(f"Server {self.config.name} not connected, attempting reconnection...")
                await self._reconnect()
                if self.state != ConnectionState.CONNECTED:
                    raise Exception(f"Server {self.config.name} is not available")
            else:
                raise Exception(f"Server {self.config.name} is not connected")
        
        # Check if tool exists (fast lookup in cached tools)
        if tool_name not in self.tools:
            raise Exception(f"Tool {tool_name} not found on server {self.config.name}")
        
        # Create response future and queue the request
        response_future = asyncio.Future()
        
        async def callback(response, error):
            if error:
                response_future.set_exception(Exception(error))
            else:
                if "error" in response:
                    response_future.set_exception(Exception(response["error"]))
                else:
                    response_future.set_result(response.get("result", {}))
        
        request = ToolCallRequest(
            tool_name=tool_name,
            arguments=arguments,
            request_id=str(self._get_next_id()),
            timeout=timeout,
            callback=callback
        )
        
        await self.request_queue.put(request)
        return await response_future
    
    async def _send_request_direct(self, request: Dict[str, Any]):
        """Send a request directly (for initialization and health checks)."""
        if not self.process or not self.process.stdin:
            raise Exception("Process not available for sending request")
        
        request_json = json.dumps(request) + "\n"
        
        try:
            self.process.stdin.write(request_json.encode('utf-8'))
            await self.process.stdin.drain()
        except (BrokenPipeError, OSError) as e:
            raise Exception(f"Failed to send request: {e}")
    
    async def _read_response_direct(self, timeout: float = 60.0) -> Dict[str, Any]:
        """Read a response directly (for initialization and health checks)."""
        if not self.process or not self.process.stdout:
            raise Exception("Process not available for reading response")
        
        try:
            line = await asyncio.wait_for(
                self._read_line_async(),
                timeout=timeout
            )
            
            if not line:
                raise Exception("Empty response from server")
            
            return json.loads(line.strip())
            
        except asyncio.TimeoutError:
            raise Exception(f"Timeout waiting for response from {self.config.name}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {e}")
    
    async def _read_line_async(self) -> str:
        """Read a line asynchronously."""
        if not self.process or not self.process.stdout:
            raise Exception("Process stdout not available")
        
        line = await self.process.stdout.readline()
        return line.decode('utf-8')
    
    def _get_next_id(self) -> int:
        """Get the next request ID."""
        request_id = self.request_id
        self.request_id += 1
        return request_id
    
    def _update_stats(self, success: bool, execution_time: float):
        """Update performance statistics."""
        self.stats["total_requests"] += 1
        self.stats["last_request_time"] = datetime.now()
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # Update average response time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if self.stats["avg_response_time"] == 0:
            self.stats["avg_response_time"] = execution_time
        else:
            self.stats["avg_response_time"] = (
                alpha * execution_time + 
                (1 - alpha) * self.stats["avg_response_time"]
            )
    
    async def _cleanup_process(self):
        """Clean up the server process and tasks."""
        # Cancel background tasks
        for task in [self.health_check_task, self.request_handler_task, self.read_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clear response futures
        for future in self.response_futures.values():
            if not future.done():
                future.set_exception(Exception("Connection closed"))
        self.response_futures.clear()
        
        # Terminate process
        if self.process:
            try:
                if self.process.returncode is None:
                    self.process.terminate()
                    
                    try:
                        await asyncio.wait_for(
                            self._wait_for_process(),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        if hasattr(self.process, 'kill'):
                            self.process.kill()
                        
                self.process = None
                
            except Exception as e:
                logger.warning(f"Error cleaning up process for {self.config.name}: {e}")
    
    async def _wait_for_process(self):
        """Wait for the process to terminate."""
        if self.process:
            await self.process.wait()
    
    async def stop(self):
        """Stop the persistent connection."""
        async with self._lock:
            if self.state == ConnectionState.SHUTTING_DOWN:
                return
            
            logger.info(f"Stopping persistent connection to {self.config.name}")
            self.state = ConnectionState.SHUTTING_DOWN
            
            # Signal request handler to stop
            await self.request_queue.put(None)
            
            # Clean up everything
            await self._cleanup_process()
            
            self.state = ConnectionState.DISCONNECTED
            logger.info(f"Stopped persistent connection to {self.config.name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information."""
        uptime = None
        if self.stats["uptime_start"]:
            uptime = (datetime.now() - self.stats["uptime_start"]).total_seconds()
        
        return {
            "name": self.config.name,
            "state": self.state.value,
            "last_connected": self.last_connected.isoformat() if self.last_connected else None,
            "last_error": self.last_error,
            "connection_attempts": self.connection_attempts,
            "tools_count": len(self.tools),
            "tools": list(self.tools.keys()),
            "process_running": self.process is not None and self.process.returncode is None,
            "uptime_seconds": uptime,
            "stats": self.stats.copy(),
            "pending_requests": self.request_queue.qsize(),
            "active_requests": len(self.response_futures)
        }


class MCPManager:
    """Manages multiple high-performance MCP connections."""
    
    def __init__(self):
        self.connections: Dict[str, MCPConnection] = {}
        self.initialized = False
        self._lock = asyncio.Lock()
    
    async def add_server(self, config: MCPServerConfig) -> bool:
        """Add and start a new MCP server connection."""
        async with self._lock:
            if config.name in self.connections:
                logger.warning(f"Server {config.name} already exists, stopping old connection")
                await self.remove_server(config.name)
            
            connection = MCPConnection(config)
            success = await connection.start()
            
            if success:
                self.connections[config.name] = connection
                logger.info(f"✅ Added persistent server: {config.name}")
            else:
                logger.error(f"❌ Failed to add persistent server: {config.name}")
            
            return success
    
    async def remove_server(self, name: str):
        """Remove and stop a server connection."""
        async with self._lock:
            if name in self.connections:
                await self.connections[name].stop()
                del self.connections[name]
                logger.info(f"Removed server: {name}")
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any], timeout: float = DEFAULT_REQUEST_TIMEOUT) -> Any:
        """Call a tool on a specific server."""
        if server_name not in self.connections:
            raise Exception(f"Server {server_name} not found")
        
        connection = self.connections[server_name]
        return await connection.call_tool(tool_name, arguments, timeout)
    
    async def initialize_default_servers(self):
        """Initialize default MCP servers with optimized settings."""
        if self.initialized:
            return

        app_home = os.getcwd()
        python_executable = sys.executable
        mcp_server_script = os.path.join(app_home, "mcp_server.py")

        # Financial data server configuration
        financial_config = MCPServerConfig(
            name="financial-data-inr",
            command=python_executable,
            args=[mcp_server_script],
            working_dir=app_home,
            max_retries=3,
            retry_delay=1.0,
            health_check_interval=60.0,  # Check every minute
            request_timeout=60.0,  # Faster timeout for financial queries
            auto_reconnect=True
        )

        if Path(financial_config.args[0]).exists():
            success = await self.add_server(financial_config)
            if success:
                logger.info("✅ Persistent financial data MCP server initialized")
            else:
                logger.error("❌ Failed to initialize persistent financial data MCP server")
        else:
            logger.error(f"❌ Financial data server script not found at {financial_config.args[0]}")

        self.initialized = True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of all servers."""
        health = {
            "total_servers": len(self.connections),
            "connected_servers": 0,
            "servers": {}
        }
        
        for name, connection in self.connections.items():
            server_status = connection.get_status()
            health["servers"][name] = server_status
            if server_status["state"] == "connected":
                health["connected_servers"] += 1
        
        health["overall_status"] = "healthy" if health["connected_servers"] > 0 else "unhealthy"
        return health
    
    async def get_all_tools(self) -> Dict[str, List[str]]:
        """Get all available tools from all connected servers."""
        all_tools = {}
        for name, connection in self.connections.items():
            if connection.state == ConnectionState.CONNECTED:
                all_tools[name] = list(connection.tools.keys())
            else:
                all_tools[name] = []
        return all_tools
    
    def get_server_status(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific server."""
        if server_name in self.connections:
            return self.connections[server_name].get_status()
        return None
    
    def list_servers(self) -> List[str]:
        """List all server names."""
        return list(self.connections.keys())
    
    async def shutdown(self):
        """Shutdown all connections."""
        async with self._lock:
            for name, connection in self.connections.items():
                try:
                    await connection.stop()
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")
            
            self.connections.clear()
            logger.info("All persistent MCP connections shutdown")


# Global MCP manager instance
mcp_manager = None

async def get_mcp_manager() -> MCPManager:
    """Get or create the global MCP manager."""
    global mcp_manager
    
    if mcp_manager is None:
        mcp_manager = MCPManager()
        await mcp_manager.initialize_default_servers()
    
    return mcp_manager
