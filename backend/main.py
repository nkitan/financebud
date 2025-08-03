"""
FinanceBud Backend API
====================

Production-ready FastAPI backend that integrates financial agents with MCP servers 
for comprehensive financial analysis. Features robust error handling, WebSocket support,
and comprehensive monitoring capabilities. Now supports multiple LLM providers!
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import json
import asyncio
import logging
import uuid
from datetime import datetime
import uvicorn
import os
from contextlib import asynccontextmanager

# Import new generic agent instead of LM Studio specific one
from .agents.financial_agent import GenericFinancialAgent, get_financial_agent
from .agents.llm_providers import LLMConfig, ProviderType, get_default_config
from .mcp.client import MCPClientManager
from .config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
mcp_manager = MCPClientManager()
financial_agent = None

# Pydantic models for API
class ChatMessage(BaseModel):
    content: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    agent_state: Dict[str, Any]
    tools_used: List[str]
    execution_time: float
    metadata: Dict[str, Any]

class ServerStatus(BaseModel):
    status: str
    servers: Dict[str, Any]
    agent_available: bool
    lm_studio_available: bool

class ToolCall(BaseModel):
    server_name: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

class ConnectionManager:
    """Manages WebSocket connections for real-time communication."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_sessions: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_sessions[websocket] = session_id
        logger.info(f"WebSocket connected for session: {session_id}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_sessions:
            session_id = self.connection_sessions[websocket]
            del self.connection_sessions[websocket]
            logger.info(f"WebSocket disconnected for session: {session_id}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

connection_manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    global financial_agent
    
    # Startup
    logger.info("🚀 Starting FinanceBud backend...")
    
    try:
        # Initialize MCP servers
        await mcp_manager.initialize_default_servers()
        
        # Initialize the financial agent with default config
        financial_agent = await get_financial_agent()
        
        logger.info(f"✅ Backend started successfully with {financial_agent.config.provider.value} provider!")
    except Exception as e:
        logger.error(f"❌ Failed to start backend: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down FinanceBud backend...")
    try:
        await mcp_manager.close_all()
        logger.info("✅ Backend shutdown completed")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="FinanceBud API",
    description="Production-Ready Financial Analysis Backend with LangGraph Agents and MCP Integration",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (for serving the frontend)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Dependency to ensure agent is available
async def get_agent() -> GenericFinancialAgent:
    if financial_agent is None:
        raise HTTPException(status_code=503, detail="Financial agent not initialized")
    return financial_agent

# Routes
@app.get("/")
async def root():
    """Serve the main frontend page or API info."""
    frontend_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file)
    return {
        "message": "FinanceBud API is running!",
        "status": "healthy",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=ServerStatus)
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Check MCP servers
        mcp_health = await mcp_manager.health_check()
        
        # Check agent
        agent_health = {}
        if financial_agent:
            agent_health = await financial_agent.get_health()
        
        return ServerStatus(
            status="healthy" if mcp_health.get("connected_servers", 0) > 0 else "degraded",
            servers=mcp_health,
            agent_available=financial_agent is not None,
            lm_studio_available=agent_health.get("provider_available", False)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return ServerStatus(
            status="unhealthy",
            servers={},
            agent_available=False,
            lm_studio_available=False
        )

# API Routes - grouped under /api prefix for consistency
@app.post("/chat", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage, agent: GenericFinancialAgent = Depends(get_agent)):
    """Main chat endpoint for financial analysis queries."""
    import time
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        session_id = message.session_id or str(uuid.uuid4())
        
        # Get session history before the new message to track tools used
        history_before = agent.get_session_history(session_id)
        messages_before = len(history_before)
        
        # Process the message using the new chat interface
        response = await agent.chat(message.content, session_id)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Get session history after to track which tools were used
        history_after = agent.get_session_history(session_id)
        
        # Extract tools used from the session history
        tools_used = []
        for msg in history_after[messages_before:]:
            if msg["role"] == "tool":
                # Try to extract tool name from the conversation context
                tools_used.append("financial_tool")  # Generic for now
        
        # If we can access the agent's session directly, get more specific tool names
        if session_id in agent.sessions:
            session_msgs = agent.sessions[session_id]
            for msg in session_msgs:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if 'function' in tool_call and 'name' in tool_call['function']:
                            tool_name = tool_call['function']['name']
                            if tool_name not in tools_used:
                                tools_used.append(tool_name)
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            agent_state={"provider": agent.config.provider.value, "model": agent.config.model},
            tools_used=tools_used,
            execution_time=execution_time,
            metadata={"messages_in_session": len(history_after)}
        )
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/api/financial/summary")
async def get_financial_summary(agent: GenericFinancialAgent = Depends(get_agent)):
    """Get a financial summary using the agent."""
    try:
        # Generate a unique session for this summary request
        session_id = f"summary_{uuid.uuid4()}"
        
        # Ask the agent for a financial summary
        response = await agent.chat("Provide a comprehensive financial summary with current balance, total transactions, and date range of data available.", session_id)
        
        # Try to extract structured data from the response
        # This is a simplified approach - in production you might want more sophisticated parsing
        summary_data = {
            "current_balance_inr": "₹0",
            "total_transactions": 0,
            "date_range": {
                "earliest": "N/A",
                "latest": "N/A"
            },
            "raw_response": response
        }
        
        # Basic parsing to extract numbers if present in response
        import re
        balance_match = re.search(r'₹([\d,]+\.?\d*)', response)
        if balance_match:
            summary_data["current_balance_inr"] = f"₹{balance_match.group(1)}"
        
        transaction_match = re.search(r'(\d+)\s+transactions?', response, re.IGNORECASE)
        if transaction_match:
            summary_data["total_transactions"] = int(transaction_match.group(1).replace(',', ''))
        
        return summary_data
        
    except Exception as e:
        logger.error(f"Financial summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting financial summary: {str(e)}")



@app.get("/tools/list")
async def list_tools_endpoint(agent: GenericFinancialAgent = Depends(get_agent)):
    """List all available MCP tools."""
    try:
        # Return the financial tools available in the agent
        tools = [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in agent.tools
        ]
        return {"tools": tools, "count": len(tools)}
    
    except Exception as e:
        logger.error(f"List tools error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing tools: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session_endpoint(session_id: str, agent: GenericFinancialAgent = Depends(get_agent)):
    """Get session state and conversation history."""
    try:
        history = agent.get_session_history(session_id)
        if not history:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "history": history,
            "message_count": len(history)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session error: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

@app.delete("/sessions/{session_id}")
async def clear_session_endpoint(session_id: str, agent: GenericFinancialAgent = Depends(get_agent)):
    """Clear a specific session."""
    try:
        # Clear session by removing it from the agent's sessions
        if session_id in agent.sessions:
            del agent.sessions[session_id]
        return {"message": f"Session {session_id} cleared successfully"}
    
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

@app.get("/servers")
async def list_servers_endpoint():
    """List all MCP servers and their status."""
    try:
        health = await mcp_manager.health_check()
        return health
    
    except Exception as e:
        logger.error(f"List servers error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing servers: {str(e)}")

@app.post("/servers/{server_name}/reconnect")
async def reconnect_server_endpoint(server_name: str):
    """Reconnect to a specific MCP server."""
    try:
        success = await mcp_manager.reconnect_server(server_name)
        return {
            "server_name": server_name,
            "reconnected": success,
            "message": "Reconnection successful" if success else "Reconnection failed"
        }
    
    except Exception as e:
        logger.error(f"Reconnect server error: {e}")
        raise HTTPException(status_code=500, detail=f"Error reconnecting server: {str(e)}")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    await connection_manager.connect(websocket, session_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)
                content = message_data.get("content", "")
                context = message_data.get("context", {})
                
                if financial_agent and content:
                    # Process message using the new chat interface
                    import time
                    start_time = time.time()
                    
                    response = await financial_agent.chat(content, session_id)
                    execution_time = time.time() - start_time
                    
                    # Get basic tools used information
                    tools_used = []
                    if session_id in financial_agent.sessions:
                        # Try to extract tool usage from recent session activity
                        tools_used = ["financial_analysis"]  # Generic tool name
                    
                    result = {
                        "type": "response",
                        "data": {
                            "response": response,
                            "session_id": session_id,
                            "agent_state": {"provider": financial_agent.config.provider.value},
                            "tools_used": tools_used,
                            "execution_time": execution_time
                        }
                    }
                    
                    # Send response
                    await connection_manager.send_personal_message(
                        json.dumps(result),
                        websocket
                    )
                else:
                    await connection_manager.send_personal_message(
                        json.dumps({
                            "type": "error", 
                            "error": "Agent not available or empty message"
                        }),
                        websocket
                    )
            
            except json.JSONDecodeError:
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "error": "Invalid JSON format"
                    }),
                    websocket
                )
            except Exception as e:
                logger.error(f"WebSocket message processing error: {e}")
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "error": f"Error processing message: {str(e)}"
                    }),
                    websocket
                )
    
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info(f"WebSocket client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)

@app.get("/metrics")
async def metrics_endpoint():
    """Get application metrics and statistics."""
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "active_websocket_connections": len(connection_manager.active_connections),
            "mcp_servers": {},
            "agent_status": "unavailable"
        }
        
        # MCP server metrics
        health = await mcp_manager.health_check()
        metrics["mcp_servers"] = health
        
        # Agent metrics
        if financial_agent:
            agent_health = await financial_agent.get_health()
            metrics["agent_status"] = "available"
            metrics["agent_details"] = agent_health
        
        return metrics
    
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")

# Error handlers
@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

@app.exception_handler(503)
async def service_unavailable_handler(request, exc):
    logger.warning(f"Service unavailable: {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": "Service temporarily unavailable"}
    )

if __name__ == "__main__":
    # Use wsproto instead of websockets to avoid deprecation warnings
    uvicorn.run(
        "backend.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.debug,
        log_level="info",
        ws="wsproto"  # Use wsproto instead of websockets
    )
