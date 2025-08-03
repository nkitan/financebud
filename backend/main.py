"""
FinanceBud Backend API
====================

High-performance FastAPI backend with integrated financial agents and MCP servers 
for comprehensive financial analysis. Features persistent connections, connection pooling,
advanced caching, and optimized database operations.

Key Performance Features:
- Persistent MCP connections with health monitoring
- Database connection pooling with optimized SQLite configuration
- LRU caching with TTL for query results
- Parallel tool call processing
- WebSocket support for real-time updates
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

# Import high-performance components
from .agents.financial_agent import FinancialAgent, get_financial_agent
from .agents.llm_providers import LLMConfig, ProviderType, get_default_config
from .mcp.client import MCPManager, get_mcp_manager
from .config import config

# Import centralized logging configuration
from .logging_config import setup_logging, get_logger_with_context, ContextLogger

# Configure logging with optimized settings
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=os.getenv("LOG_DIR", "logs")
)

logger: ContextLogger = get_logger_with_context(__name__)

# Global application state
financial_agent: Optional[FinancialAgent] = None
mcp_manager: Optional[MCPManager] = None

# API Models
class ChatMessage(BaseModel):
    content: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    processing_time: float
    tool_calls: List[str] = []

class HealthResponse(BaseModel):
    status: str
    mcp_health: Dict[str, Any]
    agent_status: str
    performance_metrics: Dict[str, Any]

class ToolListResponse(BaseModel):
    tools: List[Dict[str, Any]]
    server_count: int

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New WebSocket connection established")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket connection closed")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with optimized startup and shutdown."""
    global financial_agent, mcp_manager
    
    # Startup
    logger.info("🚀 Starting FinanceBud backend with performance optimizations...")
    
    try:
        # Initialize high-performance MCP manager
        mcp_manager = await get_mcp_manager()
        await mcp_manager.initialize_default_servers()
        
        # Initialize financial agent
        financial_agent = await get_financial_agent()
        
        logger.info(f"✅ Backend started successfully with {financial_agent.llm_config.provider.value} provider!")
        logger.info("🔥 Performance features enabled: persistent connections, connection pooling, caching")
    except Exception as e:
        logger.error(f"❌ Failed to start backend: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down FinanceBud backend...")
    try:
        if mcp_manager:
            await mcp_manager.shutdown()
        logger.info("✅ Backend shutdown completed")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Initialize FastAPI app with optimized configuration
app = FastAPI(
    title="FinanceBud API",
    description="High-Performance Financial Analysis Backend with AI Agents and MCP Integration",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware with optimized settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for serving the frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Dependency to ensure agent is available
async def get_agent() -> FinancialAgent:
    if financial_agent is None:
        raise HTTPException(status_code=503, detail="Financial agent not initialized")
    return financial_agent

# Routes
@app.get("/")
async def serve_frontend():
    """Serve the main frontend application."""
    frontend_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file)
    return {"message": "FinanceBud API", "version": "3.0.0", "status": "running"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with performance metrics."""
    try:
        # MCP health check
        mcp_health = {}
        if mcp_manager:
            mcp_health = await mcp_manager.health_check()
        
        # Agent performance metrics
        agent_metrics = {}
        if financial_agent:
            agent_metrics = financial_agent.get_metrics()
        
        return HealthResponse(
            status="healthy",
            mcp_health=mcp_health,
            agent_status="ready" if financial_agent else "not_initialized",
            performance_metrics={
                "agent_metrics": agent_metrics,
                "websocket_connections": len(manager.active_connections),
                "mcp_servers": len(mcp_health.get("servers", {}))
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage, agent: FinancialAgent = Depends(get_agent)):
    """Process chat messages with performance tracking."""
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing chat message: {message.content[:100]}...")
        
        # Process message with the agent
        response = await agent.process_message(message.content)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Chat processed in {processing_time:.3f}s")
        
        return ChatResponse(
            response=response,
            session_id=agent.session_id,
            processing_time=processing_time,
            tool_calls=[]  # Could be extracted from agent if needed
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Chat error: {e} (processing time: {processing_time:.3f}s)")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/tools", response_model=ToolListResponse)
async def list_tools_endpoint(agent: FinancialAgent = Depends(get_agent)):
    """List all available financial analysis tools."""
    try:
        tools = agent.get_openai_tools()
        
        # Get MCP server information
        server_count = 0
        if mcp_manager:
            servers = await mcp_manager.get_all_tools()
            server_count = len(servers)
        
        return ToolListResponse(
            tools=tools,
            server_count=server_count
        )
        
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing tools: {str(e)}")

@app.get("/session/{session_id}")
async def get_session_endpoint(session_id: str, agent: FinancialAgent = Depends(get_agent)):
    """Get session state and conversation history."""
    try:
        return {
            "session_id": session_id,
            "metrics": agent.get_metrics(),
            "conversation_length": len(agent.conversation_history)
        }
    
    except Exception as e:
        logger.error(f"Get session error: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

@app.delete("/session/{session_id}")
async def clear_session_endpoint(session_id: str, agent: FinancialAgent = Depends(get_agent)):
    """Clear session conversation history."""
    try:
        agent.clear_conversation()
        logger.info(f"Cleared session {session_id}")
        
        return {
            "session_id": session_id, 
            "status": "cleared",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

@app.get("/mcp/health")
async def mcp_health_endpoint():
    """Get detailed MCP server health information."""
    try:
        if not mcp_manager:
            raise HTTPException(status_code=503, detail="MCP manager not initialized")
            
        health = await mcp_manager.health_check()
        return health
        
    except Exception as e:
        logger.error(f"MCP health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"MCP health check failed: {str(e)}")

@app.get("/metrics")
async def metrics_endpoint():
    """Get comprehensive performance metrics."""
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "application": {
                "status": "running",
                "websocket_connections": len(manager.active_connections)
            }
        }
        
        # Agent metrics
        if financial_agent:
            metrics["agent"] = financial_agent.get_metrics()
        
        # MCP metrics
        if mcp_manager:
            mcp_health = await mcp_manager.health_check()
            metrics["mcp"] = mcp_health
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

# WebSocket endpoint for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat interface."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                content = message_data.get("content", "")
                
                if content.strip():
                    # Process with financial agent
                    if financial_agent:
                        start_time = datetime.now()
                        response = await financial_agent.process_message(content)
                        processing_time = (datetime.now() - start_time).total_seconds()
                        
                        # Send response back to client
                        response_data = {
                            "type": "response",
                            "content": response,
                            "session_id": financial_agent.session_id,
                            "processing_time": processing_time,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        await manager.send_personal_message(
                            json.dumps(response_data), 
                            websocket
                        )
                    else:
                        # Agent not available
                        error_response = {
                            "type": "error",
                            "content": "Financial agent not available",
                            "timestamp": datetime.now().isoformat()
                        }
                        await manager.send_personal_message(
                            json.dumps(error_response), 
                            websocket
                        )
                        
            except json.JSONDecodeError:
                # Handle invalid JSON
                error_response = {
                    "type": "error",
                    "content": "Invalid message format",
                    "timestamp": datetime.now().isoformat()
                }
                await manager.send_personal_message(
                    json.dumps(error_response), 
                    websocket
                )
                
            except Exception as e:
                # Handle processing errors
                logger.error(f"WebSocket processing error: {e}")
                error_response = {
                    "type": "error",
                    "content": f"Error processing message: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                await manager.send_personal_message(
                    json.dumps(error_response), 
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured logging."""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with structured logging."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
