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

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage, agent: FinancialAgent = Depends(get_agent)):
    """Handle chat messages via HTTP when WebSocket is not available."""
    try:
        start_time = datetime.now()
        
        # Process the message with the financial agent
        response = await agent.process_message(message.content)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Get session ID (use provided or generate new one)
        session_id = message.session_id or str(uuid.uuid4())
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            processing_time=processing_time,
            tool_calls=[]  # TODO: Extract tool calls from agent response if needed
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat message: {str(e)}"
        )

@app.get("/api/financial-summary")
async def get_financial_summary():
    """Get financial summary directly from the database."""
    try:
        # Import database manager directly
        from .database.db import get_db_manager
        
        db_manager = get_db_manager()
        
        # Get total transactions count
        total_result = db_manager.execute_query(
            "SELECT COUNT(*) as count FROM transactions",
            cache_ttl=30
        )
        total_transactions = total_result.data[0]['count']
        
        # Get date range
        date_range_result = db_manager.execute_query(
            "SELECT MIN(transaction_date) as earliest_date, MAX(transaction_date) as latest_date FROM transactions",
            cache_ttl=300
        )
        date_range = date_range_result.data[0]
        
        # Get latest balance
        latest_balance_result = db_manager.execute_query(
            "SELECT balance FROM transactions ORDER BY transaction_date DESC, transaction_id DESC LIMIT 1",
            cache_ttl=10
        )
        latest_balance = latest_balance_result.data[0]['balance'] if latest_balance_result.data else 0
        
        # Get totals
        totals_result = db_manager.execute_query(
            "SELECT SUM(debit_amount) as total_debits, SUM(credit_amount) as total_credits FROM transactions WHERE debit_amount IS NOT NULL OR credit_amount IS NOT NULL",
            cache_ttl=60
        )
        totals = totals_result.data[0]
        
        # Format amounts as INR currency
        def format_inr(amount):
            if amount is None:
                return "₹0.00"
            return f"₹{amount:,.2f}"
        
        # Prepare summary data
        summary = {
            "data": {
                "total_transactions": total_transactions,
                "date_range": {
                    "earliest": date_range['earliest_date'], 
                    "latest": date_range['latest_date']
                },
                "current_balance_inr": format_inr(latest_balance),
                "current_balance_raw": latest_balance,
                "total_debits_inr": format_inr(totals['total_debits']) if totals['total_debits'] else "₹0.00",
                "total_credits_inr": format_inr(totals['total_credits']) if totals['total_credits'] else "₹0.00",
                "total_debits_raw": totals['total_debits'] or 0,
                "total_credits_raw": totals['total_credits'] or 0,
                "performance": {
                    "queries_executed": 4,
                    "cache_hits": sum(1 for r in [total_result, date_range_result, latest_balance_result, totals_result] if r.cached),
                    "total_execution_time": sum(r.execution_time for r in [total_result, date_range_result, latest_balance_result, totals_result])
                }
            },
            "currency": "INR",
            "generated_at": datetime.now().isoformat(),
            "server": "direct-database-access"
        }
        
        # Return the result with proper CORS headers
        return JSONResponse(
            content=summary,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting financial summary: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get financial summary: {str(e)}"
        )

# WebSocket endpoint for real-time communication
@app.websocket("/ws/{session_id}")
async def websocket_session_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat interface with session ID."""
    logger.info(f"WebSocket connection attempt for session: {session_id}")
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
                            "session_id": session_id,
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
        logger.info(f"WebSocket client disconnected for session: {session_id}")
        
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
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
    # For development, you might want to disable reload to avoid file watching issues
    # Set reload=False for stable development, or reload=True with proper excludes
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["backend"],  # Only watch the backend directory
        reload_excludes=[
            "logs",
            "*.db",
            "*.db-*",  # SQLite journal/WAL files
            "*.log", 
            "*.sqlite",
            "*.sqlite3",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".git",
            "venv",
            "Bank-Statements",
            "tests",
            "frontend"
        ],
        log_level="info"
    )
