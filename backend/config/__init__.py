"""
Configuration Management
========================

Configuration management for FinanceBud backend.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field

# Also export the new settings system
from .settings import get_settings, settings

class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    description: Optional[str] = None

class LLMConfig(BaseModel):
    """Configuration for LLM integration."""
    provider: str = "openai"  # "openai", "local", etc.
    base_url: Optional[str] = "http://localhost:1234/v1"  # For local LM Studio
    api_key: Optional[str] = "lm-studio"
    model: str = "local-model"
    temperature: float = 0.7
    max_tokens: Optional[int] = 2000
    timeout: int = 300

class ServerConfig(BaseModel):
    """Main server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    
class DatabaseConfig(BaseModel):
    """Database configuration."""
    path: str = "/home/notroot/Work/financebud/financial_data.db"

class Config(BaseModel):
    """Main application configuration."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    
    @classmethod
    def load_from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()
        
        # Server config
        config.server.host = os.getenv("SERVER_HOST", config.server.host)
        config.server.port = int(os.getenv("SERVER_PORT", config.server.port))
        config.server.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Database config
        config.database.path = os.getenv("DATABASE_PATH", config.database.path)
        
        # LLM config
        config.llm.provider = os.getenv("LLM_PROVIDER", config.llm.provider)
        config.llm.base_url = os.getenv("LLM_BASE_URL", config.llm.base_url)
        config.llm.api_key = os.getenv("LLM_API_KEY", config.llm.api_key)
        config.llm.model = os.getenv("LLM_MODEL", config.llm.model)
        
        # Default MCP servers
        config.mcp_servers = [
            MCPServerConfig(
                name="financial-data-inr",
                command="/home/notroot/Work/financebud/venv/bin/python",
                args=["/home/notroot/Work/financebud/mcp_server.py"],
                description="Financial database server for Indian bank statements"
            )
        ]
        
        return config

# Global config instance
config = Config.load_from_env()

__all__ = ['config', 'get_settings', 'settings', 'Config', 'MCPServerConfig', 'LLMConfig', 'ServerConfig', 'DatabaseConfig']
