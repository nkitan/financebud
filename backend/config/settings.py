"""
Production-Ready Configuration Management
========================================

Comprehensive settings management using Pydantic Settings with environment variable
support, validation, and production-ready defaults.
"""

import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings
from enum import Enum


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # SQLite Database Configuration
    database_path: Path = Field(
        default=Path("financial_data.db"),
        description="Path to SQLite database file"
    )
    database_url: str = Field(
        default="sqlite:///financial_data.db",
        description="Database connection URL"
    )
    
    # Connection Pool Settings
    pool_size: int = Field(default=20, ge=5, le=100, description="Database connection pool size")
    max_overflow: int = Field(default=30, ge=0, le=50, description="Maximum connection overflow")
    pool_timeout: float = Field(default=30.0, ge=1.0, le=300.0, description="Pool connection timeout")
    pool_recycle: int = Field(default=3600, ge=300, le=7200, description="Pool connection recycle time")
    
    # Query Cache Settings
    cache_size: int = Field(default=2000, ge=100, le=10000, description="Query cache size")
    cache_ttl: int = Field(default=300, ge=60, le=3600, description="Default cache TTL in seconds")
    
    # Performance Settings
    pragma_journal_mode: str = Field(default="WAL", description="SQLite journal mode")
    pragma_synchronous: str = Field(default="NORMAL", description="SQLite synchronous mode")
    pragma_cache_size: int = Field(default=10000, description="SQLite cache size")
    pragma_temp_store: str = Field(default="MEMORY", description="SQLite temp store")
    pragma_mmap_size: int = Field(default=268435456, description="SQLite mmap size (256MB)")
    
    class Config:
        env_prefix = "DB_"


class LLMSettings(BaseSettings):
    """LLM provider configuration settings."""
    
    # Provider Selection
    provider: LLMProvider = Field(default=LLMProvider.OLLAMA, description="LLM provider to use")
    
    # Common Settings
    timeout: float = Field(default=300.0, ge=10.0, le=600.0, description="Request timeout in seconds")
    max_tokens: int = Field(default=2000, ge=100, le=8192, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    
    # Ollama Settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_model: str = Field(default="llama3.1", description="Ollama model name")
    
    # OpenAI Settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_base_url: str = Field(default="https://api.openai.com", description="OpenAI base URL")
    openai_model: str = Field(default="gpt-4-turbo", description="OpenAI model name")
    openai_organization: Optional[str] = Field(default=None, description="OpenAI organization ID")
    
    # Google Gemini Settings
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai",
        description="Gemini base URL"
    )
    gemini_model: str = Field(default="gemini-2.0-flash-exp", description="Gemini model name")
    
    # OpenRouter Settings
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api", description="OpenRouter base URL")
    openrouter_model: str = Field(default="anthropic/claude-3-sonnet", description="OpenRouter model name")
    
    # Anthropic Settings
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    anthropic_base_url: str = Field(default="https://api.anthropic.com", description="Anthropic base URL")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", description="Anthropic model name")
    
    # Advanced Settings
    retry_attempts: int = Field(default=3, ge=1, le=10, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Delay between retries")
    enable_streaming: bool = Field(default=False, description="Enable streaming responses")
    enable_function_calling: bool = Field(default=True, description="Enable function calling")
    
    # Performance Optimizations
    skip_final_synthesis: bool = Field(default=True, description="Skip final synthesis for faster responses")
    tool_call_timeout: float = Field(default=30.0, ge=5.0, le=120.0, description="Tool call timeout")
    max_tool_calls_per_request: int = Field(default=5, ge=1, le=20, description="Max tool calls per request")
    
    class Config:
        env_prefix = "LLM_"
    
    @model_validator(mode='after')
    def validate_api_keys(self):
        """Validate API keys are present for cloud providers."""
        provider = self.provider
        
        if provider == LLMProvider.OPENAI and not self.openai_api_key:
            raise ValueError(f"openai_api_key is required for {provider} provider")
        elif provider == LLMProvider.GEMINI and not self.gemini_api_key:
            raise ValueError(f"gemini_api_key is required for {provider} provider")
        elif provider == LLMProvider.OPENROUTER and not self.openrouter_api_key:
            raise ValueError(f"openrouter_api_key is required for {provider} provider")
        elif provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            raise ValueError(f"anthropic_api_key is required for {provider} provider")
        
        return self


class MCPSettings(BaseSettings):
    """MCP server configuration settings."""
    
    # Server Configuration
    server_timeout: float = Field(default=30.0, ge=5.0, le=300.0, description="MCP server timeout")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(default=2.0, ge=0.5, le=10.0, description="Retry delay in seconds")
    health_check_interval: float = Field(default=60.0, ge=10.0, le=300.0, description="Health check interval")
    auto_reconnect: bool = Field(default=True, description="Enable auto-reconnection")
    
    # Connection Pool Settings
    max_connections: int = Field(default=10, ge=1, le=50, description="Maximum concurrent connections")
    connection_timeout: float = Field(default=10.0, ge=1.0, le=60.0, description="Connection timeout")
    
    # Performance Settings
    request_queue_size: int = Field(default=1000, ge=10, le=10000, description="Request queue size")
    enable_request_batching: bool = Field(default=True, description="Enable request batching")
    batch_size: int = Field(default=10, ge=1, le=100, description="Request batch size")
    batch_timeout: float = Field(default=1.0, ge=0.1, le=10.0, description="Batch timeout")
    
    # Tool Cache Settings
    enable_tool_cache: bool = Field(default=True, description="Enable tool result caching")
    tool_cache_size: int = Field(default=1000, ge=10, le=10000, description="Tool cache size")
    tool_cache_ttl: int = Field(default=300, ge=10, le=3600, description="Tool cache TTL")
    
    class Config:
        env_prefix = "MCP_"


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    # API Security
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    secret_key: str = Field(default="your-secret-key-change-this", description="Secret key for JWT")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry")
    
    # CORS Settings
    allowed_origins: List[str] = Field(
        default=["http://nginx:8001", "http://localhost:8001", "*"],
        description="Allowed CORS origins"
    )
    allowed_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"],
        description="Allowed HTTP methods"
    )
    allowed_headers: List[str] = Field(
        default=["*"],
        description="Allowed headers"
    )
    
    # Rate Limiting
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Security Headers
    enable_security_headers: bool = Field(default=True, description="Enable security headers")
    
    class Config:
        env_prefix = "SECURITY_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    enable_structured_logging: bool = Field(default=True, description="Enable structured logging")
    
    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, ge=1000, le=65535, description="Metrics server port")
    
    # Health Checks
    enable_health_checks: bool = Field(default=True, description="Enable health check endpoints")
    health_check_timeout: float = Field(default=5.0, ge=1.0, le=30.0, description="Health check timeout")
    
    # Performance Monitoring
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    slow_query_threshold: float = Field(default=1.0, ge=0.1, le=10.0, description="Slow query threshold")
    
    # Error Tracking
    enable_error_tracking: bool = Field(default=True, description="Enable error tracking")
    error_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Error sampling rate")
    
    class Config:
        env_prefix = "MONITORING_"


class ServerSettings(BaseSettings):
    """Server configuration settings."""
    
    # Basic Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1000, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, le=32, description="Number of worker processes")
    
    # Performance Settings
    keepalive: int = Field(default=2, ge=1, le=300, description="Keep alive timeout")
    max_requests: int = Field(default=1000, ge=1, le=100000, description="Max requests per worker")
    max_requests_jitter: int = Field(default=100, ge=0, le=1000, description="Max requests jitter")
    timeout: int = Field(default=30, ge=1, le=300, description="Worker timeout")
    
    # SSL Settings
    ssl_enabled: bool = Field(default=False, description="Enable SSL")
    ssl_certfile: Optional[str] = Field(default=None, description="SSL certificate file")
    ssl_keyfile: Optional[str] = Field(default=None, description="SSL key file")
    
    # Static Files
    static_files_enabled: bool = Field(default=True, description="Serve static files")
    static_files_directory: str = Field(default="static", description="Static files directory")
    
    class Config:
        env_prefix = "SERVER_"


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    debug: bool = Field(default=False, description="Debug mode")
    testing: bool = Field(default=False, description="Testing mode")
    
    # Application Info
    app_name: str = Field(default="FinanceBud", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    app_description: str = Field(
        default="Production-ready financial analysis agent with real-time data processing",
        description="Application description"
    )
    
    # Configuration Sections
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    
    # Feature Flags
    enable_websockets: bool = Field(default=True, description="Enable WebSocket support")
    enable_api_docs: bool = Field(default=True, description="Enable API documentation")
    enable_admin_panel: bool = Field(default=False, description="Enable admin panel")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
    # Note: In Pydantic v2, cross-field validation moved to model_validator
    # This field validation is kept simple
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING or self.testing


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment (useful for testing)."""
    global _settings
    _settings = Settings()
    return _settings


# Export commonly used settings
settings = get_settings()
