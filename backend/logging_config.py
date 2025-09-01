"""
Centralized Logging Configuration for FinanceBud
===============================================

This module provides a centralized logging configuration with:
- Rolling log files (5 files, 10MB each)
- Debug level logging for tool calls, chat requests, and responses
- Structured logging format
- File and console output
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


# Define ContextLogger class globally for proper typing
class ContextLogger(logging.Logger):
    """Custom logger with structured logging methods."""
    
    def __init__(self, name):
        super().__init__(name)
        
    def log_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: Any = None, 
                     execution_time: Optional[float] = None, session_id: Optional[str] = None, 
                     level: int = logging.DEBUG):
        """Log a tool call with structured data."""
        tool_call_data = {
            'name': tool_name,
            'arguments': arguments
        }
        if result is not None:
            tool_call_data['result'] = str(result)[:1000]  # Truncate long results
        
        extra: Dict[str, Any] = {'tool_call': tool_call_data}
        if execution_time is not None:
            extra['execution_time'] = execution_time
        if session_id is not None:
            extra['session_id'] = session_id
        
        self.log(level, f"Tool call: {tool_name}", extra=extra)
    
    def log_chat_request(self, messages: List[Dict[str, Any]], session_id: Optional[str] = None,
                        level: int = logging.DEBUG):
        """Log a chat request with structured data."""
        # Truncate messages for logging
        truncated_messages = []
        for msg in messages[-3:]:  # Only log last 3 messages
            truncated_msg = {**msg}
            if 'content' in truncated_msg and len(str(truncated_msg['content'])) > 500:
                truncated_msg['content'] = str(truncated_msg['content'])[:500] + "..."
            truncated_messages.append(truncated_msg)
        
        extra: Dict[str, Any] = {'chat_request': {'messages': truncated_messages}}
        if session_id is not None:
            extra['session_id'] = session_id
        
        self.log(level, f"Chat request with {len(messages)} messages", extra=extra)
    
    def log_chat_response(self, response: str, execution_time: Optional[float] = None, 
                         session_id: Optional[str] = None, tools_used: Optional[List[str]] = None,
                         level: int = logging.DEBUG):
        """Log a chat response with structured data."""
        # Truncate response for logging
        truncated_response = response[:1000] + "..." if len(response) > 1000 else response
        
        response_data: Dict[str, Any] = {'content': truncated_response}
        if tools_used:
            response_data['tools_used'] = tools_used
        
        extra: Dict[str, Any] = {'chat_response': response_data}
        if execution_time is not None:
            extra['execution_time'] = execution_time
        if session_id is not None:
            extra['session_id'] = session_id
        
        self.log(level, f"Chat response generated", extra=extra)


class StructuredFormatter(logging.Formatter):
    """Custom formatter that provides human-readable structured logging."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record):
        # Start with base message
        base_message = super().format(record)
        
        # Add context information on same line
        context_parts = []
        
        # Add session info
        if hasattr(record, 'session_id'):
            context_parts.append(f"session={getattr(record, 'session_id')}")
        
        # Add execution time
        if hasattr(record, 'execution_time'):
            exec_time = getattr(record, 'execution_time')
            context_parts.append(f"time={exec_time:.3f}s")
        
        # Add tool call info
        if hasattr(record, 'tool_call'):
            tool_call = getattr(record, 'tool_call')
            if isinstance(tool_call, dict):
                tool_name = tool_call.get('name', 'unknown')
                context_parts.append(f"tool={tool_name}")
                # Add result preview if available
                if 'result' in tool_call:
                    result_preview = str(tool_call['result'])[:50]
                    if len(str(tool_call['result'])) > 50:
                        result_preview += "..."
                    context_parts.append(f"result='{result_preview}'")
        
        # Add chat request info
        if hasattr(record, 'chat_request'):
            chat_req = getattr(record, 'chat_request')
            if isinstance(chat_req, dict) and 'messages' in chat_req:
                msg_count = len(chat_req['messages'])
                context_parts.append(f"messages={msg_count}")
        
        # Add chat response info
        if hasattr(record, 'chat_response'):
            chat_resp = getattr(record, 'chat_response')
            if isinstance(chat_resp, dict):
                if 'tools_used' in chat_resp:
                    tools = chat_resp['tools_used']
                    if tools:
                        context_parts.append(f"tools_used={','.join(tools)}")
                if 'content' in chat_resp:
                    content_preview = str(chat_resp['content'])[:30]
                    if len(str(chat_resp['content'])) > 30:
                        content_preview += "..."
                    context_parts.append(f"response='{content_preview}'")
        
        # Add function call info
        if hasattr(record, 'function_call'):
            func_call = getattr(record, 'function_call')
            if isinstance(func_call, dict) and 'function' in func_call:
                context_parts.append(f"function={func_call['function']}")
        
        # Combine base message with context
        if context_parts:
            base_message += f" [{' | '.join(context_parts)}]"
        
        # Add exception info on new lines if present
        if record.exc_info:
            base_message += "\n" + self.formatException(record.exc_info)
        
        return base_message


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for console output."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record):
        # Add context information to console messages
        base_message = super().format(record)
        
        # Add extra context for important operations
        if hasattr(record, 'tool_call'):
            tool_call = getattr(record, 'tool_call')
            if isinstance(tool_call, dict) and 'name' in tool_call:
                base_message += f" [TOOL: {tool_call.get('name', 'unknown')}]"
        if hasattr(record, 'session_id'):
            base_message += f" [SESSION: {getattr(record, 'session_id')}]"
        if hasattr(record, 'execution_time'):
            base_message += f" [TIME: {getattr(record, 'execution_time'):.3f}s]"
        
        return base_message


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Set up centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create file handler with rotation (5 files, 10MB each)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path / "financebud.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG and above to files
    file_handler.setFormatter(StructuredFormatter())
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(ConsoleFormatter())
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set specific loggers to appropriate levels
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, log_dir={log_dir}")


def get_logger_with_context(name: str) -> ContextLogger:
    """
    Get a logger with context capabilities.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        ContextLogger with structured logging methods
    """
    # Create and configure the logger
    context_logger = ContextLogger(name)
    base_logger = logging.getLogger(name)
    
    # Set the same level as the base logger
    context_logger.setLevel(base_logger.level)
    
    # Copy handlers from the root logger if the base logger doesn't have any
    if not base_logger.handlers:
        root_logger = logging.getLogger()
        context_logger.handlers = root_logger.handlers[:]
    else:
        context_logger.handlers = base_logger.handlers[:]
    
    # Enable propagation so messages go to parent loggers
    context_logger.propagate = True
    
    return context_logger
