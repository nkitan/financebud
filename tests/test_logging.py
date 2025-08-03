#!/usr/bin/env python3
"""
Test script to verify the new logging configuration
"""

import asyncio
import sys
import os

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.logging_config import setup_logging, get_logger_with_context

async def test_logging():
    """Test the logging configuration."""
    print("🔍 Testing FinanceBud Logging Configuration")
    print("=" * 50)
    
    # Setup logging
    setup_logging(log_level="DEBUG", log_dir="logs")
    
    # Get a logger with context
    logger = get_logger_with_context(__name__)
    
    print("📝 Testing basic logging...")
    logger.info("Testing basic info message")
    logger.debug("Testing debug message")
    logger.warning("Testing warning message")
    
    print("🔧 Testing structured logging...")
    
    # Test tool call logging
    logger.log_tool_call(
        "get_account_balance",
        {"account_type": "checking"},
        result={"balance": 1234.56},
        execution_time=0.123,
        session_id="test-session-123"
    )
    
    # Test chat request logging
    logger.log_chat_request(
        [{"role": "user", "content": "What is my account balance?"}],
        session_id="test-session-123"
    )
    
    # Test chat response logging
    logger.log_chat_response(
        "Your current account balance is $1,234.56",
        execution_time=1.45,
        session_id="test-session-123",
        tools_used=["get_account_balance", "format_currency"]
    )
    
    # Test error logging
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        logger.error("Testing error logging with exception", exc_info=True)
    
    print("✅ Logging tests completed!")
    print("📁 Check the 'logs/financebud.log' file for structured JSON logs")
    print("📊 Console shows human-readable format")
    
    # Check if log file was created
    if os.path.exists("logs/financebud.log"):
        with open("logs/financebud.log", "r") as f:
            lines = f.readlines()
            print(f"📄 Log file contains {len(lines)} entries")
            if lines:
                print("📋 Sample log entry:")
                print(lines[-1][:200] + "..." if len(lines[-1]) > 200 else lines[-1])
    else:
        print("⚠️  Log file not found")

if __name__ == "__main__":
    asyncio.run(test_logging())
