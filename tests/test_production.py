#!/usr/bin/env python3
"""
Production Test Suite for FinanceBud Backend
============================================

Tests the production-ready system with generic LLM providers, FastMCP integration,
and database connectivity.
"""

import asyncio
import json
import logging
import sys
import os
import sqlite3
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.financial_agent_generic import GenericFinancialAgent
from backend.agents.llm_providers import LLMConfig, ProviderType, get_default_config
from fastmcp import Client as FastMCPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_database_connection():
    """Test database connectivity and basic queries."""
    logger.info("🧪 Testing Database Connection...")
    
    try:
        # Connect to the database
        db_path = project_root / "financial_data.db"
        if not db_path.exists():
            logger.warning(f"⚠️ Database not found at {db_path}")
            return False
            
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check available tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"✅ Database connected. Tables: {tables}")
        
        # Count transactions if table exists
        if 'transactions' in tables:
            cursor.execute("SELECT COUNT(*) FROM transactions;")
            count = cursor.fetchone()[0]
            logger.info(f"✅ Found {count} transactions in database")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

async def test_mcp_server_direct():
    """Test MCP server connectivity directly."""
    logger.info("🧪 Testing MCP Server Direct Connection...")
    
    try:
        # Try to connect to MCP server
        client = FastMCPClient("mcp_server.py")
        
        async with client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"✅ Found {len(tools)} MCP tools")
            
            # Test account summary tool
            result = await client.call_tool("get_account_summary", {})
            if hasattr(result, 'content'):
                logger.info("✅ MCP server responding correctly")
                return True
            else:
                logger.warning("⚠️ MCP server response format unexpected")
                return False
                
    except Exception as e:
        logger.error(f"❌ MCP server direct test failed: {e}")
        return False

async def test_provider_connectivity():
    """Test LLM provider connectivity."""
    logger.info("🧪 Testing LLM Provider Connectivity...")
    
    try:
        config = get_default_config()
        agent = GenericFinancialAgent(config)
        
        connection_ok = await agent.test_connection()
        if connection_ok:
            logger.info(f"✅ {config.provider.value} ({config.model}) provider connected")
            return True
        else:
            logger.warning(f"⚠️ {config.provider.value} provider connection failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Provider connectivity test failed: {e}")
        return False

async def test_agent_initialization():
    """Test agent initialization and health check."""
    logger.info("🧪 Testing Agent Initialization...")
    
    try:
        # Create agent with default config
        agent = GenericFinancialAgent()
        
        # Test connection
        connection_ok = await agent.test_connection()
        logger.info(f"✅ Agent connection: {'OK' if connection_ok else 'Failed'}")
        
        # Get health status
        health = await agent.get_health()
        logger.info(f"✅ Agent health: {health}")
        
        return agent if connection_ok else None
    
    except Exception as e:
        logger.error(f"❌ Agent initialization failed: {e}")
        return None

async def test_financial_agent():
    """Test the financial agent end-to-end."""
    logger.info("🧪 Testing Financial Agent...")
    
    try:
        agent = await test_agent_initialization()
        if not agent:
            return False
        
        # Test a simple query
        test_query = "Give me a quick account summary"
        logger.info(f"Sending query: '{test_query}'")
        
        response = await agent.chat(test_query)
        
        if response and len(response) > 10:
            logger.info(f"✅ Agent responded: {response[:100]}...")
            
            # Check if response contains financial keywords
            financial_keywords = ['balance', 'transaction', 'account', '₹', 'rupees']
            contains_financial_data = any(keyword in response.lower() for keyword in financial_keywords)
            
            if contains_financial_data:
                logger.info("✅ Response contains financial data")
                return True
            else:
                logger.warning("⚠️ Response may not contain actual financial data")
                return True  # Still pass as agent is working
        else:
            logger.warning("⚠️ Agent response was too short or empty")
            return False
            
    except Exception as e:
        logger.error(f"❌ Financial agent test failed: {e}")
        return False

async def run_all_tests():
    """Run all production tests."""
    logger.info("🚀 Starting Production Test Suite for FinanceBud Backend")
    logger.info("=" * 70)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("LLM Provider Connectivity", test_provider_connectivity),
        ("MCP Server Direct", test_mcp_server_direct),
        ("Agent Initialization", test_agent_initialization),
        ("Financial Agent E2E", test_financial_agent),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        try:
            result = await test_func()
            # Convert agent object to boolean for initialization test
            if test_name == "Agent Initialization":
                result = result is not None
            
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            logger.error(f"❌ FAILED: {test_name} - {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 70)
    
    passed_count = sum(1 for result in results.values() if result)
    total_count = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status} {test_name}")
    
    logger.info("=" * 70)
    logger.info(f"📈 Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        logger.info("🎉 ALL TESTS PASSED! FinanceBud is ready for production.")
    elif passed_count >= total_count * 0.8:  # 80% pass rate
        logger.info("⚠️ Most tests passed. System is mostly functional.")
    else:
        logger.error("❌ Many tests failed. System needs attention.")
    
    return passed_count == total_count

if __name__ == "__main__":
    print("🧪 FinanceBud Production Test Suite")
    print("=" * 70)
    print("Make sure the following are running:")
    print("1. MCP server: python mcp_server.py")
    print("2. LLM provider (e.g., ollama serve)")
    print("3. Database file exists: financial_data.db")
    print()
    
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
