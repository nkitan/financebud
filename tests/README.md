# FinanceBud Test Suite

This directory contains comprehensive tests for the FinanceBud financial analysis system.

## Test Files

### Core Tests

- **`test_production.py`** - Complete production test suite
  - Database connectivity
  - LLM provider connectivity  
  - MCP server integration
  - Agent initialization
  - End-to-end financial agent testing

- **`test_tool_support.py`** - Tests which Ollama models support tool calling
  - Tests multiple models for OpenAI-compatible tool calling
  - Identifies the best models for FinanceBud

- **`test_providers.py`** - Tests different LLM providers
  - Tests Ollama, OpenAI, OpenRouter, Gemini providers
  - Demonstrates provider switching
  - Health checks for each provider

- **`test_financial_agent.py`** - Focused financial agent testing
  - Tests the GenericFinancialAgent specifically
  - Verifies tool calling with financial queries
  - Simple agent functionality verification

## Running Tests

### Prerequisites

1. **Start required services:**
   ```bash
   # Terminal 1: MCP Server
   python mcp_server.py
   
   # Terminal 2: Ollama (if using Ollama)
   ollama serve
   ```

2. **Ensure database exists:**
   ```bash
   ls financial_data.db
   ```

3. **Configure your LLM provider in `.env`**

### Run All Tests

```bash
# Run the complete production test suite
python tests/test_production.py

# Test tool calling support for models
python tests/test_tool_support.py

# Test different LLM providers
python tests/test_providers.py

# Test just the financial agent
python tests/test_financial_agent.py
```

### Individual Test Commands

```bash
# Activate virtual environment first
source venv/bin/activate

# Run any test file
python tests/test_[name].py
```

## Test Results Interpretation

### `test_production.py`
- ✅ All pass = Production ready
- ⚠️  80%+ pass = Mostly functional  
- ❌ <80% pass = Needs attention

### `test_tool_support.py`
- Shows which models support OpenAI-compatible tool calling
- Recommends the best model for FinanceBud

### `test_providers.py`
- Tests connectivity to different LLM providers
- Useful for troubleshooting provider issues

## Common Issues

### Database Issues
- Ensure `financial_data.db` exists in project root
- Check file permissions

### MCP Server Issues  
- Make sure `python mcp_server.py` is running
- Check for port conflicts

### LLM Provider Issues
- **Ollama**: Ensure `ollama serve` is running and model is pulled
- **OpenAI**: Check API key and credits
- **Others**: Verify API keys and configurations

### Tool Calling Issues
- Use models that support tools (llama3.1, qwen2.5, etc.)
- Avoid models like Gemma that don't support tool calling

## Adding New Tests

When adding new tests:

1. Follow the naming convention: `test_[feature].py`
2. Include proper docstrings and error handling
3. Use the logging module for output
4. Test both success and failure cases
5. Update this README with the new test description

## Test Files

### Production Tests
- **`test_production.py`** - Complete production test suite for the FastMCP-based system
  - Tests database connectivity
  - Tests LM Studio integration  
  - Tests FastMCP tools
  - Tests financial analysis agent
  - Tests MCP server direct connection

### Tool Testing
- **`test_fastmcp_tools.py`** - Comprehensive testing of individual FastMCP tools
  - Account summary
  - Transaction search
  - Date range queries
  - Monthly summaries
  - Spending analysis
  - UPI transaction analysis

### Utilities
- **`list_mcp_tools.py`** - Utility to list all available MCP tools and their parameters

## Running Tests

### Run All Production Tests
```bash
cd /home/notroot/Work/financebud
venv/bin/python tests/test_production.py
```

### Test FastMCP Tools
```bash
cd /home/notroot/Work/financebud
venv/bin/python tests/test_fastmcp_tools.py
```

### List Available MCP Tools
```bash
cd /home/notroot/Work/financebud
venv/bin/python tests/list_mcp_tools.py
```

## Test Requirements

All tests require:
- Python virtual environment activated (`venv/`)
- SQLite database (`financial_data.db`) with transaction data
- MCP server (`mcp_server.py`) accessible
- LM Studio running on `localhost:1234` (optional for some tests)

## Test Coverage

The test suite covers:
- ✅ Database connectivity and data integrity
- ✅ FastMCP protocol integration
- ✅ Financial analysis tools
- ✅ LangGraph agent workflows
- ✅ LM Studio AI integration
- ✅ Session management
- ✅ Error handling and fallbacks

## Architecture

The current system uses:
- **FastMCP** for MCP protocol handling (replaces custom MCP client)
- **LangGraph** for agent orchestration
- **LM Studio** for local AI responses
- **SQLite** for financial data storage
- **LangChain Tools** for tool execution
