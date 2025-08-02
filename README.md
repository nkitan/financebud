# FinanceBud - AI Financial Analysis Platform

A sophisticated financial analysis platform that combines Fa```
financebud/
├── 🤖 Backend System
│   ├── backend/
│   │   ├── main.py                      # FastAPI server with WebSocket
│   │   ├── config.py                    # Configuration management
│   │   ├── agents/
│   │   │   ├── financial_agent_generic.py  # Multi-provider financial agent
│   │   │   ├── financial_agent.py          # Legacy LM Studio agent (backward compatibility)
│   │   │   └── llm_providers.py            # LLM provider implementations
│   │   └── mcp/
│   │       └── client.py                # MCP client manager
│   │
├── 🌐 Frontend
│   └── frontend/
│       └── index.html               # Responsive web interface
│   │
├── 💾 Data & MCP
│   ├── mcp_server.py                # FastMCP server (6 tools)
│   ├── financial_data.db            # SQLite database (5,657+ transactions)
│   ├── consolidate_statements.py    # Database management
│   └── Bank-Statements/             # Original Excel files (2023-2025)
│   │
├── 🧪 Testing
│   ├── tests/
│   │   ├── test_production.py       # Complete test suite
│   │   ├── test_providers.py        # LLM provider tests
│   │   ├── test_tool_support.py     # Tool functionality tests
│   │   └── list_mcp_tools.py        # Tool discovery utility
│   │
├── ⚙️ Configuration & Setup
│   ├── .env.example                 # Environment configuration template
│   ├── setup_multi_provider.sh      # Multi-provider setup script
│   ├── requirements.txt             # Python dependencies
│   └── package.json                 # Project metadata and scripts
│   │
└── 📖 Documentation
    ├── README.md                    # This file
    └── ARCHITECTURE.md             # Technical details
``` with multi-provider LLM support (Ollama, OpenAI, Gemini, OpenRouter). Built specifically for Indian banking with comprehensive INR support and UPI transaction analysis.

## 🚀 Quick Start

```bash
# 1. Run the setup script
./setup_multi_provider.sh

# 2. Configure your LLM provider in .env
cp .env.example .env
# Edit .env with your preferred provider settings

# 3. Start the MCP server (Terminal 1)
python mcp_server.py

# 4. Start the backend server (Terminal 2)
source .venv/bin/activate
python -m backend.main

# 5. Access the web interface
open http://localhost:8000

# 6. Test the system
python tests/test_production.py
```

## 🌟 Features

### AI-Powered Financial Analysis
- **Natural Language Queries**: Ask questions like "What's my account balance?" or "Show me UPI transactions this month"
- **Multi-Provider LLM Support**: Use Ollama (local), OpenAI, Google Gemini, or OpenRouter
- **FastMCP Integration**: Modern MCP protocol with automatic tool discovery
- **Real-time Chat**: WebSocket-based communication with session management

### Financial Tools
- **Account Summaries**: Current balances and transaction overviews in INR
- **Transaction Search**: Find specific payments, transfers, and UPI transactions  
- **Monthly Analysis**: Spending patterns and trends with INR formatting
- **UPI Analytics**: Digital payment insights specific to Indian banking
- **Recurring Payments**: Automatic detection of subscriptions and bills
- **Date Range Queries**: Analyze transactions within specific periods
- **Custom Analysis**: Flexible SQL queries with security limits

### Technical Features
- **Provider-Agnostic Design**: Easy switching between LLM providers via configuration
- **Production-Ready**: Comprehensive error handling, logging, and health monitoring
- **FastAPI Backend**: Async API with WebSocket support and auto-documentation
- **Responsive Frontend**: Modern web interface with real-time updates
- **Database**: SQLite with 5,657+ transactions from 2023-2025

## 📊 Database Overview

- **Total Transactions**: 5,657
- **Date Range**: January 2023 - August 2025  
- **Current Balance**: ₹40,650.11
- **UPI Transactions**: 5,576 (98.5%)
- **Currency**: All amounts in INR (₹) with proper formatting

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │  LLM Provider   │
│   (HTML/JS)     │◄──►│   (FastAPI)      │◄──►│  (Configurable) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        │
                       ┌──────────────────┐              │
                       │ Generic Financial│              │
                       │     Agent        │              │
                       │ (Multi-Provider) │              │
                       └──────────────────┘              │
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  FastMCP Tools   │    │ LLM Providers:  │
                       │  (6 Financial    │    │ • Ollama        │
                       │   Tools)         │    │ • OpenAI        │
                       └──────────────────┘    │ • Gemini        │
                                │              │ • OpenRouter    │
                                ▼              └─────────────────┘
                       ┌──────────────────┐
                       │  MCP Server      │
                       │  (financial_data.│
                       │   db)            │
                       └──────────────────┘
```

## 📁 Project Structure

```
financebud/
├── 🤖 Backend System
│   ├── backend/
│   │   ├── main.py                  # FastAPI server with WebSocket
│   │   ├── agents/
│   │   │   └── financial_agent.py   # LangGraph agent with FastMCP tools
│   │   ├── models/
│   │   │   └── schemas.py           # Pydantic data models
│   │   └── config.py                # Configuration management
│   │
├── 🌐 Frontend
│   └── frontend/
│       └── index.html               # Responsive web interface
│   │
├── 💾 Data & MCP
│   ├── mcp_server.py                # FastMCP server (10 tools)
│   ├── financial_data.db            # SQLite database
│   ├── consolidate_statements.py    # Database management
│   └── Bank-Statements/             # Original Excel files
│   │
├── 🧪 Testing
│   ├── tests/
│   │   ├── test_production.py       # Complete test suite
│   │   ├── test_fastmcp_tools.py    # Individual tool tests
│   │   └── list_mcp_tools.py        # Tool discovery utility
│   │
└── � Documentation
    ├── README.md                    # This file
    ├── SETUP_GUIDE.md              # Installation and setup
    └── ARCHITECTURE.md             # Technical details
```

## 🛠️ Dependencies

### Core Technologies
- **Python 3.8+** with asyncio support
- **FastMCP 2.11.0** - Modern MCP client/server implementation
- **FastAPI** - High-performance web framework
- **Multi-Provider LLM Support** - Ollama, OpenAI, Gemini, OpenRouter

### Key Python Packages
```
fastapi>=0.104.0
fastmcp>=2.11.0
aiohttp>=3.9.0
uvicorn>=0.24.0
websockets>=12.0
openai>=1.0.0  # For OpenAI-compatible API clients
sqlite3 (built-in)
```

### LLM Provider Requirements
- **Ollama**: Local installation (`curl -fsSL https://ollama.ai/install.sh | sh`)
- **OpenAI**: API key from OpenAI platform
- **Google Gemini**: API key from Google AI Studio
- **OpenRouter**: API key from OpenRouter platform

## 🔧 Available Tools

The system provides 6 FastMCP tools for financial analysis:

1. **`get_account_summary`** - Account overview with current balance
2. **`search_transactions`** - Find transactions by description pattern
3. **`get_transactions_by_date_range`** - Query by date range
4. **`get_monthly_summary`** - Monthly spending analysis
5. **`analyze_spending_trends`** - Multi-month trend analysis
6. **`get_upi_transaction_analysis`** - UPI-specific insights

## 🤖 LLM Provider Support

### 🦙 Ollama (Recommended for Privacy)
- **Best for**: Local execution, privacy, no API costs
- **Setup**: `curl -fsSL https://ollama.ai/install.sh | sh`
- **Start**: `ollama serve`
- **Model**: `ollama pull llama3.1` or `ollama pull gemma2`
- **Config**: Set `LLM_PROVIDER=ollama` in `.env`

### 🤖 OpenAI
- **Best for**: Maximum capability and speed
- **Setup**: Get API key from OpenAI
- **Config**: 
  ```env
  LLM_PROVIDER=openai
  OPENAI_API_KEY=your_key_here
  OPENAI_MODEL=gpt-4
  ```

### 🔀 OpenRouter
- **Best for**: Access to multiple models (Claude, GPT, Llama, etc.)
- **Setup**: Get API key from OpenRouter
- **Config**:
  ```env
  LLM_PROVIDER=openrouter
  OPENROUTER_API_KEY=your_key_here
  OPENROUTER_MODEL=anthropic/claude-3-sonnet
  ```

### 💎 Google Gemini
- **Best for**: Google's latest AI capabilities
- **Setup**: Get API key from Google AI Studio
- **Config**:
  ```env
  LLM_PROVIDER=gemini
  GEMINI_API_KEY=your_key_here
  GEMINI_MODEL=gemini-pro
  ```

## 🚀 API Endpoints

### Chat & Communication
- `POST /chat` - Send messages to the financial agent
- `GET /ws/{session_id}` - WebSocket for real-time chat
- `GET /sessions/{session_id}/history` - Get chat history

### Health & Monitoring  
- `GET /health` - System health check
- `GET /servers` - List available FastMCP tools
- `GET /metrics` - Detailed system metrics

### Documentation
- `GET /docs` - Interactive API documentation
- `GET /` - Web interface

## 📈 Example Queries

```python
# Natural language examples that work with the AI agent:
"What's my current account balance?"
"Show me all UPI transactions this month"
"How much did I spend in December 2024?"
"Find transactions containing 'swiggy'"
"Analyze my spending trends over the last 6 months"
"What are my recurring payments?"
```

## 🔍 Testing

```bash
# Run complete test suite with your configured provider
python tests/test_production.py

# Test all LLM providers
python tests/test_providers.py

# Test specific tools
python tests/test_tool_support.py

# List all available MCP tools
python list_mcp_tools.py
```

## 📖 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture and design decisions
- **[tests/README.md](tests/README.md)** - Testing documentation
- **[.env.example](.env.example)** - Environment configuration reference

## 🤝 Contributing

This project uses modern Python async patterns and follows FastMCP conventions. Key areas for contribution:

- Additional financial analysis tools
- Enhanced natural language processing  
- Integration with more banking formats
- Performance optimizations
- UI/UX improvements
- Support for additional LLM providers

## 📄 License

This project is designed for personal financial analysis and learning purposes.

## 🚀 Advanced Configuration

### Switching Between Providers

You can easily switch between LLM providers:

#### Method 1: Environment Variables
```bash
export LLM_PROVIDER=ollama  # or openai, gemini, openrouter
```

#### Method 2: Edit .env file
```env
LLM_PROVIDER=your_preferred_provider
```

#### Method 3: Programmatically
```python
from backend.agents.llm_providers import LLMConfig, ProviderType
from backend.agents.financial_agent_generic import GenericFinancialAgent

# Switch to Ollama
config = LLMConfig(
    provider=ProviderType.OLLAMA,
    base_url="http://localhost:11434",
    model="llama3.1"
)
agent = GenericFinancialAgent(config)

# Switch to OpenAI
config = LLMConfig(
    provider=ProviderType.OPENAI,
    base_url="https://api.openai.com",
    model="gpt-4",
    api_key="your_key"
)
agent.switch_provider(config)
```

### Benefits of Multi-Provider Architecture

✅ **Provider Independence**: No vendor lock-in  
✅ **Cost Flexibility**: Choose based on your budget  
✅ **Privacy Options**: Local execution with Ollama  
✅ **Performance Options**: Use fast cloud models when needed  
✅ **Easy Switching**: Change providers without code changes  
✅ **Standard API**: OpenAI-compatible interface everywhere  

## 🛠️ Troubleshooting

### Provider Connection Issues
```bash
# Test specific provider
python tests/test_providers.py

# Check system health
curl http://localhost:8000/health
```

### Ollama Issues
```bash
# Check if Ollama is running
ollama list

# Start Ollama
ollama serve

# Install model if missing
ollama pull llama3.1
```

### API Key Issues
- Make sure API keys are set in `.env`
- Check if keys have sufficient credits/quota
- Verify the key format is correct

---

*FinanceBud v2.0 - Now with multi-provider LLM support for ultimate flexibility! 🚀*
