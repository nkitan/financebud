#!/bin/bash

# FinanceBud Multi-Provider Setup Script
# =====================================

echo "🚀 Setting up FinanceBud with multi-provider LLM support"
echo "========================================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created. Please edit it to configure your preferred LLM provider."
else
    echo "⚠️  .env file already exists. Make sure it's configured properly."
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo ""
echo "🔍 Checking system requirements..."

# Check Python
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✅ Python 3 found: $PYTHON_VERSION"
else
    echo "❌ Python 3 not found. Please install Python 3.8 or later."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created."
else
    echo "✅ Virtual environment already exists."
fi

# Activate virtual environment and install requirements
echo "📦 Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "🔧 LLM Provider Options:"
echo "========================"
echo ""
echo "1. 🦙 Ollama (Recommended for local/private use)"
echo "   - Free and runs locally"
echo "   - No API key required"
echo "   - Good privacy"
echo "   - Install: curl -fsSL https://ollama.ai/install.sh | sh"
echo "   - Start: ollama serve"
echo "   - Download model: ollama pull llama3.1:latest"
echo ""
echo "2. 🤖 OpenAI GPT"
echo "   - Powerful and fast"
echo "   - Requires API key and credits"
echo "   - Set OPENAI_API_KEY in .env"
echo ""
echo "3. 🔀 OpenRouter"
echo "   - Access to multiple models (Claude, GPT, etc.)"
echo "   - Requires API key and credits"
echo "   - Set OPENROUTER_API_KEY in .env"
echo ""
echo "4. 💎 Google Gemini"
echo "   - Google's powerful model"
echo "   - Requires API key"
echo "   - Set GEMINI_API_KEY in .env"

echo ""
echo "📋 Quick Start Guide:"
echo "===================="
echo ""
echo "1. Configure your preferred provider in .env:"
echo "   - For Ollama: LLM_PROVIDER=ollama"
echo "   - For OpenAI: LLM_PROVIDER=openai (+ set OPENAI_API_KEY)"
echo "   - For OpenRouter: LLM_PROVIDER=openrouter (+ set OPENROUTER_API_KEY)"
echo "   - For Gemini: LLM_PROVIDER=gemini (+ set GEMINI_API_KEY)"
echo ""
echo "2. Start the MCP server:"
echo "   python mcp_server.py"
echo ""
echo "3. In another terminal, start the backend:"
echo "   source venv/bin/activate"
echo "   python -m backend.main"
echo ""
echo "4. Test the setup:"
echo "   python tests/test_providers.py"
echo "   python tests/test_production.py"

# Check if Ollama is available
echo ""
if command_exists ollama; then
    echo "🦙 Ollama detected! Testing availability..."
    if ollama list >/dev/null 2>&1; then
        echo "✅ Ollama is running and ready to use."
        echo "📥 Available models:"
        ollama list
        
        # Check if llama3.1:latest is available
        if ollama list | grep -q "llama3.1:latest"; then
            echo "✅ llama3.1:latest model is ready!"
        else
            echo "📥 Downloading llama3.1:latest model (this may take a while)..."
            ollama pull llama3.1:latest
        fi
    else
        echo "⚠️  Ollama is installed but not running. Start it with: ollama serve"
    fi
else
    echo "⚠️  Ollama not found. Install it from: https://ollama.ai"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env to configure your LLM provider"
echo "2. Start the services as shown above"
echo "3. Open http://localhost:8000 in your browser"
echo ""
echo "For help switching providers dynamically, check out test_providers.py"
echo ""
echo "Happy financial analysis! 💰📊"
