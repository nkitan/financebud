#!/usr/bin/env python3
"""
Simple test to verify environment loading
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from backend.agents.llm_providers import get_default_config

# Load environment variables
load_dotenv()

print("🔍 Testing environment variable loading...")

# Check if .env file exists
env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
print(f"📁 .env file path: {env_file}")
print(f"📁 .env file exists: {os.path.exists(env_file)}")

# Check environment variables directly
print(f"🔑 LLM_PROVIDER: {os.getenv('LLM_PROVIDER', 'NOT_SET')}")
print(f"🔑 GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY', 'NOT_SET')[:20]}...") # Only show first 20 chars
print(f"🔑 GEMINI_BASE_URL: {os.getenv('GEMINI_BASE_URL', 'NOT_SET')}")
print(f"🔑 GEMINI_MODEL: {os.getenv('GEMINI_MODEL', 'NOT_SET')}")

# Test the default config
print("\n🔧 Testing get_default_config()...")
try:
    config = get_default_config()
    print(f"✅ Config loaded successfully:")
    print(f"   Provider: {config.provider}")
    print(f"   Base URL: {config.base_url}")
    print(f"   Model: {config.model}")
    print(f"   API Key: {config.api_key[:20]}..." if config.api_key else "   API Key: None")
    print(f"   Timeout: {config.timeout}")
    print(f"   Max Tokens: {config.max_tokens}")
    print(f"   Temperature: {config.temperature}")
    
    if config.api_key and config.api_key != "your_api_key_here" and config.api_key != "GENERIC_API_KEY":
        print("✅ API key loaded correctly from environment!")
    else:
        print("❌ API key not loaded correctly from environment")
        
except Exception as e:
    print(f"❌ Error loading config: {e}")
