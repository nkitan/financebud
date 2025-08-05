"""
Generic LLM Provider Interface for Financial Agent
=================================================

This module provides a unified interface for different LLM providers,
making it easy to switch between Ollama, OpenAI, Google Gemini, OpenRouter, etc.
"""

import os
import json
import logging
import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import centralized logging
from ..logging_config import get_logger_with_context

logger = get_logger_with_context(__name__)

class ProviderType(Enum):
    """Supported LLM provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: ProviderType
    base_url: str
    api_key: str = "GENERIC_API_KEY"
    model: str = "llama3.1"
    timeout: int = 300
    max_tokens: int = 1500
    temperature: float = 0.7

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Send a chat completion request."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if the provider is available."""
        pass
    
    def _error_response(self, error: str) -> Dict[str, Any]:
        """Create error response in OpenAI format."""
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"I'm experiencing connection issues: {error}"
                }
            }]
        }

class OllamaProvider(LLMProvider):
    """Ollama provider - OpenAI compatible."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.chat_url = f"{config.base_url}/v1/chat/completions"
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Send chat completion request to Ollama."""
        import time
        start_time = time.time()
        
        # Log the chat request
        logger.log_chat_request(messages, session_id=f"ollama_{self.config.model}")
        
        # Set longer timeouts for complex tool processing
        timeout = self.config.timeout
        if tools and len(tools) > 0:
            timeout = min(timeout, 300)  # Max 5 minutes when processing with tools
        else:
            timeout = min(timeout, 300)  # Max 5 minutes for regular chat
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.chat_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        execution_time = time.time() - start_time
                        
                        # Extract response content for logging
                        response_content = ""
                        if 'choices' in result and len(result['choices']) > 0:
                            choice = result['choices'][0]
                            if 'message' in choice and 'content' in choice['message']:
                                response_content = choice['message']['content']
                        
                        # Log the successful response
                        logger.log_chat_response(
                            response_content,
                            execution_time=execution_time,
                            session_id=f"ollama_{self.config.model}"
                        )
                        
                        logger.debug(f"Ollama API success: {result}")
                        return result
                    else:
                        execution_time = time.time() - start_time
                        error_text = await response.text()
                        error_msg = f"Ollama returned status {response.status}: {error_text}"
                        
                        # Log the error response
                        logger.log_chat_response(
                            f"ERROR: {error_msg}",
                            execution_time=execution_time,
                            session_id=f"ollama_{self.config.model}",
                            level=logging.ERROR
                        )
                        logger.error(error_msg)
                        raise Exception(error_msg)
        except asyncio.TimeoutError as e:
            execution_time = time.time() - start_time
            error_msg = f"Ollama API timeout after {self.config.timeout}s"
            logger.log_chat_response(
                f"ERROR: {error_msg}",
                execution_time=execution_time,
                session_id=f"ollama_{self.config.model}",
                level=logging.ERROR
            )
            logger.error(error_msg)
            return self._error_response(error_msg)
        except aiohttp.ClientError as e:
            execution_time = time.time() - start_time
            error_msg = f"Ollama API client error: {str(e)}"
            logger.log_chat_response(
                f"ERROR: {error_msg}",
                execution_time=execution_time,
                session_id=f"ollama_{self.config.model}",
                level=logging.ERROR
            )
            logger.error(error_msg)
            return self._error_response(error_msg)
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Ollama API unexpected error: {type(e).__name__}: {str(e)}"
            logger.log_chat_response(
                f"ERROR: {error_msg}",
                execution_time=execution_time,
                session_id=f"ollama_{self.config.model}",
                level=logging.ERROR
            )
            logger.error(error_msg)
            return self._error_response(error_msg)
    
    async def test_connection(self) -> bool:
        """Test Ollama connection."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.chat_url,
                    json={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5
                    },
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    is_ok = response.status == 200
                    if is_ok:
                        logger.debug("Ollama connection test successful")
                    else:
                        logger.warning(f"Ollama connection test failed with status {response.status}")
                    return is_ok
        except asyncio.TimeoutError:
            logger.warning("Ollama connection test timed out")
            return False
        except Exception as e:
            logger.warning(f"Ollama connection test failed: {type(e).__name__}: {str(e)}")
            return False

class OpenAIProvider(LLMProvider):
    """OpenAI provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.chat_url = f"{config.base_url}/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Send chat completion request to OpenAI."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.chat_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"OpenAI returned status {response.status}: {error_text}")
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return self._error_response(str(e))
    
    async def test_connection(self) -> bool:
        """Test OpenAI connection."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.chat_url,
                    json={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5
                    },
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"OpenAI connection test failed: {e}")
            return False

class GeminiProvider(LLMProvider):
    """Google Gemini provider (OpenAI-compatible endpoint)."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # Use the correct OpenAI-compatible endpoint for Gemini
        # The base URL should already include the path, so we just append the chat completions endpoint
        self.chat_url = f"{config.base_url}/chat/completions"
        print(self.chat_url, config.api_key)
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Send chat completion request to Gemini."""
        import time
        start_time = time.time()
        
        # Log the chat request
        logger.log_chat_request(messages, session_id=f"gemini_{self.config.model}")
        
        # Validate messages for Gemini compatibility - check for empty function names
        validated_messages = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "tool":
                # Ensure tool messages have a proper name field
                msg_copy = msg.copy()  # Don't modify original
                
                # Check and fix name field with more robust validation
                name_value = msg_copy.get("name")
                if not name_value or not isinstance(name_value, str) or str(name_value).strip() == "":
                    logger.warning(f"Tool message {i} missing or invalid name field. Original: {name_value}, Full msg: {msg}")
                    msg_copy["name"] = f"fallback_function_{i}"
                
                # Also ensure tool_call_id is present for tool messages
                tool_call_id = msg_copy.get("tool_call_id")
                if not tool_call_id or not isinstance(tool_call_id, str) or str(tool_call_id).strip() == "":
                    logger.warning(f"Tool message {i} missing or invalid tool_call_id. Original: {tool_call_id}")
                    msg_copy["tool_call_id"] = f"fallback_call_{i}"
                
                # Final validation to ensure both fields are non-empty strings
                final_name = str(msg_copy.get("name", "")).strip()
                final_tool_call_id = str(msg_copy.get("tool_call_id", "")).strip()
                
                if not final_name:
                    msg_copy["name"] = f"emergency_function_{i}"
                    logger.error(f"Tool message {i} name still empty after validation, using emergency fallback")
                
                if not final_tool_call_id:
                    msg_copy["tool_call_id"] = f"emergency_call_{i}"
                    logger.error(f"Tool message {i} tool_call_id still empty after validation, using emergency fallback")
                
                validated_messages.append(msg_copy)
                logger.info(f"Final validated tool message {i}: name='{msg_copy.get('name')}', tool_call_id='{msg_copy.get('tool_call_id')}'")
            else:
                validated_messages.append(msg)
        
        # Set longer timeouts for complex tool processing
        timeout = self.config.timeout
        if tools and len(tools) > 0:
            timeout = min(timeout, 300)  # Max 5 minutes when processing with tools
        else:
            timeout = min(timeout, 300)  # Max 5 minutes for regular chat
        
        payload = {
            "model": self.config.model,
            "messages": validated_messages,  # Use validated messages instead of original
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        # Add tools if provided (OpenAI format works directly)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        # Debug logging for Gemini requests with tool messages
        tool_messages = [msg for msg in validated_messages if msg.get("role") == "tool"]
        if tool_messages:
            logger.info(f"Sending {len(tool_messages)} tool messages to Gemini:")
            for i, tool_msg in enumerate(tool_messages):
                logger.info(f"  Tool message {i+1}: name='{tool_msg.get('name')}', tool_call_id='{tool_msg.get('tool_call_id')}', content_length={len(tool_msg.get('content', ''))}")
            
            # Also log the full payload structure for debugging (but truncate content)
            debug_payload = payload.copy()
            debug_payload["messages"] = []
            for msg in validated_messages:
                debug_msg = msg.copy()
                if len(debug_msg.get("content", "")) > 100:
                    debug_msg["content"] = debug_msg["content"][:100] + "... [truncated]"
                debug_payload["messages"].append(debug_msg)
            logger.debug(f"Full Gemini payload structure: {json.dumps(debug_payload, indent=2)}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.chat_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        execution_time = time.time() - start_time
                        
                        # Fix empty tool call IDs that Gemini returns
                        if 'choices' in result:
                            for choice in result['choices']:
                                if 'message' in choice and 'tool_calls' in choice['message']:
                                    tool_calls = choice['message']['tool_calls']
                                    for i, tool_call in enumerate(tool_calls):
                                        if not tool_call.get('id') or tool_call.get('id') == '':
                                            # Generate a unique ID for this tool call
                                            import uuid
                                            tool_call['id'] = f"call_{uuid.uuid4().hex[:8]}"
                                            logger.debug(f"Generated tool call ID: {tool_call['id']}")
                        
                        # Extract response content for logging
                        response_content = ""
                        if 'choices' in result and len(result['choices']) > 0:
                            choice = result['choices'][0]
                            if 'message' in choice and 'content' in choice['message']:
                                response_content = choice['message']['content']
                        
                        # Log the successful response
                        logger.log_chat_response(
                            response_content,
                            execution_time=execution_time,
                            session_id=f"gemini_{self.config.model}"
                        )
                        
                        logger.debug(f"Gemini API success: {result}")
                        return result
                    else:
                        execution_time = time.time() - start_time
                        error_text = await response.text()
                        error_msg = f"Gemini returned status {response.status}: {error_text}"
                        
                        # Log the request payload that caused the error for debugging
                        logger.debug(f"Gemini request payload that caused error: {json.dumps(payload, indent=2)}")
                        
                        # Log the error response
                        logger.log_chat_response(
                            f"ERROR: {error_msg}",
                            execution_time=execution_time,
                            session_id=f"gemini_{self.config.model}",
                            level=logging.ERROR
                        )
                        logger.error(error_msg)
                        return self._error_response(error_msg)
        except asyncio.TimeoutError as e:
            execution_time = time.time() - start_time
            error_msg = f"Gemini API timeout after {timeout}s"
            logger.log_chat_response(
                f"ERROR: {error_msg}",
                execution_time=execution_time,
                session_id=f"gemini_{self.config.model}",
                level=logging.ERROR
            )
            logger.error(error_msg)
            return self._error_response(error_msg)
        except aiohttp.ClientError as e:
            execution_time = time.time() - start_time
            error_msg = f"Gemini API client error: {str(e)}"
            logger.log_chat_response(
                f"ERROR: {error_msg}",
                execution_time=execution_time,
                session_id=f"gemini_{self.config.model}",
                level=logging.ERROR
            )
            logger.error(error_msg)
            return self._error_response(error_msg)
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Gemini API unexpected error: {type(e).__name__}: {str(e)}"
            logger.log_chat_response(
                f"ERROR: {error_msg}",
                execution_time=execution_time,
                session_id=f"gemini_{self.config.model}",
                level=logging.ERROR
            )
            logger.error(error_msg)
            return self._error_response(error_msg)
    
    async def test_connection(self) -> bool:
        """Test Gemini connection."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.chat_url,
                    json={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5
                    },
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    is_ok = response.status == 200
                    if is_ok:
                        logger.debug("Gemini connection test successful")
                    else:
                        logger.warning(f"Gemini connection test failed with status {response.status}")
                    return is_ok
        except Exception as e:
            logger.warning(f"Gemini connection test failed: {e}")
            return False

class OpenRouterProvider(LLMProvider):
    """OpenRouter provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.chat_url = f"{config.base_url}/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/financebud",  # Required by OpenRouter
            "X-Title": "FinanceBud"  # Optional but recommended
        }
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Send chat completion request to OpenRouter."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.chat_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"OpenRouter returned status {response.status}: {error_text}")
        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            return self._error_response(str(e))
    
    async def test_connection(self) -> bool:
        """Test OpenRouter connection."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.chat_url,
                    json={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5
                    },
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"OpenRouter connection test failed: {e}")
            return False

def create_provider(config: LLMConfig) -> LLMProvider:
    """Factory function to create appropriate provider."""
    provider_map = {
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.GEMINI: GeminiProvider,
        ProviderType.OPENROUTER: OpenRouterProvider,
    }
    
    if config.provider not in provider_map:
        raise ValueError(f"Unsupported provider: {config.provider}")
    
    return provider_map[config.provider](config)

def get_default_config() -> LLMConfig:
    """Get default configuration from environment variables."""
    provider_name = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    try:
        provider = ProviderType(provider_name)
    except ValueError:
        logger.warning(f"Unknown provider '{provider_name}', defaulting to Ollama")
        provider = ProviderType.OLLAMA
    
    # Default configurations for different providers
    defaults = {
        ProviderType.OLLAMA: {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "model": os.getenv("OLLAMA_MODEL", "llama3.1"),
            "api_key": None
        },
        ProviderType.OPENAI: {
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
            "model": os.getenv("OPENAI_MODEL", "gpt-4"),
            "api_key": os.getenv("OPENAI_API_KEY", "your_api_key_here")
        },
        ProviderType.GEMINI: {
            "base_url": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "api_key": os.getenv("GEMINI_API_KEY", "your_api_key_here")
        },
        ProviderType.OPENROUTER: {
            "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api"),
            "model": os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-sonnet"),
            "api_key": os.getenv("OPENROUTER_API_KEY", "your_api_key_here")
        }
    }
    
    config_data = defaults[provider]
    
    # Set provider-specific timeout defaults - increased for complex queries
    default_timeout = "300" if provider == ProviderType.OLLAMA else "300"
    
    return LLMConfig(
        provider=provider,
        base_url=config_data["base_url"],
        api_key=config_data["api_key"],
        model=config_data["model"],
        timeout=int(os.getenv("LLM_TIMEOUT", default_timeout)),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),  # Increased for complete responses
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
