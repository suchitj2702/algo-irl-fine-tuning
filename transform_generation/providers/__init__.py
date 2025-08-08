"""
LLM Provider Modules for Algo-IRL Fine-Tuning

This package contains modular implementations for different LLM providers:
- Claude (Anthropic)
- GPT (OpenAI) 
- Gemini (Google)

Each provider handles its own specific API requirements and parameter mappings.
"""

from .base_provider import BaseProvider, ProviderConfig, TaskType
from .provider_factory import get_provider, list_available_providers, get_provider_info, get_all_providers_info
from .claude_provider import ClaudeProvider
from .gpt_provider import GPTProvider
from .gemini_provider import GeminiProvider

__all__ = [
    'BaseProvider',
    'ProviderConfig', 
    'TaskType',
    'get_provider',
    'list_available_providers',
    'get_provider_info',
    'get_all_providers_info',
    'ClaudeProvider',
    'GPTProvider',
    'GeminiProvider'
]