"""
Provider Factory

Creates and manages LLM provider instances dynamically based on configuration.
This is the main entry point for getting provider instances.
"""

from typing import Dict, Type, List, Optional
from .base_provider import BaseProvider, ProviderConfig, TaskType
from .claude_provider import ClaudeProvider
from .gpt_provider import GPTProvider
from .gemini_provider import GeminiProvider
import logging

logger = logging.getLogger(__name__)


# Registry of available providers
PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {
    "claude": ClaudeProvider,
    "gpt": GPTProvider,
    "gemini": GeminiProvider
}


def get_provider(provider_name: str, task_type: TaskType, 
                 custom_config: Optional[Dict[str, any]] = None) -> BaseProvider:
    """
    Get a provider instance for the specified provider and task type
    
    Args:
        provider_name: Name of the provider ("claude", "gpt", "gemini")
        task_type: Type of task (TaskType enum)
        custom_config: Optional custom configuration to override defaults
        
    Returns:
        Configured provider instance
        
    Raises:
        ValueError: If provider_name is not supported
        Exception: If provider initialization fails
    """
    if provider_name not in PROVIDER_REGISTRY:
        available = list(PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown provider '{provider_name}'. Available providers: {available}")
    
    provider_class = PROVIDER_REGISTRY[provider_name]
    
    # Get default configuration for this provider and task type
    default_configs = provider_class.get_default_configs()
    if task_type not in default_configs:
        raise ValueError(f"Provider '{provider_name}' does not support task type '{task_type.value}'")
    
    default_config = default_configs[task_type]
    
    # Merge with custom configuration if provided
    if custom_config:
        # Create a copy to avoid modifying the default
        merged_config = default_config.copy()
        merged_config.update(custom_config)
        config_dict = merged_config
    else:
        config_dict = default_config
    
    # Create ProviderConfig instance
    provider_config = ProviderConfig(
        task_type=task_type,
        model=config_dict.get("model"),
        api_key=config_dict.get("api_key"),
        thinking_enabled=config_dict.get("thinking_enabled", False),
        thinking_budget=config_dict.get("thinking_budget", 0),
        max_output_tokens=config_dict.get("max_output_tokens", 15000),
        custom_params=config_dict.get("custom_params", {})
    )
    
    # Initialize and return the provider
    try:
        provider_instance = provider_class(provider_config)
        logger.info(f"Successfully initialized {provider_name} provider for {task_type.value}")
        return provider_instance
    except Exception as e:
        logger.error(f"Failed to initialize {provider_name} provider: {e}")
        raise


def list_available_providers() -> List[str]:
    """
    Get list of all available provider names
    
    Returns:
        List of provider names
    """
    return list(PROVIDER_REGISTRY.keys())


def get_provider_info(provider_name: str) -> Dict[str, any]:
    """
    Get information about a specific provider
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        Dictionary with provider information
        
    Raises:
        ValueError: If provider_name is not supported
    """
    if provider_name not in PROVIDER_REGISTRY:
        available = list(PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown provider '{provider_name}'. Available providers: {available}")
    
    provider_class = PROVIDER_REGISTRY[provider_name]
    
    return {
        "name": provider_class.get_provider_name(),
        "class": provider_class.__name__,
        "default_configs": {
            task_type.value: config 
            for task_type, config in provider_class.get_default_configs().items()
        }
    }


def get_all_providers_info() -> Dict[str, Dict[str, any]]:
    """
    Get information about all available providers
    
    Returns:
        Dictionary mapping provider names to their information
    """
    return {
        provider_name: get_provider_info(provider_name)
        for provider_name in list_available_providers()
    }


def register_provider(name: str, provider_class: Type[BaseProvider]):
    """
    Register a new provider class
    
    Args:
        name: Name to register the provider under
        provider_class: Provider class that inherits from BaseProvider
        
    Raises:
        ValueError: If provider_class doesn't inherit from BaseProvider
    """
    if not issubclass(provider_class, BaseProvider):
        raise ValueError(f"Provider class {provider_class} must inherit from BaseProvider")
    
    PROVIDER_REGISTRY[name] = provider_class
    logger.info(f"Registered provider '{name}' with class {provider_class.__name__}")


def create_provider_with_config(provider_name: str, config: ProviderConfig) -> BaseProvider:
    """
    Create a provider instance with a fully configured ProviderConfig
    
    Args:
        provider_name: Name of the provider
        config: Fully configured ProviderConfig instance
        
    Returns:
        Configured provider instance
        
    Raises:
        ValueError: If provider_name is not supported
    """
    if provider_name not in PROVIDER_REGISTRY:
        available = list(PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown provider '{provider_name}'. Available providers: {available}")
    
    provider_class = PROVIDER_REGISTRY[provider_name]
    
    try:
        provider_instance = provider_class(config)
        logger.info(f"Successfully created {provider_name} provider with custom config")
        return provider_instance
    except Exception as e:
        logger.error(f"Failed to create {provider_name} provider with custom config: {e}")
        raise