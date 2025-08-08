"""
Base Provider Abstract Class

Defines the common interface that all LLM providers must implement.
This allows the ProblemTransformer to work with any provider without
knowing the specific implementation details.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks that can be performed with LLM providers"""
    PROBLEM_GENERATION = "problemGeneration"
    COMPANY_DATA_GENERATION = "companyDataGeneration"
    CUSTOM_PROMPT_TRANSFORM = "customPromptTransform"


@dataclass
class ProviderConfig:
    """Base configuration that all providers use"""
    task_type: TaskType
    model: str
    api_key: Optional[str] = None
    thinking_enabled: bool = False
    thinking_budget: int = 0
    max_output_tokens: int = 15000  # Generic name for output token limit
    custom_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize custom_params if not provided"""
        if self.custom_params is None:
            self.custom_params = {}


class BaseProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    def __init__(self, config: ProviderConfig):
        """Initialize the provider with configuration"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Validate required configuration
        self._validate_config()
        
        # Initialize the client
        self.client = self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider-specific client (e.g., OpenAI, Anthropic)"""
        pass
    
    @abstractmethod
    def _validate_config(self):
        """Validate provider-specific configuration requirements"""
        pass
    
    @abstractmethod
    def generate_completion(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Generate a completion using the provider's API
        
        Args:
            system_prompt: The system/instruction prompt
            user_prompt: The user's input prompt
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The generated text completion
            
        Raises:
            Exception: If the API call fails after retries
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the provider connection is working
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    

    @classmethod
    @abstractmethod
    def get_default_configs(cls) -> Dict[TaskType, Dict[str, Any]]:
        """
        Get default configurations for different task types
        
        Returns:
            Dictionary mapping TaskType to configuration parameters
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_provider_name(cls) -> str:
        """
        Get the name/identifier for this provider
        
        Returns:
            Provider name (e.g., "claude", "gpt", "gemini")
        """
        pass
    
    def _call_with_retry(self, api_call_func, max_retries: int = 3, **kwargs) -> str:
        """
        Call an API function with retry logic and exponential backoff
        
        Args:
            api_call_func: Function to call
            max_retries: Maximum number of retry attempts
            **kwargs: Arguments to pass to the API function
            
        Returns:
            API response content
            
        Raises:
            Exception: If all retries fail
        """
        import time
        
        for attempt in range(max_retries):
            try:
                return api_call_func(**kwargs)
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
        
        return ""  # Should be unreachable
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration for logging/debugging
        
        Returns:
            Dictionary with configuration details
        """
        return {
            "provider": self.get_provider_name(),
            "model": self.config.model,
            "task_type": self.config.task_type.value,
            "thinking_enabled": self.config.thinking_enabled,
            "thinking_budget": self.config.thinking_budget,
            "max_output_tokens": self.config.max_output_tokens,
            "has_api_key": bool(self.config.api_key),
            "custom_params": self.config.custom_params
        }