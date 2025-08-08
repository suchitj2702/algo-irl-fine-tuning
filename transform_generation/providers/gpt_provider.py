"""
GPT (OpenAI) Provider Implementation

Handles all OpenAI-specific API calls and parameter mappings.
Supports different parameter names across GPT versions (e.g., max_completion_tokens for GPT-5).
"""

import os
from typing import Dict, Any, List
from .base_provider import BaseProvider, ProviderConfig, TaskType

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None


class GPTProvider(BaseProvider):
    """Provider implementation for OpenAI's GPT models"""
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        if not self.config.api_key:
            raise ValueError("OPENAI_API_KEY not found in configuration or environment")
        
        return openai.OpenAI(api_key=self.config.api_key)
    
    def _validate_config(self):
        """Validate GPT-specific configuration"""
        if not self.config.api_key:
            # Try to get from environment
            self.config.api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.config.api_key:
            self.logger.warning("No OPENAI_API_KEY found. API calls will fail.")
        
        # Model name validation removed - providers can use any model name
    
    def generate_completion(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate completion using OpenAI API"""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        def _make_api_call(**call_kwargs):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Build request parameters
            request_params = {
                "model": self.config.model,
                "messages": messages
            }
            
            # Handle different token parameter names based on model
            if self._is_newer_model(self.config.model):
                # GPT-5 and newer models use max_completion_tokens
                request_params["max_completion_tokens"] = self.config.max_output_tokens
            else:
                # Older models use max_tokens
                request_params["max_tokens"] = self.config.max_output_tokens
            
            # Note: Reasoning for OpenAI models (o1, o2) is built into the models themselves
            # and doesn't require explicit API parameters. The models automatically use
            # chain-of-thought reasoning when appropriate.
            # TODO: Implement proper reasoning parameter format if/when available in API
            if False:  # Temporarily disabled: self.config.thinking_enabled and self._supports_reasoning(self.config.model):
                request_params["reasoning"] = {"effort": "medium"}
            
            # Add any custom parameters
            if self.config.custom_params:
                mapped_params = self._map_generic_to_openai_params(self.config.custom_params)
                request_params.update(mapped_params)
            
            # Override with any kwargs
            request_params.update(call_kwargs)
            
            response = self.client.chat.completions.create(**request_params)
            return response.choices[0].message.content
        
        return self._call_with_retry(_make_api_call, **kwargs)
    
    def test_connection(self) -> bool:
        """Test OpenAI API connection"""
        try:
            test_prompt = "Say 'Connection successful' in exactly three words."
            response = self.generate_completion(
                system_prompt="You are a helpful assistant.",
                user_prompt=test_prompt
            )
            self.logger.info(f"OpenAI connection test successful: {response[:50]}...")
            return True
        except Exception as e:
            self.logger.error(f"OpenAI connection test failed: {e}")
            return False
    

    
    @classmethod
    def get_default_configs(cls) -> Dict[TaskType, Dict[str, Any]]:
        """Get default configurations for GPT models"""
        return {
            TaskType.PROBLEM_GENERATION: {
                "model": "gpt-4-turbo-preview",
                "max_output_tokens": 16384,
                "thinking_enabled": True,
                "thinking_budget": 8192
            },
            TaskType.COMPANY_DATA_GENERATION: {
                "model": "gpt-3.5-turbo",
                "max_output_tokens": 5000,
                "thinking_enabled": False,
                "thinking_budget": 0
            },
            TaskType.CUSTOM_PROMPT_TRANSFORM: {
                "model": "gpt-5",
                "max_output_tokens": 15000,
                "thinking_enabled": True,
                "thinking_budget": 10000
            }
        }
    
    @classmethod
    def get_provider_name(cls) -> str:
        """Get provider name"""
        return "gpt"
    
    def _is_newer_model(self, model: str) -> bool:
        """
        Check if model uses newer parameter structure (e.g., max_completion_tokens)
        
        Args:
            model: Model name
            
        Returns:
            True if model uses newer parameters
        """
        # Currently no OpenAI models use max_completion_tokens
        # This is future-proofing for when newer models are released
        newer_models = ["gpt-5", "o2-preview", "o2-mini"]  # Future models
        return any(model.startswith(newer) for newer in newer_models)
    
    def _supports_reasoning(self, model: str) -> bool:
        """
        Check if model supports reasoning/thinking features
        
        Args:
            model: Model name
            
        Returns:
            True if model supports reasoning
        """
        reasoning_models = ["o1", "o2", "gpt-4", "gpt-5"]
        return any(model.startswith(reasoning) for reasoning in reasoning_models)
    
    def _map_generic_to_openai_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map generic parameters to OpenAI-specific ones
        
        Args:
            params: Generic parameter dictionary
            
        Returns:
            OpenAI-specific parameter dictionary
        """
        mapped = {}
        
        for key, value in params.items():
            if key == "max_output_tokens":
                # Choose the right parameter name based on model
                if self._is_newer_model(self.config.model):
                    mapped["max_completion_tokens"] = value
                else:
                    mapped["max_tokens"] = value
            elif key == "temperature":
                mapped["temperature"] = value
            elif key == "top_p":
                mapped["top_p"] = value
            elif key == "frequency_penalty":
                mapped["frequency_penalty"] = value
            elif key == "presence_penalty":
                mapped["presence_penalty"] = value
            elif key == "stop":
                mapped["stop"] = value
            else:
                # Pass through unknown parameters
                mapped[key] = value
        
        return mapped