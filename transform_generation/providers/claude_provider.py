"""
Claude (Anthropic) Provider Implementation

Handles all Anthropic-specific API calls and parameter mappings.
"""

import os
from typing import Dict, Any, List
from .base_provider import BaseProvider, ProviderConfig, TaskType

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None


class ClaudeProvider(BaseProvider):
    """Provider implementation for Anthropic's Claude models"""
    
    def _initialize_client(self):
        """Initialize Anthropic client"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
        
        if not self.config.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in configuration or environment")
        
        return anthropic.Anthropic(api_key=self.config.api_key)
    
    def _validate_config(self):
        """Validate Claude-specific configuration"""
        if not self.config.api_key:
            # Try to get from environment
            self.config.api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not self.config.api_key:
            self.logger.warning("No ANTHROPIC_API_KEY found. API calls will fail.")
        
        # Model name validation removed - providers can use any model name
    
    def generate_completion(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate completion using Anthropic API"""
        if not self.client:
            raise ValueError("Claude client not initialized")
        
        def _make_api_call(**call_kwargs):
            messages = [{"role": "user", "content": user_prompt}]
            
            # Build request parameters
            request_params = {
                "model": self.config.model,
                "max_tokens": self.config.max_output_tokens,
                "system": system_prompt,
                "messages": messages
            }
            
            # Add thinking/reasoning if enabled and supported
            # Note: Thinking feature is currently disabled due to API compatibility issues
            # The exact parameter format may vary by model and API version
            # TODO: Implement proper thinking parameter format once documented
            if False:  # Temporarily disabled: self.config.thinking_enabled and self.config.thinking_budget > 0:
                request_params["thinking"] = {
                    "type": "reasoning",
                    "budget": self.config.thinking_budget
                }
            
            # Add any custom parameters
            if self.config.custom_params:
                request_params.update(self.config.custom_params)
            
            # Override with any kwargs
            request_params.update(call_kwargs)
            
            response = self.client.messages.create(**request_params)
            
            # Extract text content from response
            if hasattr(response, 'content') and response.content:
                text_content = next((block.text for block in response.content if block.type == 'text'), None)
                if text_content:
                    return text_content
                else:
                    raise ValueError("No text content found in Anthropic response")
            else:
                raise ValueError("Invalid response format from Anthropic API")
        
        return self._call_with_retry(_make_api_call, **kwargs)
    
    def test_connection(self) -> bool:
        """Test Anthropic API connection"""
        try:
            test_prompt = "Say 'Connection successful' in exactly three words."
            response = self.generate_completion(
                system_prompt="You are a helpful assistant.",
                user_prompt=test_prompt
            )
            self.logger.info(f"Claude connection test successful: {response[:50]}...")
            return True
        except Exception as e:
            self.logger.error(f"Claude connection test failed: {e}")
            return False
    

    @classmethod
    def get_default_configs(cls) -> Dict[TaskType, Dict[str, Any]]:
        """Get default configurations for Claude models matching algo-irl"""
        return {
            TaskType.PROBLEM_GENERATION: {
                "model": "claude-3-7-sonnet-20250219",
                "max_output_tokens": 16384,
                "thinking_enabled": True,
                "thinking_budget": 8192
            },
            TaskType.COMPANY_DATA_GENERATION: {
                "model": "claude-3-5-haiku-20241022",
                "max_output_tokens": 5000,
                "thinking_enabled": False,
                "thinking_budget": 0
            },
            TaskType.CUSTOM_PROMPT_TRANSFORM: {
                "model": "claude-sonnet-4-20250514",
                "max_output_tokens": 15000,
                "thinking_enabled": True,
                "thinking_budget": 10000
            }
        }
    
    @classmethod
    def get_provider_name(cls) -> str:
        """Get provider name"""
        return "claude"
    
    def _map_generic_to_anthropic_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map generic parameters to Anthropic-specific ones
        
        Args:
            params: Generic parameter dictionary
            
        Returns:
            Anthropic-specific parameter dictionary
        """
        mapped = {}
        
        for key, value in params.items():
            if key == "max_output_tokens":
                mapped["max_tokens"] = value
            elif key == "temperature":
                mapped["temperature"] = value
            elif key == "top_p":
                mapped["top_p"] = value
            else:
                # Pass through unknown parameters
                mapped[key] = value
        
        return mapped