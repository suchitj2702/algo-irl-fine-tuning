"""
Gemini (Google) Provider Implementation

Handles all Google Gemini-specific API calls and parameter mappings.
"""

import os
from typing import Dict, Any, List
from .base_provider import BaseProvider, ProviderConfig, TaskType

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


class GeminiProvider(BaseProvider):
    """Provider implementation for Google's Gemini models"""
    
    def _initialize_client(self):
        """Initialize Google Gemini client"""
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library not installed. Install with: pip install google-generativeai")
        
        if not self.config.api_key:
            raise ValueError("GOOGLE_API_KEY not found in configuration or environment")
        
        # Configure the API key
        genai.configure(api_key=self.config.api_key)
        
        # Return the model instance
        return genai.GenerativeModel(self.config.model)
    
    def _validate_config(self):
        """Validate Gemini-specific configuration"""
        if not self.config.api_key:
            # Try to get from environment
            self.config.api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not self.config.api_key:
            self.logger.warning("No GOOGLE_API_KEY found. API calls will fail.")
        
        # Model name validation removed - providers can use any model name
    
    def generate_completion(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate completion using Google Gemini API"""
        if not self.client:
            raise ValueError("Gemini client not initialized")
        
        def _make_api_call(**call_kwargs):
            # Combine system and user prompts (Gemini doesn't have separate system prompts)
            combined_prompt = f"{system_prompt}\n\nUser: {user_prompt}"
            
            # Build generation config
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=self.config.max_output_tokens,
                temperature=call_kwargs.get('temperature', 0.7),
                top_p=call_kwargs.get('top_p', 0.9),
                top_k=call_kwargs.get('top_k', 40)
            )
            
            # Add any custom parameters from config
            if self.config.custom_params:
                mapped_params = self._map_generic_to_gemini_params(self.config.custom_params)
                for key, value in mapped_params.items():
                    if hasattr(generation_config, key):
                        setattr(generation_config, key, value)
            
            # Generate content
            response = self.client.generate_content(
                combined_prompt,
                generation_config=generation_config
            )
            
            # Extract text from response
            if response.text:
                return response.text
            else:
                raise ValueError("No text content found in Gemini response")
        
        return self._call_with_retry(_make_api_call, **kwargs)
    
    def test_connection(self) -> bool:
        """Test Gemini API connection"""
        try:
            test_prompt = "Say 'Connection successful' in exactly three words."
            response = self.generate_completion(
                system_prompt="You are a helpful assistant.",
                user_prompt=test_prompt
            )
            self.logger.info(f"Gemini connection test successful: {response[:50]}...")
            return True
        except Exception as e:
            self.logger.error(f"Gemini connection test failed: {e}")
            return False
    

    
    @classmethod
    def get_default_configs(cls) -> Dict[TaskType, Dict[str, Any]]:
        """Get default configurations for Gemini models"""
        return {
            TaskType.PROBLEM_GENERATION: {
                "model": "gemini-1.5-pro-latest",
                "max_output_tokens": 16384,
                "thinking_enabled": True,
                "thinking_budget": 8192,
                "custom_params": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            TaskType.COMPANY_DATA_GENERATION: {
                "model": "gemini-1.5-flash-latest",
                "max_output_tokens": 5000,
                "thinking_enabled": False,
                "thinking_budget": 0,
                "custom_params": {
                    "temperature": 0.5,
                    "top_p": 0.8
                }
            },
            TaskType.CUSTOM_PROMPT_TRANSFORM: {
                "model": "gemini-2.5-pro",
                "max_output_tokens": 15000,
                "thinking_enabled": True,
                "thinking_budget": 10000,
                "custom_params": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
        }
    
    @classmethod
    def get_provider_name(cls) -> str:
        """Get provider name"""
        return "gemini"
    
    def _map_generic_to_gemini_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map generic parameters to Gemini-specific ones
        
        Args:
            params: Generic parameter dictionary
            
        Returns:
            Gemini-specific parameter dictionary
        """
        mapped = {}
        
        for key, value in params.items():
            if key == "max_output_tokens":
                mapped["max_output_tokens"] = value
            elif key == "temperature":
                mapped["temperature"] = value
            elif key == "top_p":
                mapped["top_p"] = value
            elif key == "top_k":
                mapped["top_k"] = value
            elif key == "stop_sequences":
                mapped["stop_sequences"] = value
            else:
                # Pass through unknown parameters
                mapped[key] = value
        
        return mapped
    
    def _supports_thinking(self, model: str) -> bool:
        """
        Check if model supports thinking/reasoning features
        Note: Gemini's thinking capabilities are built into the models
        rather than being explicit parameters
        
        Args:
            model: Model name
            
        Returns:
            True if model supports advanced reasoning
        """
        # 1.5 Pro models generally have better reasoning capabilities
        return "1.5-pro" in model.lower()