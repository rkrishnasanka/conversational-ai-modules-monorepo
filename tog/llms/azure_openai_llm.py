import os
from typing import Dict, Any, List
from openai import AzureOpenAI
from tog.llms.base_llm import BaseLLM

# Azure OpenAI Configuration
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

class AzureOpenAILLM(BaseLLM):
    """
    Azure OpenAI LLM implementation that uses the Azure OpenAI service.
    """
    
    def __init__(self, model_name: str, api_key: str = None, 
                endpoint: str = None, api_version: str = None, **kwargs):
        """
        Initialize Azure OpenAI client with the given model configuration.
        
        Args:
            model_name: The name of the model to use
            api_key: The Azure OpenAI API key (defaults to environment variable if not provided)
            endpoint: The Azure OpenAI endpoint (defaults to environment variable if not provided)
            api_version: The Azure OpenAI API version (defaults to environment variable if not provided)
            **kwargs: Additional model-specific configuration parameters
        """
        super().__init__(model_name, **kwargs)
        self.client = AzureOpenAI(
            api_key=api_key or AZURE_OPENAI_KEY,
            azure_endpoint=endpoint or AZURE_OPENAI_ENDPOINT,
            api_version=api_version or AZURE_OPENAI_API_VERSION
        )
        self.model_name = model_name
    
    def generate(self, messages: List[Dict], **kwargs) -> str:
        """
        Generate a response for the given prompt using Azure OpenAI.
        
        Args:
            messages: List of message dictionaries with role and content
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text response
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    
    def generate_stream(self, messages: List[Dict], **kwargs):
        """
        Stream the response for the given prompt using Azure OpenAI.
        
        Args:
            messages: List of message dictionaries with role and content
            **kwargs: Additional generation parameters
            
        Returns:
            A generator yielding chunks of the generated response
        """
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts in batch mode using Azure OpenAI.
        
        Args:
            prompts: A list of prompts to generate responses for
            **kwargs: Additional generation parameters
            
        Returns:
            A list of generated text responses
        """
        results = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            result = self.generate(messages, **kwargs)
            results.append(result)
        return results
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """
        Return information about the loaded model.
        
        Returns:
            A dictionary containing model information
        """
        info = super().model_info
        info.update({
            "service": "Azure OpenAI",
            "endpoint": AZURE_OPENAI_ENDPOINT,
            "api_version": AZURE_OPENAI_API_VERSION
        })
        return info