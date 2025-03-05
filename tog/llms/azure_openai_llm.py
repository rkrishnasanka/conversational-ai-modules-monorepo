import os
import logging
import time
from typing import Dict, Any, List, AsyncGenerator
from openai import AzureOpenAI
from tog.llms.base_llm import BaseLLM

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

class AzureOpenAILLM(BaseLLM):
    """
    Azure OpenAI LLM implementation that uses the Azure OpenAI Service.
    """
    
    def __init__(self, model_name: str = None, api_key: str = None, api_version: str = "2023-05-15", azure_endpoint: str = None, **kwargs):
        """
        Initialize Azure OpenAI client with the given model configuration.
        
        Args:
            model_name: The name of the model to use
            deployment_id: The Azure OpenAI deployment ID (defaults to model_name if not provided)
            api_version: The API version to use
            **kwargs: Additional model-specific configuration parameters
        """
        start_time = time.time()
        super().__init__(model_name, **kwargs)
        logger.info("Initializing AzureOpenAILLM client")
        self.client = AzureOpenAI(
            api_key=api_key or AZURE_OPENAI_API_KEY,
            api_version=api_version or AZURE_OPENAI_API_VERSION,
            azure_endpoint=azure_endpoint or AZURE_OPENAI_ENDPOINT
        )
        self.model_name = model_name or AZURE_OPENAI_DEPLOYMENT
        init_time = time.time() - start_time
        logger.info(f"AzureOpenAILLM initialized with model: {self.model_name} in {init_time:.2f}s")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response for the given prompt using Azure OpenAI.
        
        Args:
            prompt: The input text to generate a response for
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text response
        """
        logger.info(f"Generating response with model: {self.model_name}")
        logger.debug(f"Prompt: {prompt[:50]}...")
        
        start_prep_time = time.time()
        generation_config = self._prepare_generation_config(kwargs)
        prep_time = time.time() - start_prep_time
        logger.debug(f"Generation config prepared in {prep_time:.2f}s: {generation_config}")
        
        try:
            start_api_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **generation_config
            )
            api_time = time.time() - start_api_time
            logger.info(f"Response successfully generated - API call took {api_time:.2f}s")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def generate_stream(self, prompt: str, **kwargs):
        """
        Stream the response for the given prompt using Azure OpenAI.
        
        Args:
            prompt: The input text to generate a response for
            **kwargs: Additional generation parameters
            
        Returns:
            A generator yielding chunks of the generated response
        """
        logger.info(f"Generating streaming response with model: {self.model_name}")
        logger.debug(f"Prompt: {prompt[:50]}...")
        
        start_prep_time = time.time()
        generation_config = self._prepare_generation_config(kwargs)
        prep_time = time.time() - start_prep_time
        logger.debug(f"Generation config prepared in {prep_time:.2f}s: {generation_config}")
        
        try:
            start_api_time = time.time()
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **generation_config
            )
            
            first_token_time = None
            chunk_count = 0
            logger.info("Stream generation started")
            
            for chunk in stream:
                if chunk_count == 0:
                    first_token_time = time.time()
                    ttft = first_token_time - start_api_time
                    logger.info(f"Time to first token: {ttft:.2f}s")
                
                chunk_count += 1
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            total_time = time.time() - start_api_time
            if chunk_count > 0:
                tokens_per_second = chunk_count / total_time
                logger.info(f"Stream generation completed in {total_time:.2f}s, yielded {chunk_count} chunks (~{tokens_per_second:.2f} tokens/sec)")
            else:
                logger.info(f"Stream generation completed in {total_time:.2f}s, no chunks received")
            
        except Exception as e:
            logger.error(f"Error generating stream response: {str(e)}")
            raise
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts in batch mode using Azure OpenAI.
        
        Args:
            prompts: A list of prompts to generate responses for
            **kwargs: Additional generation parameters
            
        Returns:
            A list of generated text responses
        """
        logger.info(f"Batch generating responses for {len(prompts)} prompts")
        results = []
        
        batch_start_time = time.time()
        for i, prompt in enumerate(prompts):
            prompt_start_time = time.time()
            logger.info(f"Processing batch item {i+1}/{len(prompts)}")
            results.append(self.generate(prompt, **kwargs))
            prompt_time = time.time() - prompt_start_time
            logger.info(f"Batch item {i+1} completed in {prompt_time:.2f}s")
            
        batch_time = time.time() - batch_start_time
        logger.info(f"Batch generation completed in {batch_time:.2f}s (avg: {batch_time/len(prompts):.2f}s per prompt)")
        return results
    
    def _prepare_generation_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the generation configuration from kwargs.
        
        Args:
            kwargs: The kwargs provided to the generate method
            
        Returns:
            A dictionary with the generation configuration
        """
        # Default generation parameters
        config = {
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        
        # Add optional parameters if provided
        for param in ["top_p", "frequency_penalty", "presence_penalty", "n"]:
            if param in kwargs:
                config[param] = kwargs[param]
                
        return config
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """
        Return information about the loaded model.
        
        Returns:
            A dictionary containing model information
        """
        logger.info("Getting model information")
        info = super().model_info
        info.update({
            "deployment_id": getattr(self, "deployment_id", self.model_name),
            "api_version": getattr(self, "api_version", None),
            "service": "Azure OpenAI"
        })
        return info

if __name__ == '__main__':
    # Example usage
    llm = AzureOpenAILLM()
    response = llm.generate("What is the capital of France?")
    print(response)