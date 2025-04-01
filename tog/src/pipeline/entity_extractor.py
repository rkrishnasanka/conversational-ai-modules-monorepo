from abc import ABC, abstractmethod
from pprint import pprint
from typing import List

from tog.src.llms.base_llm import BaseLLM
from tog.src.llms.groq_llm import GroqLLM
from tog.src.models.entity import Entity
import json

from tog.src.utils import PromptLoader
from tog.src.models.response import ExtractionResponse
from tog.src.utils.logger import setup_logger

# Set up logger for this module
logger = setup_logger(name="entity_extractor", log_filename="entity_extractor.log")

class EntityExtractor(ABC):
    @abstractmethod
    def extract_entities(self, messages: str) -> List[str]:
        """Extract entities from the given text."""
        pass

class LLMExtractor(EntityExtractor):
    def __init__(self, model_name: str):
        logger.info(f"Initializing LLMExtractor with model: {model_name}")
        self.llm: BaseLLM = self.initialize_llm(model_name)
        self.prompt_loader = PromptLoader()

    @abstractmethod
    def initialize_llm(self) -> BaseLLM:
        pass

    def extract_entities(self, text: str) -> List[Entity]:
        logger.debug(f"Extracting entities from text: {text[:50]}...")
        
        # Get the extraction prompt
        prompt = self.prompt_loader.get_prompt("extraction_prompt")
        logger.debug("Loaded extraction prompt")

        # Extract the system and user prompts
        system_prompt = prompt["system"]
        user_prompt = prompt["user"].replace("{query}", text) # Replace placeholder with input text

        # Call the LLM with the formatted prompt
        logger.info("Calling LLM for entity extraction")
        response = self.llm.generate([{"role": "system", "content": system_prompt},
                                      {"role": "user", "content": user_prompt}])

        # Parse the response using the ExtractionResponse model
        try:
            # Use the helper method to parse the extraction output
            extraction_response = ExtractionResponse.from_extraction_output(response)
            
            # Just return the list of entity strings directly
            entities = extraction_response.entities
            logger.info(f"Successfully extracted {len(entities)} entities")
            logger.debug(f"Extracted entities: {entities}")
        except Exception as e:
            # Handle parsing errors
            logger.error(f"Failed to parse extraction response: {str(e)}")
            raise ValueError(f"Failed to parse extraction response: {e}")
            
        return entities
    
class GroqEntityExtractor(LLMExtractor):
    def __init__(self, model_name: str):
        logger.info(f"Initializing GroqEntityExtractor with model: {model_name}")
        super().__init__(model_name)

    def initialize_llm(self, model_name) -> BaseLLM:
        logger.debug(f"Creating GroqLLM instance with model: {model_name}")
        return GroqLLM(model_name)
    

if __name__ == '__main__':
    # Example usage
    llm_extractor = GroqEntityExtractor("llama-3.3-70b-versatile")
    sample_text = """OpenAI's GPT-4 was released in March 2023 by Sam Altman's team. The model has 1.76 trillion parameters according to some estimates."""
    
    logger.info("Extracting entities from sample text")
    entities = llm_extractor.extract_entities(sample_text)
    pprint(entities)
    print(len(entities))