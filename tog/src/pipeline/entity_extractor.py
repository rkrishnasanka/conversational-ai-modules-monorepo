from abc import ABC, abstractmethod
from pprint import pprint
from typing import List

from tog.llms.base_llm import BaseLLM
from tog.llms.groq_llm import GroqLLM
from tog.src.models.entity import Entity
import json

from tog.prompts import PromptManager

class EntityExtractor(ABC):
    @abstractmethod
    def extract_entities(self, messages: str) -> List[Entity]:
        """Extract entities from the given text."""
        pass

class LLMExtractor(EntityExtractor):
    def __init__(self, model_name: str):
        self.llm: BaseLLM = self.initialize_llm(model_name)
        pass

    @abstractmethod
    def initialize_llm(self) -> BaseLLM:
        pass

    def extract_entities(self, text: str) -> List[Entity]:
        # Get the extraction prompt
        prompt = PromptManager.get_prompt("extraction_prompt")

        # Extract the system and user prompts
        system_prompt = prompt["system"]
        user_prompt = prompt["user"].replace("{text}", text) # Replace placeholder with input text

        # Call the LLM with the formatted prompt
        response = self.llm.generate([{"role": "system", "content": system_prompt},
                                      {"role": "user", "content": user_prompt}])

        pprint(response)

        # Assuming the response is in a format we can parse
        entities = []
        try:
            # Process the response to extract entities
            # This will depend on the exact format of the LLM response
            # For example, if response is JSON-like:
            parsed_response = json.loads(response)
            
            for item in parsed_response:
                entity = Entity(
                    name=item["name"],
                    type=item["type"],
                    metadata=item.get("metadata", None)
                )
                entities.append(entity)
        except Exception as e:
            # Handle parsing errors
            print(f"Error parsing LLM response: {e}")
            
        return entities
    
class GroqEnityExtractor(LLMExtractor):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def initialize_llm(self, model_name) -> BaseLLM:
        return GroqLLM(model_name)
    

if __name__ == '__main__':
    # Example usage
    llm_extractor = GroqEnityExtractor("mixtral-8x7b-32768")
    entities = llm_extractor.extract_entities("Who is the father of Deep Learning.")
    print(entities)