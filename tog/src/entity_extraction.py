import os
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

class EntityExtractor:
    """Class for extracting entities from text queries using Azure OpenAI."""
    
    def __init__(self):
        """Initialize the Azure OpenAI client for entity extraction."""
        try:
            # Configure client with Azure OpenAI credentials
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                deployment_name=AZURE_OPENAI_DEPLOYMENT,
                api_version=AZURE_OPENAI_API_VERSION
            )
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise

    def extract_entities(self, query: str, entity_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract entities from the given query.
        
        Args:
            query: The text query to extract entities from
            entity_types: Optional list of entity types to focus on (e.g., ['person', 'organization', 'topic'])
                          If None, all entities will be extracted
        
        Returns:
            Dictionary containing extracted entities grouped by type
        """
        if not query:
            logger.warning("Empty query provided for entity extraction")
            return {"entities": {}}

        try:
            # Default entity types if none provided
            if not entity_types:
                entity_types = ["topic", "person", "organization", "location", "product", "event", "concept"]
            
            # Create system prompt for entity extraction
            system_prompt = f"""
            You are an expert entity extraction system. Extract named entities from the provided text.
            Focus on the following entity types: {', '.join(entity_types)}.
            Return the result as a JSON object with entity types as keys and lists of extracted entities as values.
            Only include entities that are actually present in the text.
            """
            
            # Create user message with the query
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract entities from this text: {query}"}
            ]
            
            # Call Azure OpenAI API
            response = self.client.chat(
                engine=self.deployment_name,
                messages=messages,
                temperature=0,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={"type": "json_object"}
            )
            
            # Extract and return the result
            result = response.choices[0].message.content
            
            # Log the result for debugging
            logger.debug(f"Extracted entities: {result}")
            
            import json
            return json.loads(result)
        
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": {}, "error": str(e)}

    def extract_topics(self, query: str) -> List[str]:
        """
        Extract only topic entities from the given query.
        
        Args:
            query: The text query to extract topic entities from
        
        Returns:
            List of topic entities
        """
        result = self.extract_entities(query, entity_types=["topic"])
        
        # Extract topics from the result
        topics = result.get("topic", [])
        if not topics:
            # Try alternate key formats that might be returned
            topics = result.get("topics", [])
        
        return topics


# Example usage
if __name__ == "__main__":
    extractor = EntityExtractor()
    query = "Can you tell me about the latest developments in renewable energy and its impact on climate change?"
    entities = extractor.extract_entities(query)
    print(f"Extracted entities: {entities}")
    
    topics = extractor.extract_topics(query)
    print(f"Extracted topics: {topics}")
