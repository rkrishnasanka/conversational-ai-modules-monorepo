import os
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging
import json
import re

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
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            self.model = AZURE_OPENAI_DEPLOYMENT
            logger.info("Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise

    def extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text, even if it's embedded in markdown or other text.
        
        Args:
            text: The text that may contain JSON
            
        Returns:
            Parsed JSON object or empty dict if parsing fails
        """
        # Try to parse directly first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Look for JSON within markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Look for content that looks like JSON (between curly braces)
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # If all parsing attempts fail, log the issue and return an empty dict
        logger.warning(f"Failed to parse JSON from response: {text[:100]}...")
        return {}

    def extract_entities(self, query: str, prompt: Optional[str] = None, entity_types: Optional[List[str]] = None) -> List[str]:
        """
        Extract entities from the given query and return as a simple list of strings.
        
        Args:
            query: The text query to extract entities from
            prompt: Optional custom prompt to use for entity extraction
            entity_types: Optional list of entity types to focus on (e.g., ['topic'])
                          If None, all entities will be extracted
        
        Returns:
            List of extracted entity strings
        """
        if not query:
            logger.warning("Empty query provided for entity extraction")
            return []

        try:
            # Default entity types if none provided
            if not entity_types:
                entity_types = ["topic"]
            
            # Use custom prompt if provided, otherwise create a default system prompt
            if not prompt:
                system_prompt = f"""
                You are an expert entity extraction system. Extract key entities from the provided text.
                Focus on the following entity types: {', '.join(entity_types)}.
                
                Return ONLY a valid JSON object with a single key called "entities" and a list of entity strings as the value.
                Format your entire response as a valid JSON object with no additional text or explanation.
                
                Example format:
                {{
                  "entities": ["entity1", "entity2", "entity3"]
                }}
                
                Each entity should be a distinct concept or topic mentioned in the text.
                Do not categorize the entities, just list them all under the single "entities" key.
                """
            else:
                system_prompt = prompt
            
            # Create user message with the query
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract entities from this text: {query}"}
            ]
            
            # Call Azure OpenAI API
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"}  # Use structured JSON output if supported
                )
            except Exception as format_error:
                logger.warning(f"JSON format specification failed, trying without it: {format_error}")
                # Fall back to unstructured response if JSON format isn't supported
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2
                )
            
            # Extract the result
            result_text = response.choices[0].message.content
            logger.info(f"Raw response: {result_text[:100]}...")
            
            # Parse JSON from the response
            result_json = self.extract_json_from_text(result_text)
            
            # Extract entities from the parsed JSON
            entities = []
            
            # Try to extract entities from various possible response formats
            if "entities" in result_json:
                # Direct format: {"entities": ["entity1", "entity2"]}
                if isinstance(result_json["entities"], list):
                    entities = result_json["entities"]
            else:
                # Process each entity type and flatten into a single list
                for entity_type in entity_types:
                    if entity_type in result_json:
                        if isinstance(result_json[entity_type], list):
                            entities.extend(result_json[entity_type])
                    
                    # Check for plural form of entity type
                    plural_type = f"{entity_type}s"
                    if plural_type in result_json:
                        if isinstance(result_json[plural_type], list):
                            entities.extend(result_json[plural_type])
                
                # If still no entities found, try to extract from any list in the response
                if not entities:
                    for key, value in result_json.items():
                        if isinstance(value, list) and all(isinstance(item, str) for item in value):
                            entities.extend(value)
            
            # If JSON parsing failed completely, try regex extraction
            if not entities and result_text:
                # Try to extract a list pattern like: ["item1", "item2"] or [item1, item2]
                list_match = re.search(r'\[(.*?)\]', result_text)
                if list_match:
                    items = list_match.group(1)
                    # Split by comma and clean up each item
                    entities = [item.strip().strip('"\'') for item in items.split(',')]
            
            return entities
        
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []


# Example usage
if __name__ == "__main__":
    extractor = EntityExtractor()
    query = "Can you tell me about the latest developments in renewable energy and its impact on climate change?"
    
    # Using default prompt
    print("Extracting entities...")
    entities = extractor.extract_entities(query)
    print(f"Extracted entities: {entities}")
    
    # Using custom prompt
    print("\nExtracting entities with custom prompt...")
    custom_prompt = """
    You are an expert topic analyzer. Identify and extract the main topics being discussed in the given text.
    Return ONLY a valid JSON object with the following format:
    {
      "entities": ["topic1", "topic2", "topic3"]
    }
    Your entire response must be valid JSON with no additional text.
    """
    entities_custom = extractor.extract_entities(query, prompt=custom_prompt)
    print(f"Extracted entities (custom prompt): {entities_custom}")
