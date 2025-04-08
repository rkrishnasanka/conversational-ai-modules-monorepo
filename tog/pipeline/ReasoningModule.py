from typing import List, Tuple, Dict, Optional, Any
import logging
import json
import re

from tog.llms import BaseLLM
from tog.models.entity import Entity
from tog.models.relation import Relation


class ReasoningModule:
    """
    Module for reasoning with hybrid knowledge to determine if the given knowledge
    is sufficient to answer a question or to provide clues for further exploration.
    """
    
    def __init__(self, 
                 llm: BaseLLM, 
                 max_depth: int = 3, 
                 system_prompt: str = None,
                 verbose: bool = False):
        """
        Initialize the reasoning module.
        
        Args:
            llm (BaseLLM): Language model for reasoning
            max_depth (int): Maximum exploration depth
            system_prompt (str): System prompt for the LLM
            verbose (bool): Whether to log detailed information
        """
        self.llm = llm
        self.max_depth = max_depth
        self.current_depth = 0
        self.system_prompt = system_prompt or "You are a helpful assistant specialized in knowledge graph reasoning."
        self.logger = logging.getLogger(self.__class__.__name__)
        self.verbose = verbose
        
        if verbose:
            logging.basicConfig(level=logging.INFO)
        
    def evaluate_knowledge(self, 
                          query: str,
                          triple_paths: List[Tuple[Entity, Relation, Entity]],
                          context_chunks: Dict[str, str] = None,
                          previous_clues: str = None) -> Dict[str, Any]:
        """
        Evaluate if the knowledge is sufficient to answer the question.
        If yes, return the answer. If not, return clues for further exploration.
        
        Args:
            query (str): The original query
            triple_paths (List[Tuple[Entity, Relation, Entity]]): Knowledge graph triple paths
            context_chunks (Dict[str, str]): Additional context chunks for each entity
            previous_clues (str): Clues from previous iterations
            
        Returns:
            Dict with keys:
            - 'is_sufficient' (bool): Whether knowledge is sufficient
            - 'answer' (str, optional): The answer if knowledge is sufficient
            - 'clues' (str, optional): Clues for further exploration if knowledge is insufficient
            - 'optimized_query' (str, optional): Optimized query for next iteration
        """
        self.current_depth += 1
        
        # Check if we've reached maximum depth
        if self.current_depth > self.max_depth:
            self.logger.info(f"Reached maximum depth ({self.max_depth}). Forcing answer generation.")
            return self._generate_final_answer(query, triple_paths, context_chunks, previous_clues)
        
        # Extract metadata from triple objects
        triple_metadata = self._extract_triple_metadata(triple_paths)
        
        # Format context chunks if available
        formatted_context = self._format_context_chunks(context_chunks)
        
        # Create reasoning prompt
        prompt = self._create_reasoning_prompt(
            query=query,
            triple_metadata=triple_metadata,
            context=formatted_context,
            previous_clues=previous_clues,
            current_depth=self.current_depth,
            max_depth=self.max_depth
        )
        
        # Query LLM for reasoning
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.generate(messages, temperature=0.3)
        
        # Parse LLM response
        return self._parse_reasoning_response(response, query)
    
    def _extract_triple_metadata(self, triple_paths: List[Tuple[Entity, Relation, Entity]]) -> str:
        """
        Extract and format metadata from triple paths.
        
        Args:
            triple_paths: List of (source_entity, relation, target_entity) tuples
            
        Returns:
            Formatted string of triple paths with metadata
        """
        if not triple_paths:
            return "No knowledge graph triples available."
        
        formatted_triples = []
        
        for idx, (source, relation, target) in enumerate(triple_paths, 1):
            # Extract relevant metadata for each component
            source_metadata = self._extract_entity_metadata(source)
            relation_metadata = self._extract_relation_metadata(relation)
            target_metadata = self._extract_entity_metadata(target)
            
            # Format the triple with metadata
            triple_str = f"Triple {idx}:\n"
            triple_str += f"  Source: {source.name} ({source.type})\n"
            triple_str += f"    {source_metadata}\n"
            triple_str += f"  Relation: {relation.type}\n"
            triple_str += f"    {relation_metadata}\n"
            triple_str += f"  Target: {target.name} ({target.type})\n"
            triple_str += f"    {target_metadata}\n"
            
            formatted_triples.append(triple_str)
        
        return "\n".join(formatted_triples)
    
    def _extract_entity_metadata(self, entity: Entity) -> str:
        """Extract and format relevant metadata from an entity."""
        metadata_items = []
        
        # Always include entity ID
        metadata_items.append(f"ID: {entity.id}")
        
        # Include description if available
        if entity.metadata and "description" in entity.metadata:
            metadata_items.append(f"Description: {entity.metadata['description']}")
        
        # Include relevance score if available
        if entity.metadata and "relevance_score" in entity.metadata:
            score = entity.metadata["relevance_score"]
            metadata_items.append(f"Relevance: {score:.2f}")
        
        # Include additional metadata that might be useful
        for key, value in entity.metadata.items():
            if key not in ["description", "relevance_score", "id", "name"]:
                # Format based on data type
                if isinstance(value, (int, float)):
                    metadata_items.append(f"{key}: {value}")
                elif isinstance(value, str) and len(value) < 100:  # Only include short string values
                    metadata_items.append(f"{key}: {value}")
        
        return "; ".join(metadata_items)
    
    def _extract_relation_metadata(self, relation: Relation) -> str:
        """Extract and format relevant metadata from a relation."""
        metadata_items = []
        
        # Always include relation ID
        metadata_items.append(f"ID: {relation.id}")
        
        # Include description if available
        if relation.metadata and "description" in relation.metadata:
            metadata_items.append(f"Description: {relation.metadata['description']}")
        
        # Include direction information
        if relation.metadata and "is_incoming" in relation.metadata:
            is_incoming = relation.metadata["is_incoming"]
            direction = "Incoming" if is_incoming else "Outgoing"
            metadata_items.append(f"Direction: {direction}")
        
        # Include source and target names if available
        if relation.metadata and "source_name" in relation.metadata:
            metadata_items.append(f"Source: {relation.metadata['source_name']}")
        if relation.metadata and "target_name" in relation.metadata:
            metadata_items.append(f"Target: {relation.metadata['target_name']}")
        
        # Include relevance score if available
        if relation.metadata and "relevance_score" in relation.metadata:
            score = relation.metadata["relevance_score"]
            metadata_items.append(f"Relevance: {score:.2f}")
        
        # Include additional metadata that might be useful
        for key, value in relation.metadata.items():
            if key not in ["description", "relevance_score", "is_incoming", "source_name", "target_name"]:
                # Format based on data type
                if isinstance(value, (int, float)):
                    metadata_items.append(f"{key}: {value}")
                elif isinstance(value, str) and len(value) < 100:  # Only include short string values
                    metadata_items.append(f"{key}: {value}")
        
        return "; ".join(metadata_items)
    
    def _format_context_chunks(self, context_chunks: Dict[str, str]) -> str:
        """Format context chunks for inclusion in the prompt."""
        if not context_chunks:
            return "No additional context available."
        
        formatted_chunks = []
        
        for entity_id, context in context_chunks.items():
            formatted_chunks.append(f"Context for Entity {entity_id}:\n{context}\n")
            
        return "\n".join(formatted_chunks)
    
    def _create_reasoning_prompt(self, 
                               query: str, 
                               triple_metadata: str,
                               context: str,
                               previous_clues: str = None,
                               current_depth: int = 1,
                               max_depth: int = 3) -> str:
        """Create the prompt for reasoning with hybrid knowledge."""
        
        prompt = f"""
        I'll provide you with a query and knowledge retrieved from a knowledge graph. 
        Your task is to evaluate whether the provided knowledge is sufficient to answer the query.

        === QUERY ===
        {query}

        === KNOWLEDGE GRAPH TRIPLES ===
        {triple_metadata}

        === ADDITIONAL CONTEXT ===
        {context}
        """
        
        if previous_clues:
            prompt += f"""
            === CLUES FROM PREVIOUS ITERATION ===
            {previous_clues}
            """
        
        prompt += f"""
        === YOUR TASK ===
        Step 1: Analyze all the provided information carefully.
        Step 2: Determine if the knowledge is sufficient to answer the query.
        Step 3: Provide your response in the following JSON format:
        
        ```json
        {{
            "is_sufficient": true/false,
            "reasoning": "Your step-by-step reasoning process about whether the knowledge is sufficient",
            "answer": "If knowledge is sufficient, provide the complete answer here based on the provided knowledge",
            "missing_information": "If knowledge is NOT sufficient, list specific missing information needed",
            "clues": "If knowledge is NOT sufficient, provide helpful clues based on existing knowledge",
            "optimized_query": "If knowledge is NOT sufficient, provide an optimized query for the next iteration"
        }}
        ```

        Current exploration depth: {current_depth}/{max_depth}
        
        Important: If we've nearly reached the maximum depth and the core information seems available 
        even if some details are missing, lean toward providing an answer with appropriate caveats 
        rather than requesting more information.
        """
        
        return prompt
    
    def _parse_reasoning_response(self, response: str, original_query: str) -> Dict[str, Any]:
        """Parse the LLM response to extract reasoning results."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if not json_match:
                self.logger.warning("No JSON found in LLM response. Falling back to default response.")
                return {
                    "is_sufficient": False,
                    "clues": "Unable to extract structured reasoning from LLM response.",
                    "optimized_query": original_query
                }
            
            result = json.loads(json_match.group(0))
            
            # Ensure we have all required fields
            is_sufficient = result.get("is_sufficient", False)
            
            if is_sufficient:
                if "answer" not in result or not result["answer"]:
                    self.logger.warning("LLM claims knowledge is sufficient but provided no answer.")
                    result["answer"] = "Based on the knowledge, an answer should be possible but wasn't properly generated."
            else:
                if "clues" not in result or not result["clues"]:
                    result["clues"] = "No specific clues provided for further exploration."
                
                if "optimized_query" not in result or not result["optimized_query"]:
                    result["optimized_query"] = original_query
            
            return result
            
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse LLM response as JSON: {response}")
            return {
                "is_sufficient": False,
                "clues": "Unable to parse reasoning from LLM response.",
                "optimized_query": original_query
            }
    
    def _generate_final_answer(self, 
                             query: str,
                             triple_paths: List[Tuple[Entity, Relation, Entity]],
                             context_chunks: Dict[str, str] = None,
                             previous_clues: str = None) -> Dict[str, Any]:
        """
        Generate a final answer when maximum depth is reached.
        """
        # Extract metadata from triple objects
        triple_metadata = self._extract_triple_metadata(triple_paths)
        
        # Format context chunks if available
        formatted_context = self._format_context_chunks(context_chunks)
        
        # Create final answer prompt
        prompt = f"""
        We've reached the maximum exploration depth for the following query:
        
        === QUERY ===
        {query}

        === KNOWLEDGE GRAPH TRIPLES ===
        {triple_metadata}

        === ADDITIONAL CONTEXT ===
        {formatted_context}
        """
        
        if previous_clues:
            prompt += f"""
            === CLUES FROM PREVIOUS ITERATIONS ===
            {previous_clues}
            """
        
        prompt += """
        Based on all the information gathered so far, please provide the best possible answer to the query.
        If there are still knowledge gaps, acknowledge them in your answer but provide the most complete
        response possible with the available information.
        
        Format your response as a JSON object:
        ```json
        {
            "is_sufficient": true,
            "answer": "Your comprehensive answer here",
            "confidence": "high/medium/low",
            "knowledge_gaps": "List any significant knowledge gaps that remain"
        }
        ```
        """
        
        # Query LLM for final answer
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.generate(messages, temperature=0.3)
        
        # Parse LLM response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if not json_match:
                self.logger.warning("No JSON found in final answer response.")
                return {
                    "is_sufficient": True,
                    "answer": "Based on the available knowledge, a precise answer couldn't be generated. Consider refining your query or exploring additional sources."
                }
            
            result = json.loads(json_match.group(0))
            
            # Set is_sufficient to True since we're forcing an answer
            result["is_sufficient"] = True
            
            if "answer" not in result or not result["answer"]:
                result["answer"] = "Based on the available knowledge, a precise answer couldn't be generated."
            
            return result
            
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse final answer as JSON: {response}")
            return {
                "is_sufficient": True,
                "answer": "Based on the available knowledge, a precise answer couldn't be properly generated. Consider refining your query or exploring additional sources."
            }
    
    def reset(self):
        """Reset the current depth counter."""
        self.current_depth = 0


# Example usage
if __name__ == "__main__":
    from tog.llms import AzureOpenAILLM
    
    # Initialize components
    llm = AzureOpenAILLM(model_name="gpt-4")
    
    # Create reasoning module
    reasoning = ReasoningModule(llm=llm, max_depth=3, verbose=True)
    
    # Example entity and relation objects
    source_entity = Entity(
        id="3be66527971910fae63df4a4342ba4e0",
        name="Medical Cannabis",
        type="Treatment",
        metadata={"description": "Cannabis plant used for medical purposes."}
    )
    
    relation = Relation(
        id="65dab97f5bde5196a6d0c175b95c63af",
        source_id="3be66527971910fae63df4a4342ba4e0",
        target_id="192db73673d90090cf1cb7d1be13aebc",
        type="Treats",
        metadata={
            "description": "Medical cannabis is used to alleviate symptoms associated with chronic pain.",
            "strength": 0.8,
            "source_name": "Medical Cannabis",
            "target_name": "Chronic Pain"
        }
    )
    
    target_entity = Entity(
        id="192db73673d90090cf1cb7d1be13aebc",
        name="Chronic Pain",
        type="Medical Condition",
        metadata={
            "description": "A long-lasting pain that persists beyond the usual recovery period or accompanies a chronic health condition.",
            "relevance_score": 0.95
        }
    )
    
    # Create triples
    triples = [(source_entity, relation, target_entity)]
    
    # Example context chunks
    context_chunks = {
        source_entity.id: "Medical cannabis, also known as medical marijuana, refers to the use of cannabis and its constituents, THC and CBD, to treat disease or improve symptoms. Evidence suggests it can help with chronic pain, but research is limited due to legal restrictions.",
        target_entity.id: "Chronic pain is pain that lasts longer than 12 weeks despite medication or treatment. Common treatments include medications, therapy, and lifestyle modifications. Some patients report benefits from medical cannabis use."
    }
    
    # Example query
    query = "What are the medical benefits of Medical Cannabis for Chronic Pain?"
    
    # Evaluate knowledge
    result = reasoning.evaluate_knowledge(
        query=query,
        triple_paths=triples,
        context_chunks=context_chunks
    )
    
    print("\nReasoning Result:")
    print(f"Knowledge Sufficient: {result.get('is_sufficient', False)}")
    
    if result.get('is_sufficient', False):
        print(f"Answer: {result.get('answer', 'No answer provided')}")
    else:
        print(f"Clues: {result.get('clues', 'No clues provided')}")
        print(f"Optimized Query: {result.get('optimized_query', query)}")