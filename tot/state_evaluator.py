import openai
from typing import List
import logging
from embedding_utils import get_openai_embedding, cosine_similarity

logger = logging.getLogger(__name__)

def evaluate_states(states: List[str], query: str) -> List[float]:
    """
    Evaluate the given states for their relevance and usefulness in addressing the problem.

    Args:
        states (List[str]): The states to be evaluated.
        query (str): The original problem query.

    Returns:
        List[float]: The combined scores (ratings and similarities) for each state.
    """
    prompt = "Evaluate the following states in terms of their relevance and usefulness for addressing the problem. Rate each state on a scale of 0 to 10, where 10 is the most relevant and useful."
    
    states_text = "\n".join(f"{i+1}. {state}" for i, state in enumerate(states))
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant evaluating problem-solving states."},
        {"role": "user", "content": f"{prompt}\n\n{states_text}"},
        {"role": "user", "content": "Provide only the numerical ratings, one per line:"}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100,
            n=1,
            temperature=0.3
        )

        ratings_text = response.choices[0].message['content'].strip()
        ratings = [float(rating) for rating in ratings_text.split('\n') if rating.replace('.', '').isdigit()]
        
        if len(ratings) < len(states):
            ratings.extend([0.0] * (len(states) - len(ratings)))
        ratings = ratings[:len(states)]
        
        # Calculate OpenAI embeddings for query and states
        query_embedding = get_openai_embedding(query)
        states_embeddings = [get_openai_embedding(state) for state in states]
        similarities = [cosine_similarity(query_embedding, state_embedding) for state_embedding in states_embeddings]
        
        # Combine ratings and similarities
        combined_scores = [(rating + similarity) / 2 for rating, similarity in zip(ratings, similarities)]
        return combined_scores
    except Exception as e:
        logger.error(f"Error in state evaluation: {e}")
        return [0.0] * len(states)
