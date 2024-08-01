import openai
import numpy as np
import logging

# Set up the logger
logger = logging.getLogger(__name__)

def get_openai_embedding(text: str) -> np.ndarray:
    """
    Generate an embedding for a given text using OpenAI's embedding model.

    Args:
        text (str): The input text to generate an embedding for.

    Returns:
        np.ndarray: The embedding as a NumPy array. If an error occurs, returns a zero vector.
    """
    try:
        # Create an embedding using the OpenAI API
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        # Extract the embedding from the response
        embedding = response['data'][0]['embedding']
        return np.array(embedding)
    except Exception as e:
        # Log any errors that occur during the embedding creation
        logger.error(f"Error generating embedding: {e}")
        # Return a zero vector if there's an error (size may vary based on the model)
        return np.zeros(1536)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity between vec1 and vec2.
    """
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vec1, vec2)
    # Calculate the norm (magnitude) of each vector
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # Calculate and return the cosine similarity
    return dot_product / (norm_vec1 * norm_vec2)
