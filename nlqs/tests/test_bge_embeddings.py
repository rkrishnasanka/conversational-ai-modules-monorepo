import pytest
from typing import List


def test_bge_embedding_function_returns_embeddings(bge_embedding_function):
    """Test that BGE embedding function returns valid embeddings."""
    # Test with a simple query
    query = "Show me all records with high values"
    
    # Get embeddings
    embeddings = bge_embedding_function(query)
    
    # Assertions
    assert embeddings is not None, "Embeddings should not be None"
    assert isinstance(embeddings, list), "Embeddings should be a list"
    assert len(embeddings) == 384, "BGE-small-en-v1.5 should return 384-dimensional embeddings"
    assert all(isinstance(x, float) for x in embeddings), "All embedding values should be floats"


def test_bge_embedding_function_consistency(bge_embedding_function):
    """Test that BGE embedding function returns consistent embeddings for the same input."""
    query = "Find all users named John"
    
    # Get embeddings twice
    embeddings1 = bge_embedding_function(query)
    embeddings2 = bge_embedding_function(query)
    
    # Should be identical
    assert embeddings1 == embeddings2, "Same query should produce identical embeddings"


# def test_bge_vs_azure_embedding_dimensions(bge_embedding_function, embedding_function):
#     """Test that BGE and Azure embeddings have different dimensions."""
#     query = "test query"
    
#     bge_embeddings = bge_embedding_function(query)
#     azure_embeddings = embedding_function(query)
    
#     # BGE should be 384-dimensional, Azure should be 1536-dimensional
#     assert len(bge_embeddings) == 384, "BGE embeddings should be 384-dimensional"
#     assert len(azure_embeddings) == 1536, "Azure OpenAI embeddings should be 1536-dimensional"


def test_bge_embedding_different_queries(bge_embedding_function):
    """Test that different queries produce different embeddings."""
    query1 = "Show me all records"
    query2 = "Find users with high scores"
    
    embeddings1 = bge_embedding_function(query1)
    embeddings2 = bge_embedding_function(query2)
    
    # Should be different
    assert embeddings1 != embeddings2, "Different queries should produce different embeddings"