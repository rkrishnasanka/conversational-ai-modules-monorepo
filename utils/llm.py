from typing import Optional, Union
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import SecretStr
from utils.parameters import (
    AZURE_OPENAI_EMBEDDING_ENDPOINT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    OPENAI_API_KEY,  
)


def get_default_llm(use_azure: bool = True):
    """Returns the default LLM model, either from Azure OpenAI or OpenAI."""
    if use_azure:
        return AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version="2024-08-01-preview",
            model="gpt-4o",
            temperature=0.3,  # Adjusted for precision
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    else:
        return ChatOpenAI(
            model="gpt-4",
            api_key=OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=1500,
            verbose=True,
        )

# REFACTOR: move this embedding function to another file (embedders.py)
def get_default_embedding_function(use_azure: bool = False, use_local: bool = True) -> Union[AzureOpenAIEmbeddings, HuggingFaceEmbeddings]:
    """Returns the default embedding function, either from Azure OpenAI or local HuggingFace model.
    
    Args:
        use_azure: If True, use Azure OpenAI embeddings
        use_local: If True, use local BGE model (default)
    
    Returns:
        Embedding function instance
    """
    if use_local:
        # Use local BGE model
        model_kwargs = {'device': 'cpu'}  # Change to 'cuda' if you have GPU
        encode_kwargs = {'normalize_embeddings': True}
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    elif use_azure:
        return AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
        )
    else:
        raise NotImplementedError("Non-Azure, non-local embeddings are not implemented yet.")
