from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI
from pydantic.v1 import SecretStr
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
            api_key=SecretStr(AZURE_OPENAI_KEY),
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
            api_key=SecretStr(OPENAI_API_KEY),
            temperature=0.3,
            max_tokens=1500,
            verbose=True,
        )


def get_default_embedding_function(use_azure: bool = True):
    """Returns the default embedding function, either from Azure OpenAI or OpenAI."""
    if use_azure:
        return AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=SecretStr(AZURE_OPENAI_KEY),
            azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
        )
    else:
        raise NotImplementedError("Non-Azure embeddings are not implemented yet.")
