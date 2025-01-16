from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI
from pydantic.v1 import SecretStr

from utils.parameters import (
    AZURE_OPENAI_EMBEDDING_ENDPOINT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    OPENAI_API_KEY,
)


def get_default_llm():
    # llm = ChatOpenAI(model="gpt-4", api_key=SecretStr(OPENAI_API_KEY), temperature=0.3)  # Adjust for precision
    # Old Expert Setting was using
    # ChatOpenAI(
    #         api_key=SecretStr(OPENAI_API_KEY),
    #         temperature=0.1,
    #         model="gpt-4",
    #         verbose=True,
    #         max_tokens=1500,
    #     )

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=SecretStr(AZURE_OPENAI_KEY),
        api_version="2024-08-01-preview",
        model="gpt-4o",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    return llm


def get_default_embedding_function():
    # Return the Azure embeddings

    embedding_function = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",  # Note, we are using this for the Cannabis Chatbot
        api_key=SecretStr(AZURE_OPENAI_KEY),
        azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
    )
    return embedding_function
