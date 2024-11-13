from typing import List, Optional, Tuple, Union

import chromadb
from chromadb.config import Settings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic.v1 import SecretStr

from expert_system.chat_reference import ChatReference
from expert_system.parameters import (
    OPENAI_API_KEY,
    VECTORDB_HOST,
    VECTORDB_PASSWORD,
    VECTORDB_PORT,
    VECTORDB_USERNAME,
)
from expert_system.prompts import EXPERT_PROMPT_CONCISE, EXPERT_PROMPT_VERBOSE


def query_template(
    previous_messages: Optional[List[Union[HumanMessage, AIMessage]]] = None,
    prompt: str = EXPERT_PROMPT_VERBOSE,
) -> ChatPromptTemplate:
    messages = [
        ("system", prompt),
        ("user", "{context}"),
    ]

    # TODO: Loop through previous messages and add them to the template based on AI or Human
    if previous_messages is not None:
        for message in previous_messages:
            if isinstance(message, HumanMessage):
                messages.append(("human", str(message.content)))
            elif isinstance(message, AIMessage):
                messages.append(("ai", str(message.content)))
    template = ChatPromptTemplate.from_messages(messages)

    return template


class Chatbot:
    """Singleton class for the Chatbot"""

    _instance = None

    @classmethod
    def instance(cls):
        """Singleton instance of the Chatbot class

        Returns:
            Chatbot: The singleton instance of the Chatbot class
        """
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            # more init operation here
            cls._instance.initialize_vectorstore()
            cls._instance.initialize_qachain()
        return cls._instance

    def __init__(self) -> None:
        raise Exception("Singleton class, cannot instantiate")

    def initialize_vectorstore(self) -> None:
        """Initializes the Vector Store

        Returns:
            Chroma: The vector store database instace that we will be using
        """
        chroma_client = chromadb.HttpClient(
            host=VECTORDB_HOST,
            port=int(VECTORDB_PORT),
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
                chroma_client_auth_credentials=f"{VECTORDB_USERNAME}:{VECTORDB_PASSWORD}",
            ),
        )
        openai_embedding_function = OpenAIEmbeddings(api_key=SecretStr(OPENAI_API_KEY))
        self.vectordb = Chroma(
            collection_name="ced-library",
            embedding_function=openai_embedding_function,
            client=chroma_client,
        )

        # Test the connection
        print(f"Testing connection to VectorDB (find a number > 0):{chroma_client.heartbeat()}")

    def initialize_qachain(self) -> None:
        """Initializes the QA Chain"""

        self.llm = ChatOpenAI(
            api_key=SecretStr(OPENAI_API_KEY),
            temperature=0.1,
            model="gpt-4",
            verbose=True,
            max_tokens=1500,
        )

    def converse(
        self,
        user_input: str,
        previous_messages: Optional[List[Union[HumanMessage, AIMessage]]] = None,
    ) -> Tuple[str, List[ChatReference]]:
        """Converse with the chatbot

        Args:
            user_input (str): Question from the user
            previous_messages (list, optional): List of previous messages. Defaults to [].

        Returns:
            Tuple[str, List[ChatReference]]: Response from the chatbot and the list of references
        """
        if previous_messages is None:
            previous_messages = []
        previous_messages.append(HumanMessage(content=user_input))

        prompt = query_template(
            previous_messages=previous_messages,
        )

        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
        )

        retriever = self.vectordb.as_retriever()

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        result = retrieval_chain.invoke({"input": user_input})
        print("Result from qachain:")
        print(result)
        refernces_list = []
        for source_document in result["context"]:
            ref = ChatReference(
                title=source_document.metadata["title"],
                description="Coming Soon...",
                context=source_document.page_content,
                ref_url="Coming Soon...",
            )
            refernces_list.append(ref)
        return result["answer"], refernces_list

    def converse_concise(
        self,
        user_input: str,
        previous_messages: Optional[List[Union[HumanMessage, AIMessage]]] = None,
    ) -> Tuple[str, List[ChatReference]]:
        """Converse with the chatbot (concise)

        Args:
            user_input (str): Question from the user
            previous_messages (list, optional): List of previous messages. Defaults to [].

        Returns:
            Tuple[str, List[ChatReference]]: Response from the chatbot and the list of references
        """
        if previous_messages is None:
            previous_messages = []
        previous_messages.append(HumanMessage(content=user_input))

        prompt = query_template(
            previous_messages=previous_messages,
            prompt=EXPERT_PROMPT_CONCISE,
        )

        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
        )

        retriever = self.vectordb.as_retriever()

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        result = retrieval_chain.invoke({"input": user_input})
        print("Result from qachain:")
        print(result)
        refernces_list = []
        for source_document in result["context"]:
            ref = ChatReference(
                title=source_document.metadata["title"],
                description="Coming Soon...",
                context=source_document.page_content,
                ref_url="Coming Soon...",
            )
            refernces_list.append(ref)
        return result["answer"], refernces_list
