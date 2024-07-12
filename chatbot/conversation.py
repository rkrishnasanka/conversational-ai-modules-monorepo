from re import template
from typing import Any, List, Literal, Optional, Tuple, Union

import chromadb
from chromadb.config import Settings
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.llms import OpenAIChat
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.prompts.chat import BaseMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from pydantic.v1 import SecretStr

from chatbot.chat_reference import ChatReference
from chatbot.parameters import (
    OPENAI_API_KEY,
    VECTORDB_HOST,
    VECTORDB_PASSWORD,
    VECTORDB_PERSIST_DIRECTORY,
    VECTORDB_PORT,
    VECTORDB_USERNAME,
)
from discord_bot.parameters import OUTPUT_COLUMNS


def query_template(output_columns, previous_messages: Optional[List[Union[HumanMessage, AIMessage]]] = None):
    messages = [
        (
            "system",
            """The Input: A user input, data retrieved for answering the user input and the chat history. sometimes the retrieved data contain the url/links of the products highlight them and also sometimes the data retrieved can be none.
         Act as: Act as a consultant and subject matter expert educating, by the provided context, on the topic of Evidence-based Medical Cannabis. The material sourced for the output script should prioritize primary resources and sources of information of the highest academic quality, including meta-analyses, randomized controlled trials, and other clinical-trial high quality data, reviews, and publications. Published, peer-reviewed data should be prioritized over expert opinion, and non-published information and/or non-expert opinions should be disregarded when mining for source materials.
The Goal for the output: The goal of this output is to answer the following question from a someone looking to learn about medical cannabis according to the retrieved data. 
The Output: The length of each response should be concise, taking information from the provided conext and utilizing the following guidelines:
1. Reading level: The reading level of the material should be no more advanced than a 12th grade reading level. Where possible, scientific jargon or words should be minimized or preferentially traded for more simplified language with explanations.
2. Perspective: The response of the work should be coming from the perspective of an expert researcher / medical clinician author and medical authority, speaking to a generally educated audience.
3. Goal: for the purposes of providing clear and concise, simply understood and evidence-based education to an under-informed readership.
4. Tone: should be pleasant, approachable, not condescending to the reader but framed with optimism and positivity, where appropriate.
5. The Material Referenced: Where product-specific information is poignant to the output, please refer to material relevant for both THC and CBD, the provided context and not merely one or the other.
6. Comparative benefit: Please include at least a sentence covering the cross-comparison of cannabinoid therapy with existing traditional choices. Highlight some differences in benefits and where relevant, describe how cannabis works physiologically, as compared with other traditional substance-based treatments.
7. Delivery methods: Consider and please refer to the numerous medicalized non-combustible forms of cannabis currently available, and refer to some of the known differences in delivery (timing of action, duration, strengths, delivery methods and that this isn’t just about smoking/vaping.
8. Risks - in once sentence, please describe who should avoid cannabinoid therapies and, from a high level. 
9. Conclusion/next steps - at the conclusion of the output, please include a reasonable conclusion for the listener, for example to consult with your primary care doctor and/or an expert cannabis clinician for ongoing care and clinical guidance.
Important Note: Answers should only be sourced from the provided context or data and analysis mined from published, peer-reviewed scientific literature. Under no circumstances should creativity or fabricated information find its way into outputs at any time. Where low-quality data may have been sourced for information shared, it should be indicated by the following parenthetical phrase at the end of the relevant sentence “(sourced from potentially low quality sources)”

Other Notes: in the output, avoid self-referencing to the speaker. Simply jump into education rather than referencing the speaker. Avoid self-referencing like AI, Assistant, Chatbot, Bot, etc.""",
        ),
    ]

    # TODO: Loop through previous messages and add them to the template based on AI or Human
    if previous_messages is not None:
        for message in previous_messages:
            if isinstance(message, HumanMessage):
                if not output_columns:
                    messages.append(("human", message.content))
                else:
                    messages.append(
                        (
                            "human",
                            message.content
                            + "I specifically only want to know about the columns: "
                            + " ".join(output_columns),
                        )
                    )
            elif isinstance(message, AIMessage):
                messages.append(("ai", message.content))
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

    def initialize_qachain(self) -> None:
        """Initializes the QA Chain"""

        llm = ChatOpenAI(api_key=SecretStr(OPENAI_API_KEY), temperature=0.9, model="gpt-4")

        self.qachain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(),
            return_source_documents=True,
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

        prompt = query_template(output_columns=OUTPUT_COLUMNS, previous_messages=previous_messages)
        result = self.qachain({"query": prompt.format(user_question=user_input)})

        refernces_list = []
        for source_document in result["source_documents"]:
            ref = ChatReference(
                title=source_document.metadata["title"],
                description="Coming Soon...",
                context=source_document.page_content,
                ref_url="Coming Soon...",
            )
            refernces_list.append(ref)
        return result["result"], refernces_list
