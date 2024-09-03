import re
from typing import List, Optional, Tuple, Union
import chromadb
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic.v1 import SecretStr
from chatbot.chat_reference import ChatReference
from chatbot.parameters import (
    OPENAI_API_KEY,
    VECTORDB_HOST,
    VECTORDB_PASSWORD,
    VECTORDB_PORT,
    VECTORDB_USERNAME,
)


def query_template(
    user_input: str,
    retrieved_data,
    previous_messages: Optional[List[Union[HumanMessage, AIMessage]]] = None,
):
    messages = [
        (
            "system",
            """Act as: A consultant and subject matter expert educating, by the provided context, on the topic of Evidence-based Medical Cannabis. 
        
        The material sourced for the output script should prioritize primary resources and sources of information of the highest academic quality, including meta-analyses, randomized controlled trials, and other high-quality clinical-trial data, reviews, and publications. Published, peer-reviewed data should be prioritized over expert opinion, and non-published information and/or non-expert opinions should be disregarded when mining for source materials.

        The Goal for the Output: The goal of this output is to answer the following question from someone looking to learn about medical cannabis. 

        The Output: The length of each response should be concise, taking information from the provided context and utilizing the following guidelines:

        1. **Handle Basic Greetings**: If the user input is a simple greeting (e.g., "hello", "hi", "hey", "greetings"), respond with a friendly greeting message. Skip the structured analysis and JSON output for these cases.
       - Example response: "Hello! How can I assist you today? and any other appropriate responses."

        2. Reading Level: The reading level of the material should be no more advanced than a 12th-grade reading level. Scientific jargon or words should be minimized or preferentially traded for more simplified language with explanations.

        3. Perspective: The response should come from the perspective of an expert researcher/medical clinician author and medical authority, speaking to a generally educated audience.

        4. Goal: Provide clear and concise, simply understood, and evidence-based education to an under-informed readership.

        5. Tone: Maintain a pleasant, approachable tone that is not condescending to the reader but is framed with optimism and positivity, where appropriate.

        6. The Material Referenced: Where product-specific information is relevant, refer to material for both THC and CBD from the provided context, not merely one or the other.

        7. Comparative Benefit: Include at least a sentence covering the cross-comparison of cannabinoid therapy with existing traditional choices. Highlight some differences in benefits and, where relevant, describe how cannabis works physiologically compared to other traditional substance-based treatments.

        8. Delivery Methods: Consider and refer to the numerous medicalized non-combustible forms of cannabis currently available. Discuss known differences in delivery (timing of action, duration, strengths, delivery methods) and emphasize that this isn’t just about smoking/vaping.

        9. Risks: In one sentence, describe who should avoid cannabinoid therapies and provide a high-level overview.

        10. Conclusion/Next Steps: After the output, include a reasonable conclusion for the listener, such as recommending consultation with a primary care doctor and/or an expert cannabis clinician for ongoing care and clinical guidance.

        Important Note: Answers should only be sourced from the provided context or data and analysis mined from published, peer-reviewed scientific literature. Under no circumstances should creativity or fabricated information find its way into outputs at any time. Where low-quality data may have been sourced, indicate this by using the parenthetical phrase “(sourced from potentially low-quality sources)” at the end of the relevant sentence.

        Other Notes: Avoid self-referencing or mentioning "I," "we," or "AI" in the output. Directly provide the information without referencing the speaker. If you receive any links in the input, please highlight them in the output.""",
        ),
        ("user", f"user input:{user_input} retreived data: {retrieved_data}"),
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

    def initialize_qachain(self) -> None:
        """Initializes the QA Chain"""

        llm = ChatOpenAI(
            api_key=SecretStr(OPENAI_API_KEY), temperature=0.1, model="gpt-4", verbose=True, max_tokens=1500
        )

        self.qachain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(),
            return_source_documents=True,
        )

    def converse(
        self,
        user_input: str,
        retrieved_data,
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
        updated_retrieved_data = re.sub("{|}", "", str(retrieved_data))

        prompt = query_template(
            user_input=user_input,
            retrieved_data=updated_retrieved_data,
            previous_messages=previous_messages,
        )
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
