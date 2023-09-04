from chatbot import Chatbot, create_chat_query_template
from core.parameters import OPENAI_API_KEY
from langchain.vectorstores import Chroma
import chromadb
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import os

from pathlib import Path
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.embeddings.openai import OpenAIEmbeddings


chroma_client = chromadb.HttpClient()
chromadb_client = chromadb.HttpClient(
    host="204.48.26.135", 
    port="8000",
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
        chroma_client_auth_credentials="admin:potter"
    )
)

print(chroma_client.heartbeat())
chroma_client.query()

chat_history = [
    ("human", "Can elaborate on the uses of medical marajuiana for arthritis"),
    ("ai", """Medical marijuana has been found to be helpful in managing symptoms associated with arthritis. Here are some key points to consider:

    Medical marijuana, specifically cannabinoids like THC and CBD, have shown potential in reducing pain and inflammation associated with arthritis. (sourced from potentially low quality sources)
    Studies have shown that cannabinoids interact with the body's endocannabinoid system, which plays a role in regulating pain and inflammation.
    Cannabinoids can be consumed through various methods such as oral capsules, oils, sprays, and topical creams. These non-combustible forms provide alternative options for those who prefer not to smoke or vape.
    Compared to traditional treatments for arthritis, such as nonsteroidal anti-inflammatory drugs (NSAIDs) or opioids, medical marijuana may offer a more natural alternative with potentially fewer side effects. (sourced from potentially low quality sources)
    It is important to note that medical marijuana may not be suitable for everyone. Individuals with certain medical conditions, such as cardiovascular disease, respiratory disorders, or a history of substance abuse, should avoid cannabinoid therapies. It is best to consult with a healthcare professional or a specialist in medical cannabis to determine if it is a suitable option.
    """),
]

prompt = create_chat_query_template(previous_messages=chat_history)

prompt_text = prompt.format(user_question="TEST Question")

qachain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.9, model="gpt-3.5-turbo"), 
            chain_type="stuff", 
            retriever=self.vectordb.as_retriever(),
            return_source_documents=True
        )

result = qachain({"query": prompt.format(user_question=input)})
