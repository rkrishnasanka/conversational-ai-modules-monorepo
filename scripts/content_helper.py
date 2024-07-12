"""
This script loads a PDF document, splits it into chunks, embeds the chunks, and stores the embeddings in a VectorDBQA.
"""

from langchain.chains import VectorDBQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic.v1 import SecretStr

OPEN_API_KEY = "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"

# Load and process the text
loader = PyPDFLoader("../../content/data/handbook.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
PERSIST_DIRECTORY = "../../content/db"

embedding = OpenAIEmbeddings(api_key=SecretStr(OPEN_API_KEY))
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=PERSIST_DIRECTORY)

vectordb.persist()
