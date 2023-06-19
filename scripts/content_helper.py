
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

OPEN_API_KEY = "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"

# Load and process the text
loader = PyPDFLoader("../../content/data/handbook.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
PERSIST_DIRECTORY = '../../content/db'

embedding = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=PERSIST_DIRECTORY)

vectordb.persist()