from discord_bot.parameters import OPENAI_API_KEY
import pandas as pd
import sqlite3
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from pydantic.v1 import SecretStr


# Load data from CSV and initialize FAISS vector store
loader = CSVLoader(file_path='product_descriptions.csv', encoding='ISO-8859-1')
data = loader.load()
db_csv = Chroma.from_documents(data, OpenAIEmbeddings(api_key=SecretStr(OPENAI_API_KEY)))

# SQLite database file
DB_FILE = "aegion.db"

# Fetch data from SQLite
def fetch_data_from_sqlite():
    try:
        conn = sqlite3.connect(DB_FILE)
        query = "SELECT * FROM new_dataset"
        df = pd.read_sql_query(query, conn)
    except sqlite3.Error as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    finally:
        conn.close()
    return df

df = fetch_data_from_sqlite()

# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-4", api_key=OPENAI_API_KEY)

# Default system prompt
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Function to get the prompt
def get_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    return f"[INST]{SYSTEM_PROMPT}{instruction}[/INST]"

# Function to identify qualitative and quantitative data and user intent
def summarize(user_input, chat_history):
    # Check for empty or whitespace-only input
    if not user_input.strip():
        response = "Please provide a valid input."
        chat_history.append((user_input, response))
        return response, chat_history

    # First summarize the input
    instruction = f"Summarize the following user input for me: {user_input}"
    system_prompt = "You are an expert in summarization and expressing key ideas succinctly."
    prompt = get_prompt(instruction, system_prompt)
    prompt_template = PromptTemplate(template=prompt, input_variables=["user_input"])
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    summarized_input = llm_chain.run({"user_input": user_input})
    print(summarized_input)

    # Analyze the summarized input to find user intent
    return analyze_intent(user_input, summarized_input, chat_history)

# Function to analyze summary and determine if an SQL query can be generated
def analyze_summary(user_input, summarized_input, chat_history):
    instruction = f"Analyse the following summary and tell me if we can create an SQL query. The columns in the database were Product, Category, MedicalBenefits, CustomerRating ,PurchaseFrequency ,description. Name of the table is new_dataset. Just answer only TRUE or FALSE. Summary: {summarized_input}"
    system_prompt = "You are an expert in analyzing data."
    prompt = get_prompt(instruction, system_prompt)
    prompt_template = PromptTemplate(template=prompt, input_variables=["user_input"])
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    analysis_result = llm_chain.run({"user_input": summarized_input}).strip().lower()
    print(f"analysis result: {analysis_result}")
    if analysis_result == "true":
        return generate_query(user_input, summarized_input, chat_history)
    else:
        return similarity_search(user_input, summarized_input, chat_history)    

# Function to analyze user intent
def analyze_intent(user_input, summarized_input, chat_history):
    # Instruction to classify user intent
    instruction = f"Classify the user's intent based on the summary. The summary is: {summarized_input}\n\nPossible intents: phatic_communication, sql_injection, profinity and other. Provide only the intent."
    system_prompt = "You are an expert in natural language understanding."
    prompt = get_prompt(instruction, system_prompt)
    prompt_template = PromptTemplate(template=prompt, input_variables=["user_input"])
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    intent = llm_chain.run({"user_input": summarized_input}).strip().lower()
    print(f"User intent: {intent}")

    if intent == "phatic_communication":
        return generate_greeting_response(user_input, chat_history)
    elif intent == "sql_injection":
        return sql_injection(user_input, chat_history)
    elif intent == "profinity":
        return generate_default_response(user_input, chat_history)
    elif intent == "other":
        return analyze_summary(user_input, summarized_input, chat_history)
        # return analyze_similarity_result(user_input, summarized_input, result, chat_history)
        # return handle_similarity_search(user_input, summarized_input, chat_history)

def similarity_search(user_input,summarized_input, chat_history):
        result = db_csv.similarity_search(user_input)
        result = result[0].page_content
        print(result)
        # return analyze_similarity_result(user_input, summarized_input, result, chat_history)
        return generate_response_non_query(user_input, summarized_input, result, chat_history)

# def analyze_similarity_result(user_input, summarized_input, result, chat_history):
#     instruction = f"Analyse the following user_input, result and tell me if result matches what the user is looking for. Just answer only TRUE or FALSE. user_input: {user_input}\n\n result: {result}"
#     system_prompt = "You are an expert in analyzing data."
#     prompt = get_prompt(instruction, system_prompt)
#     prompt_template = PromptTemplate(template=prompt, input_variables=["user_input"])
#     llm_chain = LLMChain(prompt=prompt_template, llm=llm)

#     analysis_result = llm_chain.run({"user_input": user_input}).strip().lower()
#     print(f"analyze_similarity_search:{analysis_result}")
#     if analysis_result == "true":
#         return generate_response_non_query(user_input, summarized_input, result, chat_history)
#     else:
#         return generate_default_response(user_input, chat_history)

# Function to generate a greeting response for phatic communication
def generate_greeting_response(user_input, chat_history):
    instruction = f"Provide an answer based on the user input. User input: {user_input}\n\n"
    system_prompt = "You are a helpful medical assistant."
    prompt = get_prompt(instruction, system_prompt)
    prompt_template = PromptTemplate(template=prompt, input_variables=["chat_history", "summarized_input"])
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True, memory=memory)
    response = llm_chain.run({"chat_history": chat_history, "user_input": user_input})
    
    chat_history.append((user_input, response))
    return response, chat_history

# Function to generate a default response for unrecognized intents
def generate_default_response(user_input, chat_history):
    response = "Sorry, I couldn't answer your question. Could you please provide more details or ask something else?"
    chat_history.append((user_input, response))
    return response, chat_history
        
def sql_injection(user_input, chat_history):
    response = "Lol 🤣🤣, you are tring an SQL injection on an LLM."
    chat_history.append((user_input, response))
    return response, chat_history


# from typing import List, Tuple
# from langchain.schema import Document

# def handle_similarity_search(user_input, summarized_input, chat_history):
#     # Perform similarity search
#     results = db_csv.similarity_search_with_score(user_input)
    
#     # Define the similarity score threshold
#     threshold = 0.3
    
#     # Filter and sort results based on similarity score
#     filtered_results = [result for result in results if result[1] >= threshold]
#     filtered_results.sort(key=lambda x: x[1], reverse=True)
#     print(filtered_results[0])
#     if filtered_results:
#         best_result = filtered_results[0].page_content
        
#         return generate_response_non_query(user_input, summarized_input, best_result, chat_history)
#     else:
#         return generate_default_response(user_input, chat_history)

def generate_query(user_input, summarized_input, chat_history):
    instruction = f"Generate only a sqlite query without any extra text based on the following user input. Don't query using '=' try to use 'like' The columns in the database were Product, Category, MedicalBenefits, CustomerRating, PurchaseFrequency, description. Name of the table is new_dataset. Generate the query according to user input. user input: {user_input}"
    system_prompt = "You are an expert in SQL queries."
    prompt = get_prompt(instruction, system_prompt)
    prompt_template = PromptTemplate(template=prompt, input_variables=["user_input"])
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    query = llm_chain.run({"user_input": summarized_input}).strip()
    print(query)
    return execute_query(user_input, summarized_input, query, chat_history)

# Function to execute the SQL query
def execute_query(user_input, summarized_input, query, chat_history):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()
        result = str(result)

        return generate_response(user_input, summarized_input, result if result else "No results found.", chat_history)
    except sqlite3.Error as e:
        error_message = f"Error executing SQL query: {e}"
        print(error_message)
        return generate_response(user_input, summarized_input, error_message, chat_history)


# Function to generate a response based on the query result
def generate_response(user_input, summarized_input, query_result, chat_history):
    instruction = f"Provide an answer based on the query result and user input. User input: {summarized_input}\n\nQuery result: {query_result}"
    system_prompt = "You are a professional medical assistant, adept at handling inquiries related to medical products."
    prompt = get_prompt(instruction, system_prompt)
    prompt_template = PromptTemplate(template=prompt, input_variables=["chat_history", "user_input"])
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True, memory=memory)
    response = llm_chain.run({"chat_history": chat_history, "user_input": query_result})
    
    chat_history.append((user_input, response))
    return response, chat_history

# Function to generate a response based on similarity search result
def generate_response_non_query(user_input, summarized_input, result, chat_history):
    instruction = f"Provide an answer based on the similarity search result and user input. User input: {summarized_input}\n\nSimilarity search result: {result}"
    system_prompt = "You are a professional medical assistant, adept at handling inquiries related to medical products."
    prompt = get_prompt(instruction, system_prompt)
    prompt_template = PromptTemplate(template=prompt, input_variables=["chat_history", "summarized_input"])
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True, memory=memory)
    response = llm_chain.run({"chat_history": chat_history, "user_input": summarized_input})
    
    chat_history.append((user_input, response))
    return response, chat_history

# # Gradio interface
# with gr.Blocks(title="Chatbot using OpenAI") as demo:
#     gr.Markdown("# Chatbot using OpenAI")

#     chatbot = gr.Chatbot([], elem_id="chatbot", height=700)
#     msg = gr.Textbox(show_copy_button=True)

#     clear = gr.ClearButton([msg, chatbot])

#     msg.submit(Summarize, [msg, chatbot], [msg, chatbot])

# demo.launch(debug=True, share=True)