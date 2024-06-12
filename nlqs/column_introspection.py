# Move all the code for generating descriptiosn of the column in here don't accept data frames but rather simple lists of info

import sqlite3
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from discord_bot.parameters import OPENAI_API_KEY, SQLITE_DB_FILE, SQL_TABLE_NAME

# Fetch data from SQLite
def fetch_data_from_sqlite(db_file, table_name):
    try:
        conn = sqlite3.connect(db_file)
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
    except sqlite3.Error as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    finally:
        conn.close()
    return df

def generate_sample_data(dataframe):
    sample_data_list = []
    for column in dataframe.columns:
        col_data = dataframe[column]
        col_type = col_data.dtype
        sample_data = col_data.dropna().sample(min(5, len(col_data))).tolist()
        sample_data_str = ', '.join(map(str, sample_data))
        sample_data_list.append(f"{column}: {sample_data_str}, {col_type}")
    return sample_data_list

def get_column_descriptions(sample_data, input_text) -> dict:
    """Get column descriptions from OpenAI API."""
    # Initialize an empty dictionary to store column descriptions
    descriptions = {}

    for item in sample_data:
        column_info, col_type = item.rsplit(", ", 1)
        column, sample_data = column_info.split(": ", 1)
        print(column, col_type, sample_data)

        # Prepare the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"""
                The following is a description of a dataset column:
                Column Name: {column}
                Data Type: {col_type}
                
                Please provide a detailed description of this column, including its potential meaning, use, and importance in a dataset. Use sample data to identify the column's meaning.
                
                Sample Data: {sample_data}
                
                Use the following format:

                For example:
                   "Product": "This column contains the name of the product. It is a text field and can be used for exact or partial matches.",
                   "Category": "This column contains the category of the product. It is a text field and can be used for exact or partial matches.",
                    "MedicalBenefits": "This column contains the medical benefits of the product. It is a text field and can be used for exact or partial matches.",
                    "CustomerRating": "This column contains the customer rating of the product. It is a numerical field and can be used for exact matches or range comparisons.",
                    "PurchaseFrequency": "This column contains the frequency of product purchase. It is a text field and can be used for exact or partial matches.",
                    "description": "This column contains the description of the product. It is a text field and can be used for exact or partial matches."
                """),
                ("user", "{user_input}")
            ]
        )

        llm = ChatOpenAI(
            model="gpt-4",
            api_key=OPENAI_API_KEY,
            temperature=0.0,
            verbose=True,
        )

        chain = LLMChain(prompt=prompt, llm=llm)
        
        # Generate the description
        response = chain.run(input_text)
        
        # Extract the description from the response
        description = response.strip()
        
        # Add the description to the dictionary
        descriptions[column] = description
        
    # Return the dictionary of column descriptions
    return descriptions

def store_descriptions_in_db(descriptions, numerical_columns, categorical_columns, db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # Create table for column descriptions
    c.execute('''
        CREATE TABLE IF NOT EXISTS column_descriptions (
            column_name TEXT PRIMARY KEY,
            description TEXT
        )
    ''')

    for column, description in descriptions.items():
        c.execute('''
            INSERT OR REPLACE INTO column_descriptions (column_name, description)
            VALUES (?, ?)
        ''', (column, description))

    # Create table for numerical and categorical columns
    c.execute('''
        CREATE TABLE IF NOT EXISTS column_types (
            column_name TEXT PRIMARY KEY,
            column_type TEXT
        )
    ''')

    for column in numerical_columns:
        c.execute('''
            INSERT OR REPLACE INTO column_types (column_name, column_type)
            VALUES (?, ?)
        ''', (column, 'numerical'))

    for column in categorical_columns:
        c.execute('''
            INSERT OR REPLACE INTO column_types (column_name, column_type)
            VALUES (?, ?)
        ''', (column, 'categorical'))

    conn.commit()
    conn.close()

df = fetch_data_from_sqlite(SQLITE_DB_FILE,SQL_TABLE_NAME)

# Generate sample data for each column
sample_data_list = generate_sample_data(df)

# Get column descriptions from OpenAI API
column_descriptions = get_column_descriptions(sample_data_list, input_text="Please provide a detailed description of each column in the given dataset.")

# Identify numerical and categorical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Store descriptions and column types in the database
store_descriptions_in_db(column_descriptions, numerical_columns, categorical_columns, SQLITE_DB_FILE)

print("Column descriptions and column types stored in the database.")