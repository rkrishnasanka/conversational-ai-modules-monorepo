# NLQS: Natural Language Query System

## Overview

The Natural Language Query System (NLQS) is designed to process user input and query a database using natural language. It utilizes a combination of summarization, quantitative search, and qualitative semantic search to retrieve relevant data based on user requests. The system is equipped to handle user intents, manage potential SQL injections, and ensure that only meaningful queries are processed.

## Algorithm Workflow

### 1. Summarization of User Input
- **Input**: User input and chat history.
- **Process**: The input is passed through a summarization function, which outputs a JSON object with five keys:
  - **summary**: A concise summary of the user input.
  - **quantitative_data**: A dictionary containing numerical column names as keys and the related data mentioned by the user.
  - **qualitative_data**: A dictionary similar to `quantitative_data`, but for categorical columns.
  - **user_requested_columns**: A list of columns from which the user wants data.
  - **user_intent**: The user's intent, categorized as one of the following: `phatic_communication`, `sql_injection`, `profanity`, or `other`.

### 2. Intent Handling
- **SQL Injection Check**: If `user_intent` is `sql_injection`, the process is terminated immediately to prevent any harmful actions.
- **Column Request Check**: If `user_requested_columns` is empty, the process is terminated, as there is no data to retrieve.

### 3. Primary Key Collection for Data Retrieval
- **Quantitative Data Retrieval**:
  - A quantitative search query is generated to find all primary keys for the quantitative columns mentioned by the user.
  - **Example**: `SELECT {primaryKey} FROM {tableName} WHERE columnData > 10`
- **Qualitative Data Retrieval**:
  - A semantic search is performed on the categorical data using a pre-built Chroma collection.
  - **Chroma Collection Setup**:
    - The collection converts all categorical data into embeddings.
    - Each embedding includes:
      - **id**: Format `columnName_primaryKey`.
      - **document**: The data from the specific column and row.
      - **metadata**: A dictionary containing details like `primaryKey`, `column_name`, and `table_name`.
  - **Similarity Search**:
    - A similarity search is conducted using the Chroma collection to find the most relevant primary keys.
    - **Example**: `collection.query("data for the particular column mentioned in the user input", n_results=5, where={"column_name": column})`
    - Only the top 5 results for each column are considered.

### 4. Data Intersection and Extraction
- **Intersection of Primary Keys**:
  - Both qualitative and quantitative primary keys are compared to find common entries.
- **SQL Query Generation**:
  - If there is an intersection, an SQL query is generated to extract the relevant data.
  - **Example**: `SELECT * FROM {tableName} WHERE {primaryKey} IN [primary keys]`
- **Fallback Mechanism**:
  - If no intersection is found, a union of both qualitative and quantitative primary keys is created.
  - The system informs the user that the data found is not an exact match but is closely related.

### 5. Data Retrieval and User Notification
- The system retrieves the data based on the SQL query and provides feedback to the user:
  - If an intersection is found: "This is the exact data you were searching for."
  - If no intersection is found: "This isn't the exact data you were searching for, but we have this related information."


![work flow](NLQS_v0.6.svg) 