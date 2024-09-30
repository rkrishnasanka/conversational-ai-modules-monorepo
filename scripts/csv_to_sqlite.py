"""
Used to convert the CSV file to a SQLite database table. This script is used in preparation for 
the API to query the SQLite database.
"""

import sqlite3
from pathlib import Path

import pandas as pd

# Constants for file paths and table names
CSV_FILE = "product_descriptions.csv"
DB_FILE = "aegion.db"
TABLE_NAME = "new_dataset"
PRIMARY_KEY = "pri_key"

# Read the CSV file into a DataFrame
# Attempt using UTF-8 encoding first
try:
    df = pd.read_csv(CSV_FILE)
except UnicodeDecodeError:
    # If UTF-8 fails, fall back to ISO-8859-1
    df = pd.read_csv(CSV_FILE, encoding="ISO-8859-1")

# Create a new column with unique values for the primary key
df[PRIMARY_KEY] = range(1, 1 + len(df))

# Move the primary key column to the first position
columns = [PRIMARY_KEY] + [col for col in df.columns if col != PRIMARY_KEY]
df = df[columns]

# Connect to the SQLite database (create if it doesn't exist)
with sqlite3.connect(DB_FILE) as conn:
    cursor = conn.cursor()

    # Create the table
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        {PRIMARY_KEY} INTEGER PRIMARY KEY,
        {', '.join([f'{col} TEXT' for col in df.columns if col != PRIMARY_KEY])}
    )
    """
    cursor.execute(create_table_sql)

    # Insert data from the DataFrame into the SQLite table
    insert_sql = f"""
    INSERT INTO {TABLE_NAME} ({', '.join(columns)})
    VALUES ({', '.join(['?' for _ in columns])})
    """

    # Use executemany for better performance
    cursor.executemany(insert_sql, df.values.tolist())

# Confirmation message
print(f"Data from '{CSV_FILE}' imported to table '{TABLE_NAME}' in '{DB_FILE}' successfully!")
