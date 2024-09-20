"""
Used to convert the CSV file to a PostgreSQL database table. This script is used in preparation for 
the API to query the PostgreSQL database.
"""

import pandas as pd
import psycopg2
from psycopg2 import sql

# Define connection parameters
conn_params = {"dbname": "Aegion", "user": "postgres", "password": "1234", "host": "localhost", "port": "5432"}

# CSV file path
csv_file_path = "product_descriptions.csv"
table_name = "new_dataset"

primary_key_column = "id"  # Primary key column name


# Establish a connection to PostgreSQL
def connect_to_db(params):
    conn = psycopg2.connect(**params)
    return conn


# Create table based on CSV columns
def create_table_from_csv(conn, df, table_name, primary_key_column):
    # Drop table if it exists
    drop_table_query = sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name))

    # Execute drop query
    with conn.cursor() as cursor:
        cursor.execute(drop_table_query)
        conn.commit()

    # Extract column names and types from DataFrame
    columns = df.columns
    column_types = ["TEXT" for _ in columns]  # Default data type is TEXT

    # Modify the primary key column to be SERIAL
    column_types[0] = "SERIAL PRIMARY KEY"

    # Create SQL query for table creation
    create_table_query = sql.SQL("CREATE TABLE {} ({})").format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(
            sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(dtype)) for col, dtype in zip(columns, column_types)
        ),
    )

    # Execute the query
    with conn.cursor() as cursor:
        cursor.execute(create_table_query)
        conn.commit()


# Insert data from DataFrame into the table
def insert_data_from_csv(conn, df, table_name):
    # Generate column placeholders
    columns = df.columns
    num_columns = len(columns)
    placeholders = ", ".join(["%s"] * num_columns)

    # Insert data into the table
    insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table_name), sql.SQL(", ").join(map(sql.Identifier, columns)), sql.SQL(placeholders)
    )

    # Convert DataFrame rows into a list of tuples
    data = [tuple(row) for row in df.values]

    # Execute batch insert
    with conn.cursor() as cursor:
        cursor.executemany(insert_query, data)
        conn.commit()


# Main function
def main():
    # Load CSV into pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Create a new primary key column with unique values
    df[primary_key_column] = range(1, 1 + len(df))

    # Move the primary key column to the first position
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index(primary_key_column)))
    df = df[cols]

    # Connect to PostgreSQL
    conn = connect_to_db(conn_params)

    try:
        # Create (or recreate) table and insert data
        create_table_from_csv(conn, df, table_name, primary_key_column)
        insert_data_from_csv(conn, df, table_name)

        print(f"Data successfully written to {table_name}!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
