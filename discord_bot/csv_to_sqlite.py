import sqlite3
import pandas as pd

# Define the CSV file path and the SQLite database name
csv_file_path = 'product_descriptions.csv'
sqlite_db_name = 'aegion.db'  # It's good practice to include the .db extension

try:
    # Read the CSV file into a DataFrame
    # Attempt using UTF-8 encoding first
    try:
        df = pd.read_csv(csv_file_path)
    except UnicodeDecodeError:
        # If UTF-8 fails, fall back to ISO-8859-1
        df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')
    
    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_db_name)

    # Write the DataFrame to a SQLite table
    df.to_sql('new_dataset', conn, if_exists='replace', index=False)

    # Commit the changes
    conn.commit()

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the connection
    conn.close()