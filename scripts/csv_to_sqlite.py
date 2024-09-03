"""
Used to convert the CSV file to a SQLite database table. This script is used in preparation for 
the API to query the SQLite database.
"""

import pandas as pd
from pathlib import Path

from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver


PRODUCT_DESCRIPTIONS_CSV = "product_descriptions.csv"

# Read the CSV file into a DataFrame
# Attempt using UTF-8 encoding first
try:
    df = pd.read_csv(PRODUCT_DESCRIPTIONS_CSV)
except UnicodeDecodeError:
    # If UTF-8 fails, fall back to ISO-8859-1
    df = pd.read_csv(PRODUCT_DESCRIPTIONS_CSV, encoding="ISO-8859-1")

# Create a new column with unique values for the primary key
primary_key_column = "id"
df[primary_key_column] = range(1, 1 + len(df))

# Move the primary key column to the first position
cols = df.columns.tolist()
cols.insert(0, cols.pop(cols.index(primary_key_column)))
df = df[cols]

print(df.info())

# Connect to the SQLite database
driver = SQLiteDriver(SQLiteConnectionConfig(db_file=Path("aegion.db"), dataset_table_name="new_dataset"))
driver.connect()

conn = driver._db_connection

if conn is None:
    raise ValueError("Database connection not established.")

# Create the table with a primary key explicitly
table_name = driver.db_config.dataset_table_name
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    {primary_key_column} INTEGER PRIMARY KEY,
    {', '.join([f'{col} TEXT' for col in df.columns if col != primary_key_column])}
)
"""

# Execute the CREATE TABLE statement
with conn:
    conn.execute(create_table_sql)

# Insert the DataFrame into the SQLite table
df.to_sql(table_name, conn, if_exists="append", index=False)

print(f"Data written to {table_name} in {driver.db_config.db_file}")

# Commit the changes
conn.commit()
conn.close()
