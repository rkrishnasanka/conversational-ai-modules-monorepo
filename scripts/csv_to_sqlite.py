"""
Used to convert the CSV file to a SQLite database table. This script is used in preparation for 
the API to query the SQLite database.
"""

import sqlite3

import pandas as pd
from pathlib import Path

from nlqs.database.sqlite import SQLiteConnectionConfig, SQLiteDriver



PRODUCT_DESCRIPTIONS_CSV ="./product_descriptions.csv"

# def convert_csv_to_sqlite():

# Read the CSV file into a DataFrame
# Attempt using UTF-8 encoding first
try:
    df = pd.read_csv(PRODUCT_DESCRIPTIONS_CSV)
except UnicodeDecodeError:
    # If UTF-8 fails, fall back to ISO-8859-1
    df = pd.read_csv(PRODUCT_DESCRIPTIONS_CSV, encoding="ISO-8859-1")

# Connect to the SQLite database
# conn = sqlite3.connect(connection_config.db_file)

driver = SQLiteDriver(SQLiteConnectionConfig(
    db_file=Path("aegion.db"),
    dataset_table_name="new_dataset"
))

driver.connect()

conn = driver._db_connection


# Write the DataFrame to a SQLite table
df.to_sql(driver.db_config.dataset_table_name, conn, if_exists="replace", index=False)

print(f"Data written to {driver.db_config.dataset_table_name} in {driver.db_config.db_file}")

# Commit the changes
conn.commit()
conn.close()

