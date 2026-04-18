import pandas as pd
import sqlite3
import os

# 1. Connect to your database
db_path = 'db/db.sqlite'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 2. Get a list of all tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Create a folder for your CSVs if it doesn't exist
os.makedirs('csv_exports', exist_ok=True)

# 3. Export each table
for table_name in tables:
    name = table_name[0]
    print(f"Exporting {name}...")
    
    # Read the table into a Pandas DataFrame
    df = pd.read_sql(f'SELECT * FROM "{name}"', conn)
    
    # Save to CSV (index=False prevents an extra "unnamed" column)
    df.to_csv(f'csv_exports/{name}.csv', index=False)

print("✅ All tables exported to the /csv_exports folder!")
conn.close()