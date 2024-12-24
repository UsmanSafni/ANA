import sqlite3
import pandas as pd

# Connect to your existing database
conn = sqlite3.connect("categories.db", check_same_thread=False)

# Dummy data with different year_month values
dummy_data = [
    ("What are the top exercise for elderly people?", "Exercise", "2024-01"),
    ("Drugs for fever", "Drugs", "2023-02"),
    ("Muscle strengthening exercises", "Exercise", "2023-03"),
    ("How to deal with sleep disorders", "Sleep", "2023-04"),
    ("How to be mentally strong","Mental Health", "2024-05"),
    ("What are the symptoms of flu?", "General health", "2023-06"),
    ("How does antibiotic drug work", "Drugs", "2023-07"),
    ("Diet plans for 20 year olds", "Diet", "2024-08"),
    ("How to nurture a healthy mind", "Mental Health", "2023-09"),
    ("What causes sleep disorders", "Sleep", "2023-10"),
    ("What are the top causes of mal nutrition", "Nutrition", "2023-10"),
]

# Insert dummy data into the database
try:
    with conn:
        conn.executemany(
            "INSERT INTO categories (question, category, year_month) VALUES (?, ?, ?)", 
            dummy_data
        )
    print("Dummy data inserted successfully!")
except sqlite3.Error as e:
    print(f"Database error: {e}")

# Verify the data
try:
    df = pd.read_sql("SELECT * FROM categories", conn)
    print("\nData in the categories table:")
    print(df)
except Exception as e:
    print(f"Error reading database: {e}")

# Close the connection
conn.close()
