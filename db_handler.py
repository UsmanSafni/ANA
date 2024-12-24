import sqlite3
import pandas as pd
from datetime import datetime

class CategoryDB:
    def __init__(self, db_path="categories.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                category TEXT,
                year_month TEXT
            )
            """
        )

    def save_category(self, question, category):
        try:
            # Get the current date in the format YYYY-MM
            current_date = datetime.now().strftime("%Y-%m")
            
            self.conn.execute(
                "INSERT INTO categories (question, category, year_month) VALUES (?, ?, ?)",
                (question, category, current_date)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def read_sql(self):
        query = """
        SELECT category, COUNT(*) AS Count
        FROM categories
        GROUP BY category
        ORDER BY Count DESC;
        """
        self.df_queries = pd.read_sql(query, self.conn)
        return self.df_queries

    def read_qns(self):
        ask = """
        SELECT year_month, COUNT(*) AS Count
        FROM categories
        GROUP BY year_month
        ORDER BY year_month DESC;
        """
        self.no_queries = pd.read_sql(ask, self.conn)
        self.no_queries['year_month']=self.no_queries['year_month'].astype(str)
        return self.no_queries
