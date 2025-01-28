import re
import logging
import sqlite3
import pandas as pd


# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    filename="sentiment.log",
    filemode="a",
    format="%(asctime)s - %(lineno)d - %(name)s - %(message)s"
)


# Clean Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text



# Load data and setup database
def setup_database(csv_file_path, db_file):
    logging.info("Starting database setup")
    
    df = pd.read_csv(csv_file_path)
    
    # Remove duplicates and null values
    df = df.drop_duplicates(subset="review", keep="first").dropna()
    
    # Clean review_text
    df["review"] = df["review"].apply(clean_text)
    
    # Connect to SQLite
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Define table schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS imdb_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT,
            sentiment TEXT
        )
    ''')
    logging.info("Table created successfully.")
    
    # Insert data
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO imdb_reviews (review_text, sentiment) VALUES (?, ?)""", (row['review'], row['sentiment'])
        )
    
    conn.commit()
    conn.close()
    logging.info("Data Ingestion completed.")


# Run setup_database function
def run_setup_database():
    setup_database(csv_file_path="IMDB Dataset.csv", db_file="imdb_reviews.db")