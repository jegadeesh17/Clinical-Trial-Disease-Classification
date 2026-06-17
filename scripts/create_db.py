import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "clinical_trials_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres@1706")

try:
    # Connect to an existing database to create the new database
    conn = psycopg2.connect(
        dbname="marketing_campaign",
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DB_NAME}'")
    exists = cursor.fetchone()
    
    if not exists:
        print(f"Creating database {DB_NAME}...")
        cursor.execute(f"CREATE DATABASE {DB_NAME}")
    else:
        print(f"Database {DB_NAME} already exists.")
        
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"Error creating database: {e}")
