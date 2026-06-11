import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sqlalchemy import text
from src.database import engine, SessionLocal, ClinicalTrial, init_db

# Programmatic NLTK download helper to prevent missing resource errors
def setup_nltk():
    """Download required NLTK resources in a fail-safe manner."""
    resources = {
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
        'tokenizers/punkt': 'punkt'
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {name}...")
            nltk.download(name, quiet=True)

# Run NLTK setup on import
setup_nltk()

# Initialize NLP components
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()
lemmatizer = WordNetLemmatizer()

# Clean text function with optimizations for large datasets
def clean_text_fast(text_data):
    """
    Cleans and preprocesses medical text:
    - Convers to lowercase
    - Strips special characters, punctuation, and numbers
    - Filters out english stopwords
    - Lemmatizes remaining words to their base root form
    """
    if not isinstance(text_data, str) or not text_data.strip():
        return ""
    
    # Lowercase
    text_data = text_data.lower()
    
    # Strip HTML-like tags if present
    text_data = re.sub(r'<[^>]*>', ' ', text_data)
    
    # Remove numbers and punctuation, keep only words
    text_data = re.sub(r'[^a-z\s]', ' ', text_data)
    
    # Tokenize by splitting on whitespace
    words = text_data.split()
    
    # Stopword removal and lemmatization
    cleaned_words = [
        lemmatizer.lemmatize(word) 
        for word in words 
        if word not in stop_words and len(word) > 2
    ]
    
    return " ".join(cleaned_words)


def run_etl(csv_path=None, force=False):
    """
    Extracts data from the CSV, transforms it by preprocessing the text, 
    and loads it in bulk into the SQLite database.
    """
    # Create tables if they do not exist
    init_db()
    
    db_session = SessionLocal()
    try:
        # Check if database is already populated
        trial_count = db_session.query(ClinicalTrial).count()
        if trial_count > 0 and not force:
            print(f"Database already contains {trial_count} clinical trials. Skipping ingestion.")
            return True
        
        # If path is not provided, look in default location
        if csv_path is None:
            csv_path = r"c:\Users\jegad\projects\Clinical Trial Disease Classification\data\clinical_trials_raw_patient2trial_conditions.csv"
            
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Source CSV file not found at: {csv_path}")
            
        print(f"Reading CSV dataset from: {csv_path}")
        # Read the raw CSV file
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV with {len(df)} rows.")
        
        # Clean missing values in critical columns
        df['brief_summary'] = df['brief_summary'].fillna("")
        df['title'] = df['title'].fillna(df['nct_id'])
        df['source_condition_query'] = df['source_condition_query'].fillna("other")
        
        # Run text preprocessing on the brief_summary
        print("Preprocessing clinical trial brief summaries. This might take up to a minute...")
        df['cleaned_summary'] = df['brief_summary'].apply(clean_text_fast)
        print("Preprocessing completed!")
        
        # Rename columns to match SQLAlchemy model attributes
        df = df.rename(columns={'source_condition_query': 'disease_category'})
        
        # Select and order columns matching database schema
        db_columns = [
            'nct_id', 'title', 'official_title', 'brief_summary', 'cleaned_summary', 
            'conditions', 'interventions', 'overall_status', 'study_type', 'phase', 
            'sex', 'minimum_age', 'maximum_age', 'healthy_volunteers', 
            'eligibility_criteria', 'clinicaltrials_url', 'disease_category'
        ]
        
        # Filter dataframe for only DB columns
        df_db = df[db_columns]
        
        # Clear existing entries for fresh run
        if trial_count > 0 and force:
            print("Clearing existing clinical trials from database...")
            db_session.execute(text("DELETE FROM clinical_trials"))
            db_session.commit()
            
        print("Bulk uploading dataset into PostgreSQL database. Writing rows...")
        # Write directly to SQL using pandas optimized engine interface
        df_db.to_sql(
            name='clinical_trials', 
            con=engine, 
            if_exists='append', 
            index=False,
            chunksize=5000
        )
        
        db_session.commit()
        final_count = db_session.query(ClinicalTrial).count()
        print(f"ETL completed! Added {final_count} records to the database.")
        return True
        
    except Exception as e:
        db_session.rollback()
        print(f"ETL Pipeline Error: {e}")
        return False
    finally:
        db_session.close()

if __name__ == "__main__":
    run_etl(force=True)
