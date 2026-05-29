import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# Default database is SQLite. If PostgreSQL is needed, set the DATABASE_URL environment variable.
# Example: DATABASE_URL=postgresql://username:password@localhost:5432/clinical_trials_db
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///clinical_trials.db")

engine = create_engine(
    DATABASE_URL, 
    # SQLite-specific arguments (ignored by other databases)
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ClinicalTrial(Base):
    """
    SQLAlchemy Model representing the raw and preprocessed clinical trial data.
    Maps directly to columns in the provided raw dataset CSV.
    """
    __tablename__ = "clinical_trials"

    id = Column(Integer, primary_key=True, index=True)
    nct_id = Column(String(50), unique=True, index=True, nullable=False)
    title = Column(String(500), nullable=True)
    official_title = Column(Text, nullable=True)
    brief_summary = Column(Text, nullable=False)
    cleaned_summary = Column(Text, nullable=True)  # Text after cleaning, tokenizing, lemmatizing
    conditions = Column(Text, nullable=True)
    interventions = Column(Text, nullable=True)
    overall_status = Column(String(100), nullable=True)
    study_type = Column(String(100), nullable=True)
    phase = Column(String(50), nullable=True)
    sex = Column(String(50), nullable=True)
    minimum_age = Column(String(50), nullable=True)
    maximum_age = Column(String(50), nullable=True)
    healthy_volunteers = Column(String(50), nullable=True)
    eligibility_criteria = Column(Text, nullable=True)
    clinicaltrials_url = Column(String(500), nullable=True)
    disease_category = Column(String(100), index=True, nullable=False)  # source_condition_query label

    def __repr__(self):
        return f"<ClinicalTrial {self.nct_id} - {self.disease_category}>"


class PredictionLog(Base):
    """
    SQLAlchemy Model for logging predictions made in the Streamlit interface.
    Enables user feedback loop and performance monitoring over time.
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=True)
    brief_summary = Column(Text, nullable=False)
    eligibility_criteria = Column(Text, nullable=True)
    predicted_category = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    actual_category = Column(String(100), nullable=True)  # User's corrected class (if any)
    is_correct = Column(String(10), nullable=True)        # User verified: "Yes" or "No"
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PredictionLog ID {self.id}: {self.predicted_category} (Conf: {self.confidence:.2f})>"


def init_db():
    """Initializes the database tables."""
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    print(f"Initializing database using connection: {DATABASE_URL}")
    init_db()
    print("Database tables created successfully!")
