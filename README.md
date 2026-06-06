# 🎯 Clinical Trial Disease Classification System

An end-to-end medical natural language processing (NLP) and relational analytics platform designed to clean unstructured study summaries, train predictive text models, and classify clinical trials into distinct therapeutic areas.

## 📖 Project Overview
Clinical trial registries generate vast amounts of free-text study summaries and eligibility criteria. Manually reading summaries to categorize trials is slow and subjective. Medical texts contain complex jargon, and trial datasets are often highly imbalanced.

This project implements a complete NLP pipeline to classify clinical trials into 8 distinct therapeutic areas (e.g., Covid-19, Breast Cancer, Type 2 Diabetes) using machine learning.

## 🏗️ Architecture & Pipeline

1. **Data Ingestion & Database Setup**: An ETL pipeline to ingest raw CSV data into a central SQLite database managed with SQLAlchemy, storing over 60,000 clinical trial records.
2. **Medical Text Preprocessing**: A custom NLTK NLP pipeline to clean, tokenize, remove stopwords, and lemmatize raw medical text.
3. **Machine Learning & Feature Engineering**: A classification pipeline extracting TF-IDF features (unigrams/bigrams, sublinear scaling, 15,000 max features) and training a class-weighted Logistic Regression model to handle label imbalance.
4. **Interactive Dashboard**: A Streamlit web application providing a clean UI to classify text inputs and explain predictions in real-time using Explainable AI (XAI) keyword visualizations.

## 🚀 How to Run

1. **Download Raw Dataset**: Download the raw dataset `clinical_trials_raw_patient2trial_conditions.csv` from [Google Drive](https://drive.google.com/file/d/1NJ1JzOR2SLQhFWsPY5L8AMDXFLrQmrK2/view?usp=sharing) and place it inside the `data/` directory.
2. **Execute Ingestion & Training**:
   Run the ETL pipeline to ingest the data and train the classifier:
   ```bash
   python -m src.pipeline
   python -m src.train
   ```
3. **Run the Streamlit Dashboard**:
   Start the local Streamlit server to open the interactive classification dashboard:
   ```bash
   python -m streamlit run app.py
   ```
