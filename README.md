# 🎯 Clinical Trial Disease Classification System

An end-to-end medical natural language processing (NLP) and relational analytics platform designed to clean unstructured study summaries, train predictive text models, and classify clinical trials into distinct therapeutic areas.

## 📖 Project Overview
Clinical trial registries generate vast amounts of free-text study summaries and eligibility criteria. Manually reading summaries to categorize trials is slow and subjective. Medical texts contain complex jargon, and trial datasets are often highly imbalanced.

This project implements a complete NLP pipeline to classify clinical trials into 8 distinct therapeutic areas (e.g., Covid-19, Breast Cancer, Type 2 Diabetes) using machine learning.

## 🏗️ Architecture & Pipeline

1. **Relational Database Design**: Models database schemas using SQLAlchemy to store and query over 60,000 clinical trial records.
2. **Medical Text Preprocessing Pipeline**: A fast, custom NLP pipeline using NLTK to:
   - Lowercase texts and strip HTML tags
   - Remove punctuation and numbers
   - Filter out English stopwords
   - Lemmatize tokens to their dictionary root forms
3. **TF-IDF Feature Engineering**: Constructs a high-dimensional text feature matrix using a TfidfVectorizer configured with unigrams, bigrams, sublinear term frequency scaling, and a vocabulary limit of 15,000 features.
4. **Balanced Classification & Explainability**: Uses Logistic Regression optimized with balanced class weighting to handle imbalanced therapeutic labels. Explains predictions via an Explainable AI (XAI) dashboard showing the top predictive keywords.

## 🚀 How to Run

1. **Verify Database**: Ensure the SQLite database `clinical_trials.db` exists in the project root.
2. **Execute Ingestion & Training**:
   ```bash
   python -m src.pipeline
   python -m src.train
   ```
3. **Run the Streamlit Dashboard**:
   ```bash
   python -m streamlit run app.py
   ```
