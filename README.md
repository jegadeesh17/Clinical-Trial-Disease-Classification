# Clinical Trial Disease Category Classification System

---

### **Project Overview**

Clinical trial registries generate vast amounts of free-text study summaries and eligibility criteria. Manually reading and categorizing these trials into therapeutic areas is slow, subjective, and prone to inconsistency. Medical texts contain complex jargon, and trial datasets are often highly imbalanced across disease categories.

This project implements a complete end-to-end NLP and relational analytics platform to clean unstructured study summaries, train a TF-IDF + Logistic Regression classification model, and classify clinical trials into 8 distinct therapeutic areas — served through an interactive Streamlit dashboard with real-time Explainable AI visualizations.

---

### **Key Features**

* **Large-Scale ETL Pipeline:** Ingests 60,000+ clinical trial records from raw CSV into a central SQLite database via SQLAlchemy.
* **Medical Text NLP Pipeline:** Custom NLTK-based tokenization, stopword removal, and lemmatization tailored for medical language.
* **TF-IDF Feature Extraction:** Unigram/bigram TF-IDF with sublinear scaling and 15,000 max features for rich text representation.
* **Class-Weighted Classification:** Logistic Regression with class-weight balancing to handle severe label imbalance.
* **8-Class Disease Taxonomy:** Classifies trials into Covid-19, Breast Cancer, Type 2 Diabetes, and 5 other therapeutic areas.
* **Explainable AI (XAI):** Real-time keyword visualization explaining which terms drove each classification prediction.
* **Interactive Streamlit Dashboard:** Clean UI for text input classification with instant results and prediction explanations.
* **Modular Pipeline Architecture:** Separate ETL, NLP preprocessing, training, and inference scripts for clean separation.

---

### **Dataset**

* **Source:** Raw clinical trials dataset (`clinical_trials_raw_patient2trial_conditions.csv`)
* **Download:** [Google Drive](https://drive.google.com/file/d/1NJ1JzOR2SLQhFWsPY5L8AMDXFLrQmrK2/view?usp=sharing)
* **Coverage:** 60,000+ clinical trial records with study summaries and disease labels
* **Format:** CSV with free-text study descriptions and categorical disease labels

#### **Disease Categories (8 Classes)**

* Covid-19
* Breast Cancer
* Type 2 Diabetes
* Cardiovascular Disease
* Mental Health / Depression
* Alzheimer's / Dementia
* Cancer (General)
* Rare Diseases

#### **Key Fields**

* Study title and summary text
* Eligibility criteria description
* Disease condition label
* Trial phase and study type

---

### **Project Structure**

```bash
ClinicalTrialClassification/
│
├── app/                          # Streamlit application files
│   └── app.py                    # Main Streamlit dashboard
├── data/                         # Project datasets
├── docs/                         # Documentation and visualizations
├── models/                       # Saved trained models
├── notebooks/                    # Jupyter notebooks (Source of Truth)
├── src/                          # Core Python logic and scripts
├── requirements.txt              # Python dependencies
└── README.md
```

---

### **How It Works**

### **1. Data Ingestion & Database Setup**

* Reads raw CSV with 60,000+ clinical trial records
* Ingests into a central SQLite database using SQLAlchemy
* Establishes indexed schema for fast query access

```python
from sqlalchemy import create_engine

engine = create_engine("sqlite:///clinical_trials.db")
df.to_sql("trials", engine, if_exists="replace", index=False)
```

---

### **2. Medical Text Preprocessing**

Custom NLTK NLP pipeline applied to study summaries:

| Step            | Operation                                         |
| --------------- | ------------------------------------------------- |
| Lowercasing     | Normalizes text casing                            |
| Tokenization    | Splits text into word tokens                      |
| Stopword Removal| Removes non-informative common words              |
| Lemmatization   | Reduces words to root form (e.g., "studying" → "study") |
| Special Char Removal | Strips punctuation and numeric tokens        |

---

### **3. TF-IDF Feature Engineering & Classification**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=15000
)

X = vectorizer.fit_transform(df['cleaned_summary'])

clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(X_train, y_train)
```

---

### **Model Performance**

| Metric              | Score              |
| ------------------- | ------------------ |
| Classification Task | Multi-class (8)    |
| Feature Type        | TF-IDF (Bi-gram)   |
| Class Handling      | Balanced weighting |
| Model               | Logistic Regression|

---

### **Interactive Application Deployment**

The project features an interactive **Streamlit Web Application** that accepts free-text clinical trial summaries and returns real-time disease category predictions with XAI keyword explanations.

#### **To Launch the Platform Locally:**
```powershell
streamlit run app/app.py
```

---

### **Technology Stack**

| Category             | Tools                          |
| -------------------- | ------------------------------ |
| Programming          | Python                         |
| NLP                  | NLTK, TF-IDF (Scikit-learn)    |
| Machine Learning     | Scikit-learn (Logistic Regression) |
| Database             | SQLite, SQLAlchemy             |
| Data Processing      | Pandas, NumPy                  |
| Explainability       | Custom XAI keyword visualization|
| Notebook Environment | Jupyter Notebook               |
| Web Framework        | Streamlit                      |

---

### **Getting Started**

### **1. Clone Repository**

```bash
git clone https://github.com/jegadeesh17/Clinical-Trial-Disease-Classification.git

cd ClinicalTrialClassification
```

---

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### **3. Download Dataset**

Download the raw dataset from [Google Drive](https://drive.google.com/file/d/1NJ1JzOR2SLQhFWsPY5L8AMDXFLrQmrK2/view?usp=sharing) and place it inside the `data/` directory as:

```
data/clinical_trials_raw_patient2trial_conditions.csv
```

---

### **4. Run ETL Pipeline & Train Model**

```bash
python -m src.pipeline
python -m src.train
```

---

### **5. Launch Dashboard**

```bash
python -m streamlit run app.py
```

---

### **Example Use Case**

A pharmaceutical research organization or clinical registry can use this platform to:

1. Automatically classify incoming trial submissions into therapeutic areas
2. Route trials to the appropriate domain review team
3. Analyze the distribution of trials across disease categories
4. Understand which medical terms are most predictive of each disease class

---

### **Future Improvements**

* Fine-tuned BioBERT or ClinicalBERT transformer model for higher NLP accuracy
* Multi-label classification for trials spanning multiple conditions
* Integration with ClinicalTrials.gov API for live trial ingestion
* Confidence score thresholding with human-in-the-loop review queue

---

### **Contributors**

* **Jegadeesh D** — Medical NLP pipeline, TF-IDF feature engineering, Logistic Regression classification, SQLite ETL, explainable AI, and Streamlit dashboard development

---

### **License**

MIT License
