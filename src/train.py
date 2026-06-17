import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.database import SessionLocal, ClinicalTrial

def train_model():
    """Trains the clinical trial disease classification model and saves metrics."""
    print("Connecting to database and fetching clinical trials data...")
    db_session = SessionLocal()
    
    try:
        # Query preprocessed data
        trials = db_session.query(ClinicalTrial.cleaned_summary, ClinicalTrial.disease_category).all()
        
        if not trials:
            print("❌ No data found in the database. Please run the ETL pipeline first:")
            print("   python -m src.pipeline")
            return False
            
        print(f"Loaded {len(trials)} trials from the database.")
        
        # Load into DataFrame
        df = pd.DataFrame(trials, columns=['cleaned_summary', 'disease_category'])
        
        # Drop rows where cleaned_summary is empty
        df = df.dropna(subset=['cleaned_summary'])
        df = df[df['cleaned_summary'].str.strip() != ""]
        
        if len(df) < 100:
            print(f"⚠️ Warning: Dataset size too small for training ({len(df)} rows).")
            print("Ingesting raw data first or continuing...")
            
        print(f"Preprocessed records available for training: {len(df)}")
        
        X = df['cleaned_summary'].values
        y = df['disease_category'].values
        
        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)} | Test set size: {len(X_test)}")
        
        tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
        
        # Define models to compare
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            "Multinomial Naive Bayes": MultinomialNB(),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        }
        
        best_model_name = None
        best_accuracy = 0
        best_pipeline = None
        
        print("Starting model training and comparison...")
        for name, clf in models.items():
            print(f"Training {name}...")
            pipeline = Pipeline([('tfidf', tfidf), ('clf', clf)])
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"{name} Accuracy: {acc:.4f}")
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                best_pipeline = pipeline

        print(f"\nBest Model Found: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        # Re-evaluate with best estimator
        print("Evaluating best model performance on test set...")
        y_pred = best_pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_str = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print("\nClassification Report:\n", report_str)
        
        # Ensure models directory exists
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Extract features and model coefficients for Explainable AI tab in Streamlit
        print("Extracting top predictive words for each disease category...")
        vectorizer = best_pipeline.named_steps['tfidf']
        classifier = best_pipeline.named_steps['clf']
        feature_names = vectorizer.get_feature_names_out()
        classes = classifier.classes_
        
        top_words_per_class = {}
        for idx, class_name in enumerate(classes):
            top_words = []
            if hasattr(classifier, 'coef_'):
                if len(classes) == 2:
                    coefs = classifier.coef_[0] if idx == 1 else -classifier.coef_[0]
                else:
                    coefs = classifier.coef_[idx]
                top_coef_indices = np.argsort(coefs)[-20:][::-1]
                top_words = [(feature_names[i], float(coefs[i])) for i in top_coef_indices]
            elif hasattr(classifier, 'feature_log_prob_'):
                coefs = classifier.feature_log_prob_[idx]
                top_coef_indices = np.argsort(coefs)[-20:][::-1]
                top_words = [(feature_names[i], float(coefs[i])) for i in top_coef_indices]
            elif hasattr(classifier, 'feature_importances_'):
                # Random Forest has general feature importances, not per-class
                coefs = classifier.feature_importances_
                top_coef_indices = np.argsort(coefs)[-20:][::-1]
                top_words = [(feature_names[i], float(coefs[i])) for i in top_coef_indices]
                
            top_words_per_class[class_name] = top_words
            
        # Structure the training metadata metrics dictionary
        metrics = {
            'accuracy': float(accuracy),
            'classification_report': report_dict,
            'confusion_matrix': cm.tolist(),
            'classes': classes.tolist(),
            'n_train_samples': int(len(X_train)),
            'n_test_samples': int(len(X_test)),
            'top_predictive_words': top_words_per_class
        }
        
        # Save model pipeline and metrics
        model_path = os.path.join(models_dir, "classifier_pipeline.joblib")
        metrics_path = os.path.join(models_dir, "training_metrics.joblib")
        
        print(f"Saving best model pipeline to: {model_path}")
        joblib.dump(best_pipeline, model_path)
        
        print(f"Saving training metrics to: {metrics_path}")
        joblib.dump(metrics, metrics_path)
        
        print("SUCCESS: Model pipeline and metrics saved successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Model training failed: {e}")
        return False
    finally:
        db_session.close()

if __name__ == "__main__":
    train_model()
