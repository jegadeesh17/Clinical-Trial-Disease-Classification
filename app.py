import os
import sys
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gr
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database import SessionLocal, ClinicalTrial, PredictionLog, engine
from src.pipeline import run_etl, clean_text_fast
from src.train import train_model

# --- PAGE CONFIGURATION & PREMIUM THEMING ---
st.set_page_config(
    page_title="Clinical Trial Intelligence Hub",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Injected CSS for Glassmorphic Styling and Dark Mode Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Typography Overrides */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom Background & Styling */
    .stApp {
        background-color: #0f172a; /* Slate 900 */
        color: #f1f5f9; /* Slate 100 */
    }
    
    /* Custom Glassmorphic Card class */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
    }
    
    /* Metric Card Custom Styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Gradient Text Header */
    .gradient-text {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Custom Styled Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.6) !important;
        background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
    }
    
    /* Text input area styles */
    .stTextArea textarea, .stTextInput input {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 1px #6366f1 !important;
    }
    
    /* Tabs Style Override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: #94a3b8;
        padding: 0 20px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9;
        background-color: rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.2) !important;
        border-color: #6366f1 !important;
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- CACHED RESOURCES & HELPERS ---
@st.cache_resource
def get_db_session():
    """Initializes and caches the database session maker."""
    return SessionLocal()

@st.cache_resource
def load_cached_model():
    """Loads and caches the trained ML pipeline from disk."""
    model_path = os.path.join("models", "classifier_pipeline.joblib")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

@st.cache_resource
def load_cached_metrics():
    """Loads and caches model evaluation metrics from disk."""
    metrics_path = os.path.join("models", "training_metrics.joblib")
    if os.path.exists(metrics_path):
        try:
            return joblib.load(metrics_path)
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    return None


def get_db_stats(session):
    """Retrieve high-level statistics from the database."""
    stats = {}
    try:
        stats['total_trials'] = session.query(ClinicalTrial).count()
        
        # Get count per category
        categories = session.query(ClinicalTrial.disease_category).distinct().all()
        stats['unique_categories'] = len(categories)
        
        # Get overall status counts
        status_counts = pd.read_sql(
            "SELECT overall_status, COUNT(*) as count FROM clinical_trials GROUP BY overall_status",
            engine
        )
        stats['status_df'] = status_counts
        
        # Get phase counts
        phase_counts = pd.read_sql(
            "SELECT phase, COUNT(*) as count FROM clinical_trials GROUP BY phase",
            engine
        )
        stats['phase_df'] = phase_counts
        
        # Log counts
        stats['total_logs'] = session.query(PredictionLog).count()
        
    except Exception as e:
        stats['total_trials'] = 0
        stats['unique_categories'] = 0
        stats['total_logs'] = 0
        stats['status_df'] = pd.DataFrame()
        stats['phase_df'] = pd.DataFrame()
        
    return stats


# --- INTERFACE ENTRY POINT ---

db = get_db_session()
db_stats = get_db_stats(db)

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.markdown("<h2 class='gradient-text' style='font-size: 1.8rem; font-weight: 700;'>Clinical Trial AI</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### System Status")
    
    # DB Status check
    if db_stats['total_trials'] > 0:
        st.success(f"Database: Connected ({db_stats['total_trials']:,} trials)")
    else:
        st.warning("Database: Empty")
        
    # Model Status Check
    model = load_cached_model()
    metrics = load_cached_metrics()
    
    if model is not None:
        st.success(f"ML Model: Active (Acc: {metrics['accuracy']:.2%})")
    else:
        st.error("ML Model: Inactive (Needs Training)")
        
    st.markdown("---")
    st.markdown("### Project Mentorship")
    st.info("College Project Submission: Clinical Trial Disease Category Classification System using NLP, SQLAlchemy & Streamlit.")
    
    st.markdown("<div style='font-size: 0.8rem; color: #64748b; margin-top: 50px;'>Antigravity Developer Assistant • v1.0.0</div>", unsafe_allow_html=True)


# --- MAIN HEADER ---
st.markdown("""
<div style="margin-bottom: 35px;">
    <h1 class="gradient-text" style="font-size: 2.8rem; font-weight: 800; margin-bottom: 2px;">Clinical Trial Intelligence Hub</h1>
    <p style="color: #94a3b8; font-size: 1.1rem; font-weight: 400; margin-top: 0px;">Real-time text preprocessing, ETL, and ML classification dashboard for drug studies</p>
</div>
""", unsafe_allow_html=True)


# Create App Tabs
tab_dash, tab_classify, tab_explore, tab_diagnostics, tab_system = st.tabs([
    "📊 Analytics Dashboard", 
    "🔮 Trial Classifier", 
    "🔍 Database Explorer", 
    "🎯 Model Diagnostics & XAI", 
    "⚙️ Pipeline Management"
])


# ==========================================
# TAB 1: ANALYTICS DASHBOARD
# ==========================================
with tab_dash:
    if db_stats['total_trials'] == 0:
        st.info("👋 Welcome! The database is currently empty. Please navigate to the **Pipeline Management** tab to ingest your CSV dataset and train the classification model.")
    else:
        # Row 1: KPI metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-value">{db_stats['total_trials']:,}</div>
                <div class="metric-label">Total Ingested Trials</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-value">{db_stats['unique_categories']}</div>
                <div class="metric-label">Disease Categories</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            acc_val = f"{metrics['accuracy']:.2%}" if metrics else "N/A"
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-value">{acc_val}</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-value">{db_stats['total_logs']}</div>
                <div class="metric-label">App Predictions Logged</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Row 2: Charts and Word Cloud
        chart_col1, chart_col2 = st.columns([3, 2])
        
        with chart_col1:
            st.markdown("<h3 style='margin-bottom:15px;'>Disease Category Distribution</h3>", unsafe_allow_html=True)
            
            # Query category counts from DB
            cat_df = pd.read_sql(
                "SELECT disease_category, COUNT(*) as count FROM clinical_trials GROUP BY disease_category ORDER BY count DESC",
                engine
            )
            
            fig_bar = px.bar(
                cat_df,
                x='count',
                y='disease_category',
                orientation='h',
                labels={'count': 'Number of Trials', 'disease_category': 'Therapeutic Area'},
                color='count',
                color_continuous_scale='purples',
                height=400
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False
            )
            fig_bar.update_yaxes(categoryorder='total ascending')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with chart_col2:
            st.markdown("<h3 style='margin-bottom:15px;'>Trial Phases Distribution</h3>", unsafe_allow_html=True)
            
            phase_df = db_stats['phase_df'].copy()
            phase_df['phase'] = phase_df['phase'].fillna('Unknown/Not Applicable')
            phase_df = phase_df[phase_df['phase'] != '']
            
            fig_pie = px.pie(
                phase_df,
                names='phase',
                values='count',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # Row 3: Word Cloud by Category
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-bottom:15px;'>Therapeutic Area Keywords (Word Cloud)</h3>", unsafe_allow_html=True)
        
        # Get unique category list
        unique_cats = cat_df['disease_category'].tolist()
        selected_cloud_cat = st.selectbox("Select a disease category to generate a word cloud:", unique_cats)
        
        if selected_cloud_cat:
            # Query preprocessed text for this category
            sample_texts = db.query(ClinicalTrial.cleaned_summary)\
                .filter(ClinicalTrial.disease_category == selected_cloud_cat)\
                .limit(1000).all()
                
            combined_text = " ".join([t[0] for t in sample_texts if t[0]])
            
            if combined_text:
                wordcloud = WordCloud(
                    width=800, 
                    height=350, 
                    background_color='#0f172a',
                    colormap='plasma',
                    max_words=100
                ).generate(combined_text)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                fig.patch.set_facecolor('#0f172a')
                
                st.pyplot(fig)
            else:
                st.info("No cleaned text available to construct word cloud.")


# ==========================================
# TAB 2: INTERACTIVE TRIAL CLASSIFIER
# ==========================================
with tab_classify:
    if model is None:
        st.error("⚠️ Model classifier not found! Please ingest the data and run the model training script under the **Pipeline Management** tab.")
    else:
        st.markdown("<h3 style='margin-bottom:10px;'>Predict Disease Category of a New Clinical Trial</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom:20px;'>Input the trial details below to preprocess and classify the trial into oncology, cardiology, diabetes, anxiety, glaucoma, etc.</p>", unsafe_allow_html=True)
        
        # Ingestion form
        with st.form("classification_form"):
            input_title = st.text_input("Clinical Trial Title", placeholder="e.g., Effect of Low Dose Aspirin on Cardiovascular Risks in Elderly Patients")
            input_summary = st.text_area("Brief Summary / Study Protocol Description", height=200, placeholder="Paste details of the trial here...")
            input_eligibility = st.text_area("Inclusion/Exclusion Criteria (Optional)", height=100, placeholder="Inclusion criteria: Age > 65...")
            
            submit_btn = st.form_submit_button("Run Classification Pipeline")
            
        if submit_btn:
            if not input_summary.strip():
                st.error("Please enter at least a brief summary of the study.")
            else:
                # Combine Title + Summary for better context if needed, or stick to summary (which is what model is trained on)
                # Let's clean the summary using our pipeline function
                cleaned_text = clean_text_fast(input_summary)
                
                if not cleaned_text:
                    st.error("The summary provided could not be cleaned. Ensure it contains english words.")
                else:
                    # Run model inference
                    probabilities = model.predict_proba([cleaned_text])[0]
                    classes = model.classes_
                    
                    # Sort classes by probability
                    sort_indices = np.argsort(probabilities)[::-1]
                    pred_class = classes[sort_indices[0]]
                    pred_conf = probabilities[sort_indices[0]]
                    
                    # Store prediction ID in session state to handle feedback later
                    st.session_state['last_pred_title'] = input_title
                    st.session_state['last_pred_summary'] = input_summary
                    st.session_state['last_pred_eligibility'] = input_eligibility
                    st.session_state['last_pred_class'] = pred_class
                    st.session_state['last_pred_conf'] = float(pred_conf)
                    
                    # Log prediction to database using SQLAlchemy
                    log_entry = PredictionLog(
                        title=input_title,
                        brief_summary=input_summary,
                        eligibility_criteria=input_eligibility,
                        predicted_category=pred_class,
                        confidence=float(pred_conf)
                    )
                    db.add(log_entry)
                    db.commit()
                    st.session_state['last_log_id'] = log_entry.id
                    
                    # Display prediction in beautiful styling
                    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 25px 0;'>", unsafe_allow_html=True)
                    
                    col_res1, col_res2 = st.columns([1, 1])
                    
                    with col_res1:
                        st.markdown(f"""
                        <div class="glass-card" style="text-align: center; border-left: 5px solid #6366f1;">
                            <div class="metric-label" style="font-size:1rem;">Predicted Category</div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: #818cf8; text-transform: capitalize; margin: 10px 0;">
                                {pred_class}
                            </div>
                            <div class="metric-label" style="font-size:0.9rem;">Confidence Score</div>
                            <div class="metric-value" style="font-size: 2rem;">{pred_conf:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_res2:
                        st.markdown("<h4 style='margin-bottom:10px;'>Model Confidence Distribution</h4>", unsafe_allow_html=True)
                        
                        # Plotly probability bar chart
                        prob_df = pd.DataFrame({
                            'Category': classes,
                            'Probability': probabilities
                        }).sort_values(by='Probability', ascending=True)
                        
                        fig_probs = px.bar(
                            prob_df,
                            x='Probability',
                            y='Category',
                            orientation='h',
                            text='Probability',
                            color='Probability',
                            color_continuous_scale='plasma'
                        )
                        fig_probs.update_traces(texttemplate='%{x:.1%}', textposition='outside')
                        fig_probs.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#f1f5f9',
                            margin=dict(l=0, r=40, t=10, b=0),
                            coloraxis_showscale=False,
                            height=250
                        )
                        st.plotly_chart(fig_probs, use_container_width=True)
                        
        # Feedback Form (displays if last prediction exists in session)
        if 'last_log_id' in st.session_state:
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            with st.expander("🛠️ Provide Model Feedback / Correct Prediction"):
                st.write("Help improve the classifier! Let us know if the predicted disease class was correct.")
                
                feedback_col1, feedback_col2 = st.columns(2)
                with feedback_col1:
                    is_correct_feedback = st.radio(
                        "Is the prediction correct?",
                        ["Yes", "No"],
                        index=0
                    )
                with feedback_col2:
                    override_class = st.selectbox(
                        "Correct Disease Category (if No):",
                        metrics['classes'] if metrics else [st.session_state['last_pred_class']],
                        index=0
                    )
                
                feedback_submit = st.button("Submit Feedback")
                if feedback_submit:
                    log_id = st.session_state['last_log_id']
                    db_log = db.query(PredictionLog).filter(PredictionLog.id == log_id).first()
                    if db_log:
                        db_log.is_correct = is_correct_feedback
                        if is_correct_feedback == "No":
                            db_log.actual_category = override_class
                        else:
                            db_log.actual_category = db_log.predicted_category
                        db.commit()
                        st.success("Thank you for your feedback! The classification record has been updated.")
                        # Clean session state variables to hide feedback box until next prediction
                        del st.session_state['last_log_id']
                        # Force database stats reload
                        st.cache_resource.clear()


# ==========================================
# TAB 3: DATABASE EXPLORER
# ==========================================
with tab_explore:
    if db_stats['total_trials'] == 0:
        st.info("Database is empty. Please run ingestion pipeline.")
    else:
        st.markdown("<h3 style='margin-bottom:10px;'>Browse and Search Clinical Trials</h3>", unsafe_allow_html=True)
        st.markdown("Search across 60,337 entries in the SQLite database by keyword or filter by study parameters.", unsafe_allow_html=True)
        
        # Filters row
        fil_col1, fil_col2, fil_col3 = st.columns([2, 1, 1])
        
        with fil_col1:
            search_query = st.text_input("🔍 Search by keywords (NCT ID, Title, or Summary)", placeholder="e.g. Pembrolizumab, Type 2, NCT02941614")
            
        with fil_col2:
            # Query actual classes
            all_classes = sorted(metrics['classes']) if metrics else []
            if not all_classes:
                all_classes = [r[0] for r in db.query(ClinicalTrial.disease_category).distinct().all()]
            
            selected_cats = st.multiselect("Filter by Disease Category:", all_classes)
            
        with fil_col3:
            # Query actual phases
            all_phases = [r[0] for r in db.query(ClinicalTrial.phase).distinct().all() if r[0]]
            selected_phases = st.multiselect("Filter by Trial Phase:", sorted(all_phases))
            
        # Perform query
        query = db.query(ClinicalTrial)
        
        if search_query:
            query = query.filter(
                (ClinicalTrial.nct_id.ilike(f"%{search_query}%")) |
                (ClinicalTrial.title.ilike(f"%{search_query}%")) |
                (ClinicalTrial.brief_summary.ilike(f"%{search_query}%"))
            )
            
        if selected_cats:
            query = query.filter(ClinicalTrial.disease_category.in_(selected_cats))
            
        if selected_phases:
            query = query.filter(ClinicalTrial.phase.in_(selected_phases))
            
        total_matched = query.count()
        st.write(f"Showing top 20 of **{total_matched:,}** trials matching the filters:")
        
        # Fetch top 20
        results = query.limit(20).all()
        
        # Display results nicely
        for trial in results:
            with st.expander(f"📌 {trial.nct_id} - {trial.title or 'No Title'} ({trial.disease_category.capitalize()})"):
                st.markdown(f"**Official Title:** {trial.official_title or 'N/A'}")
                st.markdown(f"**URL:** [View on ClinicalTrials.gov]({trial.clinicaltrials_url or 'https://clinicaltrials.gov'})")
                
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                with meta_col1:
                    st.write(f"**Phase:** {trial.phase or 'N/A'}")
                    st.write(f"**Sex:** {trial.sex or 'All'}")
                with meta_col2:
                    st.write(f"**Status:** {trial.overall_status or 'N/A'}")
                    st.write(f"**Healthy Volunteers:** {trial.healthy_volunteers or 'N/A'}")
                with meta_col3:
                    st.write(f"**Min Age:** {trial.minimum_age or 'N/A'}")
                    st.write(f"**Max Age:** {trial.maximum_age or 'N/A'}")
                    
                st.markdown("---")
                st.markdown("**Brief Summary:**")
                st.write(trial.brief_summary)
                
                if trial.eligibility_criteria:
                    st.markdown("---")
                    st.markdown("**Eligibility Criteria:**")
                    st.code(trial.eligibility_criteria)


# ==========================================
# TAB 4: MODEL DIAGNOSTICS & EXPLAINABILITY
# ==========================================
with tab_diagnostics:
    if metrics is None:
        st.error("⚠️ Evaluation metrics not found! Please run the training script under the **Pipeline Management** tab first.")
    else:
        st.markdown("<h3 style='margin-bottom:10px;'>Model Diagnostics & Coefficient Explainability (XAI)</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom:20px;'>Under the hood of the Logistic Regression model, inspect prediction reports, confusion matrices, and the specific terms driving decisions.</p>", unsafe_allow_html=True)
        
        # Classification report display
        rep_df = pd.DataFrame(metrics['classification_report']).transpose()
        # Clean dataframe for nicer viewing
        rep_df = rep_df.round(4)
        
        col_m1, col_m2 = st.columns([3, 2])
        
        with col_m1:
            st.markdown("#### Category Metrics Report")
            # Filter report to show classes specifically
            classes_in_report = [c for c in metrics['classes']]
            report_classes_df = rep_df.loc[classes_in_report].rename(columns={
                'precision': 'Precision',
                'recall': 'Recall',
                'f1-score': 'F1-Score',
                'support': 'Support (Trials)'
            })
            st.dataframe(report_classes_df, use_container_width=True)
            
        with col_m2:
            st.markdown("#### Global Summary")
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Model Type</div>
                <div style="font-size:1.4rem; font-weight:700; color:#818cf8; margin-bottom:15px;">TF-IDF + Logistic Regression</div>
                
                <div class="metric-label">Total Training Size</div>
                <div style="font-size:1.4rem; font-weight:700; color:#818cf8; margin-bottom:15px;">{metrics['n_train_samples']:,} trials</div>
                
                <div class="metric-label">Test Evaluation Size</div>
                <div style="font-size:1.4rem; font-weight:700; color:#818cf8;">{metrics['n_test_samples']:,} trials</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Section 2: Confusion Matrix
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 25px 0;'>", unsafe_allow_html=True)
        st.markdown("#### Confusion Matrix Heatmap", unsafe_allow_html=True)
        
        # Plot confusion matrix using plotly heatmap
        cm = np.array(metrics['confusion_matrix'])
        classes_labels = [c.capitalize() for c in metrics['classes']]
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Therapeutic Area", y="Actual Therapeutic Area", color="Trials"),
            x=classes_labels,
            y=classes_labels,
            color_continuous_scale='purples',
            text_auto=True
        )
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f1f5f9',
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Section 3: Explainable AI - Coefficients
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 25px 0;'>", unsafe_allow_html=True)
        st.markdown("#### Explainable AI: Top Predictive Words", unsafe_allow_html=True)
        st.markdown("Select a category to inspect the top 15 words that have the highest positive coefficient weights for that class.", unsafe_allow_html=True)
        
        selected_xai_cat = st.selectbox("Select therapeutic category:", metrics['classes'])
        
        if selected_xai_cat:
            words_coefs = metrics['top_predictive_words'][selected_xai_cat][:15]
            words_df = pd.DataFrame(words_coefs, columns=['Word', 'Coefficient']).sort_values(by='Coefficient', ascending=True)
            
            fig_coef = px.bar(
                words_df,
                x='Coefficient',
                y='Word',
                orientation='h',
                color='Coefficient',
                color_continuous_scale='blues',
                title=f"Top Tokens Driving '{selected_xai_cat.capitalize()}' Predictions"
            )
            fig_coef.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                margin=dict(l=0, r=0, t=30, b=0),
                coloraxis_showscale=False,
                height=400
            )
            st.plotly_chart(fig_coef, use_container_width=True)


# ==========================================
# TAB 5: PIPELINE MANAGEMENT
# ==========================================
with tab_system:
    st.markdown("<h3 style='margin-bottom:10px;'>System & Pipeline Operations</h3>", unsafe_allow_html=True)
    st.markdown("Manage system database, run raw ETL operations from the CSV source file, and trigger ML pipeline retraining.", unsafe_allow_html=True)
    
    # Ingestion status card
    op_col1, op_col2 = st.columns(2)
    
    with op_col1:
        st.markdown("#### Database Details")
        st.write(f"**Database URI:** `sqlite:///clinical_trials.db` (configured locally)")
        st.write(f"**Clinical Trials Count:** {db_stats['total_trials']:,} records")
        st.write(f"**Logged User Predictions:** {db_stats['total_logs']:,} logs")
        
        st.markdown("---")
        st.markdown("#### ETL Ingestion Pipeline")
        st.write("Run this to parse the raw 158MB CSV dataset, preprocess summaries (lowercase, regex, stopwords, lemmatization), and populate the database table.")
        
        etl_btn = st.button("Trigger Ingest (Force Re-run ETL)")
        if etl_btn:
            with st.spinner("Extracting, Preprocessing & Loading data. This takes around 60 seconds..."):
                success = run_etl(force=True)
                if success:
                    st.success("🎉 Data pipeline completed! 60,337 trials loaded successfully!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error("ETL pipeline failed. Please check backend execution console logs.")
                    
    with op_col2:
        st.markdown("#### Model Training Control")
        st.write("Fit a TfidfVectorizer and LogisticRegression classifier on the preprocessed texts in the database. Outputs are saved in `models/` directory.")
        
        # Show model stats if exists
        if metrics:
            st.write(f"**Trained On:** {metrics['n_train_samples'] + metrics['n_test_samples']:,} samples")
            st.write(f"**Stored Pipeline Accuracy:** {metrics['accuracy']:.4%}")
            st.write(f"**Features Mapped:** 15,000 maximum")
        else:
            st.write("*Status: No saved model pipeline found on disk.*")
            
        st.markdown("---")
        train_btn = st.button("Re-train Machine Learning Classifier")
        if train_btn:
            if db_stats['total_trials'] == 0:
                st.error("Cannot train model on empty database! Ingest the dataset first.")
            else:
                with st.spinner("Training TF-IDF + Logistic Regression model. Running fit and evaluations..."):
                    success = train_model()
                    if success:
                        st.success("🎉 ML Pipeline training completed! Model saved to disk.")
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error("Model training failed. See system console logs.")
                        
    # Display recent prediction logs
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("#### Prediction Log History (Live)", unsafe_allow_html=True)
    
    logs_df = pd.read_sql(
        "SELECT id, title, predicted_category, confidence, actual_category, is_correct, timestamp FROM prediction_logs ORDER BY timestamp DESC LIMIT 10",
        engine
    )
    if len(logs_df) > 0:
        st.dataframe(logs_df, use_container_width=True)
    else:
        st.info("No queries have been run or logged yet.")
