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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import SessionLocal, ClinicalTrial, PredictionLog, engine
from src.pipeline import clean_text_fast

# --- PAGE CONFIGURATION & PREMIUM THEMING ---
st.set_page_config(
    page_title="Clinical Trial Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Injected CSS for Light Theme Aesthetics
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
    
    /* Custom Background & Styling (Light Theme) */
    .stApp {
        background-color: #f8fafc; /* Light Slate */
        color: #0f172a; /* Dark Blue/Black */
    }
    
    /* Custom Glassmorphic Card class */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 12px 30px 0 rgba(99, 102, 241, 0.1);
    }
    
    /* Metric Card Custom Styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Gradient Text Header */
    .gradient-text {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 50%, #7c3aed 100%);
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
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.4) !important;
        background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
    }
    
    /* Text input area styles */
    .stTextArea textarea, .stTextInput input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
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
        background-color: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 10px;
        color: #64748b;
        padding: 0 20px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a;
        background-color: rgba(99, 102, 241, 0.05);
        border-color: rgba(99, 102, 241, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.1) !important;
        border-color: #6366f1 !important;
        color: #4f46e5 !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- CACHED RESOURCES & HELPERS ---
def get_db_session():
    return SessionLocal()

@st.cache_resource
def load_cached_model():
    model_path = os.path.join("models", "classifier_pipeline.joblib")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

@st.cache_resource
def load_cached_metrics():
    metrics_path = os.path.join("models", "training_metrics.joblib")
    if os.path.exists(metrics_path):
        try:
            return joblib.load(metrics_path)
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    return None

@st.cache_data
def get_db_stats():
    stats = {}
    try:
        session = get_db_session()
        stats['total_trials'] = session.query(ClinicalTrial).count()
        
        categories = session.query(ClinicalTrial.disease_category).distinct().all()
        stats['unique_categories'] = len(categories)
        
        status_counts = pd.read_sql(
            "SELECT overall_status, COUNT(*) as count FROM clinical_trials GROUP BY overall_status",
            engine
        )
        stats['status_df'] = status_counts
        
        phase_counts = pd.read_sql(
            "SELECT phase, COUNT(*) as count FROM clinical_trials GROUP BY phase",
            engine
        )
        stats['phase_df'] = phase_counts
        
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
db_stats = get_db_stats()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.markdown("<h2 class='gradient-text' style='font-size: 1.8rem; font-weight: 700;'>Clinical Trial AI</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### Executive Summary")
    st.write("This intelligence hub processes thousands of raw clinical trial protocols to automatically classify therapeutic disease categories using Natural Language Processing.")
    
    st.markdown("---")
    st.markdown("### System Health")
    model = load_cached_model()
    metrics = load_cached_metrics()
    
    if model is not None:
        st.success(f"ML Model: Online (Acc: {metrics['accuracy']:.2%})")
    else:
        st.error("ML Model: Offline")
        
    st.markdown("<div style='font-size: 0.8rem; color: #94a3b8; margin-top: 50px;'>Clinical Trial Intelligence Assistant • v1.0.0</div>", unsafe_allow_html=True)


# --- MAIN HEADER ---
st.markdown("""
<div style="margin-bottom: 35px;">
    <h1 class="gradient-text" style="font-size: 2.8rem; font-weight: 800; margin-bottom: 2px;">Clinical Trial Intelligence Hub</h1>
    <p style="color: #64748b; font-size: 1.1rem; font-weight: 500; margin-top: 0px;">Real-time text preprocessing and ML classification dashboard for drug studies</p>
</div>
""", unsafe_allow_html=True)


# Create App Tabs
tab_insights, tab_performance, tab_classifier = st.tabs([
    "📊 Insights & Trends", 
    "🎯 Model Performance",
    "🔮 Interactive Classifier"
])


# ==========================================
# TAB 1: INSIGHTS & TRENDS
# ==========================================
with tab_insights:
    if db_stats['total_trials'] == 0:
        st.info("No data currently available in the database.")
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
<div class="metric-label">User Predictions Run</div>
</div>
""", unsafe_allow_html=True)
            
        # Row 2: Charts and Word Cloud
        chart_col1, chart_col2 = st.columns([3, 2])
        
        with chart_col1:
            st.markdown("<h3 style='margin-bottom:15px; color:#0f172a;'>Disease Category Distribution</h3>", unsafe_allow_html=True)
            
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
                color_continuous_scale='blues',
                height=450
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#0f172a',
                margin=dict(l=200, r=20, t=10, b=0),
                coloraxis_showscale=False
            )
            fig_bar.update_yaxes(categoryorder='total ascending', automargin=True)
            st.plotly_chart(fig_bar, width="stretch", theme=None)
            
        with chart_col2:
            st.markdown("<h3 style='margin-bottom:15px; color:#0f172a;'>Trial Phases Distribution</h3>", unsafe_allow_html=True)
            
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
                font_color='#0f172a',
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, width="stretch", theme=None)
            
        # Phase Descriptions
        with st.expander("ℹ️ What do these trial phases mean?"):
            st.markdown("""
            * **Phase 1:** Tests safety, side effects, and best dosage on a small group of volunteers.
            * **Phase 2:** Tests efficacy and further evaluates safety on a larger group of patients with the condition.
            * **Phase 3:** Compares the new treatment to standard treatments on large groups to confirm effectiveness.
            * **Phase 4:** Post-market surveillance gathering information on long-term effects after approval.
            * **Early Phase 1:** Exploratory, first-in-human trials before Phase 1.
            * **Not Applicable:** Often used for observational studies or trials not testing a specific new drug.
            """)
            
        # Row 3: Word Cloud by Category
        st.markdown("<hr style='border-color: rgba(0,0,0,0.08); margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-bottom:15px; color:#0f172a;'>Frequently Occurring Medical Terms</h3>", unsafe_allow_html=True)
        
        unique_cats = cat_df['disease_category'].tolist()
        selected_cloud_cat = st.selectbox("Select a disease category to generate a word cloud:", unique_cats)
        
        if selected_cloud_cat:
            sample_texts = db.query(ClinicalTrial.cleaned_summary)\
                .filter(ClinicalTrial.disease_category == selected_cloud_cat)\
                .limit(1000).all()
                
            combined_text = " ".join([t[0] for t in sample_texts if t[0]])
            
            if combined_text:
                wordcloud = WordCloud(
                    width=800, 
                    height=350, 
                    background_color='#ffffff',
                    colormap='plasma',
                    max_words=100
                ).generate(combined_text)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                fig.patch.set_facecolor('#ffffff')
                
                st.pyplot(fig)
            else:
                st.info("No cleaned text available to construct word cloud.")


# ==========================================
# TAB 2: MODEL PERFORMANCE
# ==========================================
with tab_performance:
    if metrics is None:
        st.error("⚠️ Evaluation metrics not found!")
    else:
        st.markdown("<h3 style='margin-bottom:10px; color:#0f172a;'>Model Diagnostics & Prediction Accuracy</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; font-size:0.95rem; margin-bottom:20px;'>Inspect prediction reports, confusion matrices, and the specific terms driving AI decisions.</p>", unsafe_allow_html=True)
        
        # Classification report display
        rep_df = pd.DataFrame(metrics['classification_report']).transpose()
        rep_df = rep_df.round(4)
        
        col_m1, col_m2 = st.columns([3, 2])
        
        with col_m1:
            st.markdown("#### Category Metrics Report")
            classes_in_report = metrics['classes']
            encoded_keys = [str(i) for i in range(len(classes_in_report))]
            if set(encoded_keys).issubset(rep_df.index):
                report_classes_df = rep_df.loc[encoded_keys].copy()
                report_classes_df.index = classes_in_report
            else:
                report_classes_df = rep_df.loc[classes_in_report].copy()
                
            report_classes_df = report_classes_df.rename(columns={
                'precision': 'Precision',
                'recall': 'Recall',
                'f1-score': 'F1-Score',
                'support': 'Support (Trials)'
            })
            st.dataframe(report_classes_df, width="stretch")
            
        with col_m2:
            st.markdown("#### Global Summary")
            st.markdown(f"""
<div class="glass-card">
<div class="metric-label">Model Type</div>
<div style="font-size:1.4rem; font-weight:700; color:#4f46e5; margin-bottom:15px;">TF-IDF + Logistic Regression</div>

<div class="metric-label">Total Training Size</div>
<div style="font-size:1.4rem; font-weight:700; color:#4f46e5; margin-bottom:15px;">{metrics['n_train_samples']:,} trials</div>

<div class="metric-label">Test Evaluation Size</div>
<div style="font-size:1.4rem; font-weight:700; color:#4f46e5;">{metrics['n_test_samples']:,} trials</div>
</div>
""", unsafe_allow_html=True)
            
        # Section 2: Confusion Matrix
        st.markdown("<hr style='border-color: rgba(0,0,0,0.08); margin: 25px 0;'>", unsafe_allow_html=True)
        st.markdown("#### Confusion Matrix Heatmap", unsafe_allow_html=True)
        
        cm = np.array(metrics['confusion_matrix'])
        classes_labels = [c.capitalize() for c in metrics['classes']]
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Therapeutic Area", y="Actual Therapeutic Area", color="Trials"),
            x=classes_labels,
            y=classes_labels,
            color_continuous_scale='blues',
            text_auto=True
        )
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#0f172a',
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_cm, width="stretch", theme=None)
        
        # Section 3: Explainable AI - Coefficients
        st.markdown("<hr style='border-color: rgba(0,0,0,0.08); margin: 25px 0;'>", unsafe_allow_html=True)
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
                font_color='#0f172a',
                margin=dict(l=150, r=20, t=30, b=0),
                coloraxis_showscale=False,
                height=400
            )
            fig_coef.update_yaxes(automargin=True)
            st.plotly_chart(fig_coef, width="stretch", theme=None)


# ==========================================
# TAB 3: INTERACTIVE CLASSIFIER
# ==========================================
with tab_classifier:
    if model is None:
        st.error("⚠️ Model classifier not found!")
    else:
        st.markdown("<h3 style='margin-bottom:10px; color:#0f172a;'>Predict Disease Category of a New Clinical Trial</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; font-size:0.95rem; margin-bottom:20px;'>Input the trial details below to preprocess and classify the trial.</p>", unsafe_allow_html=True)
        
        with st.form("classification_form"):
            input_title = st.text_input("Clinical Trial Title", placeholder="e.g., Effect of Low Dose Aspirin on Cardiovascular Risks in Elderly Patients")
            input_summary = st.text_area("Brief Summary / Study Protocol Description", height=200, placeholder="Paste details of the trial here...")
            input_eligibility = st.text_area("Inclusion/Exclusion Criteria (Optional)", height=100, placeholder="Inclusion criteria: Age > 65...")
            
            submit_btn = st.form_submit_button("Run Classification Pipeline")
            
        if submit_btn:
            if not input_summary.strip():
                st.error("Please enter at least a brief summary of the study.")
            else:
                cleaned_text = clean_text_fast(input_summary)
                
                if not cleaned_text:
                    st.error("The summary provided could not be cleaned. Ensure it contains english words.")
                else:
                    probabilities = model.predict_proba([cleaned_text])[0]
                    if metrics and len(metrics['classes']) == len(model.classes_):
                        classes = metrics['classes']
                    else:
                        classes = [str(c) for c in model.classes_]
                    
                    sort_indices = np.argsort(probabilities)[::-1]
                    pred_class = str(classes[sort_indices[0]])
                    pred_conf = float(probabilities[sort_indices[0]])
                    
                    st.session_state['last_pred_title'] = input_title
                    st.session_state['last_pred_summary'] = input_summary
                    st.session_state['last_pred_eligibility'] = input_eligibility
                    st.session_state['last_pred_class'] = pred_class
                    st.session_state['last_pred_conf'] = float(pred_conf)
                    
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
                    
                    st.markdown("<hr style='border-color: rgba(0,0,0,0.08); margin: 25px 0;'>", unsafe_allow_html=True)
                    
                    col_res1, col_res2 = st.columns([1, 1])
                    
                    with col_res1:
                        st.markdown(f"""
<div class="glass-card" style="text-align: center; border-left: 5px solid #4f46e5;">
<div class="metric-label" style="font-size:1rem;">Predicted Category</div>
<div style="font-size: 2.2rem; font-weight: 800; color: #4f46e5; text-transform: capitalize; margin: 10px 0;">
{pred_class}
</div>
<div class="metric-label" style="font-size:0.9rem;">Confidence Score</div>
<div class="metric-value" style="font-size: 2rem; background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
{pred_conf:.2%}
</div>
</div>
""", unsafe_allow_html=True)
                        
                    with col_res2:
                        st.markdown("<h4 style='margin-bottom:10px; color:#0f172a;'>Model Confidence Distribution</h4>", unsafe_allow_html=True)
                        
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
                            color_continuous_scale='blues'
                        )
                        fig_probs.update_traces(texttemplate='%{x:.1%}', textposition='outside')
                        fig_probs.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#0f172a',
                            margin=dict(l=150, r=40, t=10, b=0),
                            coloraxis_showscale=False,
                            height=250
                        )
                        fig_probs.update_yaxes(automargin=True)
                        st.plotly_chart(fig_probs, width="stretch", theme=None)
                        
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
                    options = metrics['classes'] if metrics else [st.session_state['last_pred_class']]
                    try:
                        default_index = options.index(st.session_state['last_pred_class'])
                    except ValueError:
                        default_index = 0
                        
                    override_class = st.selectbox(
                        "Correct Disease Category (if No):",
                        options,
                        index=default_index,
                        disabled=(is_correct_feedback == "Yes")
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
                            db_log.actual_category = st.session_state['last_pred_class']
                        db.commit()
                        st.success("Thank you for your feedback! The database has been updated.")
