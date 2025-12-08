"""
Main Streamlit Application
Heart Disease Prediction System - Interactive Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from features.data_cleaning import DataCleaner, load_sample_data
from models.model_trainer import ModelTrainer
from xai.explainability import XAIExplainer
from features.risk_calculator import RiskCalculator
from features.symptom_analyzer import SymptomAnalyzer
from features.lifestyle_recommender import LifestyleRecommender
# from models.hyperparameter_tuning import HyperparameterTuner  # commented per request
# from models.ensemble_model import EnsembleModel  # commented per request
from utils.database import Database
from utils.auth import Auth
from utils.pdf_generator import PDFReportGenerator
from features.voice_analysis import VoiceStressAnalyzer
from features.chatbot import HealthChatbot

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c5aa0;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4788;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f4788;
        color: white;
        font-weight: bold;
    }
    .disclaimer {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# Initialize components
auth = Auth()
db = Database()
risk_calc = RiskCalculator()
symptom_analyzer = SymptomAnalyzer()
lifestyle_recommender = LifestyleRecommender()
pdf_generator = PDFReportGenerator()
voice_analyzer = VoiceStressAnalyzer()
chatbot = HealthChatbot()

def main():
    """Main application"""
    # Header
    st.markdown('<div class="main-header">🫀 Heart Disease Prediction System</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and research purposes only. 
        It does not replace professional medical diagnosis, treatment, or advice. 
        Always consult with qualified healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_main_app()

def show_login_page():
    """Show login/registration page"""
    st.sidebar.title("🔐 Authentication")
    
    tab1, tab2 = st.sidebar.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            user, message = auth.login_user(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user_id = user['id']
                st.session_state.username = user['username']
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error(message)
    
    with tab2:
        st.subheader("Register")
        new_username = st.text_input("Username", key="reg_username")
        new_email = st.text_input("Email", key="reg_email")
        new_password = st.text_input("Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match!")
            else:
                user_id, message = auth.register_user(new_username, new_email, new_password)
                if user_id:
                    st.success("Registration successful! Please login.")
                else:
                    st.error(message)
    
    # Show welcome message
    st.markdown("""
    ## Welcome to the Heart Disease Prediction System
    
    This comprehensive ML-based system provides:
    
    - **Multi-Model Comparison**: Compare Logistic Regression, Random Forest, XGBoost, SVM, and Neural Networks
    - **Explainable AI**: SHAP values, LIME explanations, and feature importance
    - **Real-Time Predictions**: Instant risk assessment
    - **Risk Score Calculator**: Framingham-style risk calculation
    - **Symptom Analysis**: NLP-based symptom analysis
    - **Lifestyle Recommendations**: Personalized health advice
    - **And much more!**
    
    Please login or register to continue.
    """)

def show_main_app():
    """Show main application"""
    # Sidebar
    st.sidebar.title(f"👤 {st.session_state.username}")
    
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        [
            "🏠 Dashboard",
            "🔮 Make Prediction",
            "📊 Model Training & Comparison",
            "🧠 Explainable AI (XAI)",
            "📈 Risk Calculator",
            "💬 Symptom Analyzer",
            "💡 Lifestyle Recommendations",
        # "🎯 Hyperparameter Tuning",  # commented per request
        # "🔧 Ensemble Model",  # commented per request
            "📁 Prediction History",
            "🤖 AI Health Chatbot",
            "🎤 Voice Stress Analysis",
            "📄 Generate PDF Report",
            "🧹 Data Cleaning Module"
        ]
    )
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔮 Make Prediction":
        show_prediction_page()
    elif page == "📊 Model Training & Comparison":
        show_model_training()
    elif page == "🧠 Explainable AI (XAI)":
        show_xai_page()
    elif page == "📈 Risk Calculator":
        show_risk_calculator()
    elif page == "💬 Symptom Analyzer":
        show_symptom_analyzer()
    elif page == "💡 Lifestyle Recommendations":
        show_lifestyle_recommendations()
    # elif page == "🎯 Hyperparameter Tuning":
    #     show_hyperparameter_tuning()
    # elif page == "🔧 Ensemble Model":
    #     show_ensemble_model()
    elif page == "📁 Prediction History":
        show_prediction_history()
    elif page == "🤖 AI Health Chatbot":
        show_chatbot()
    elif page == "🎤 Voice Stress Analysis":
        show_voice_analysis()
    elif page == "📄 Generate PDF Report":
        show_pdf_generator()
    elif page == "🧹 Data Cleaning Module":
        show_data_cleaning()

def show_dashboard():
    """Show main dashboard"""
    st.title("📊 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get user stats
    predictions = db.get_user_predictions(st.session_state.user_id, limit=100)
    
    with col1:
        st.metric("Total Predictions", len(predictions))
    
    with col2:
        if predictions:
            avg_risk = np.mean([p['risk_score'] for p in predictions if p['risk_score']])
            st.metric("Average Risk Score", f"{avg_risk:.1f}%" if avg_risk else "N/A")
        else:
            st.metric("Average Risk Score", "N/A")
    
    with col3:
        if predictions:
            latest = predictions[0]
            st.metric("Latest Risk", f"{latest['risk_score']:.1f}%" if latest['risk_score'] else "N/A")
        else:
            st.metric("Latest Risk", "N/A")
    
    with col4:
        if predictions:
            high_risk_count = sum(1 for p in predictions if p.get('risk_score', 0) >= 50)
            st.metric("High Risk Predictions", high_risk_count)
        else:
            st.metric("High Risk Predictions", 0)
    
    # Prediction trend chart
    if predictions:
        st.subheader("📈 Prediction History Trend")
        df_history = pd.DataFrame(predictions)
        df_history['created_at'] = pd.to_datetime(df_history['created_at'])
        df_history = df_history.sort_values('created_at')
        
        fig = px.line(df_history, x='created_at', y='risk_score',
                     title='Risk Score Over Time',
                     labels={'risk_score': 'Risk Score (%)', 'created_at': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔮 Make New Prediction"):
            st.session_state.page = "Make Prediction"
            st.rerun()
    
    with col2:
        if st.button("📊 Train Models"):
            st.session_state.page = "Model Training"
            st.rerun()
    
    with col3:
        if st.button("💬 Ask Chatbot"):
            st.session_state.page = "Chatbot"
            st.rerun()

def show_prediction_page():
    """Show prediction input page"""
    st.title("🔮 Make Prediction")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Models not trained yet. Please train models first or use sample data.")
        if st.button("Train Models with Sample Data"):
            with st.spinner("Training models..."):
                train_models_with_sample_data()
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
            cp = st.selectbox("Chest Pain Type", [
                ("Typical Angina", 0),
                ("Atypical Angina", 1),
                ("Non-anginal Pain", 2),
                ("Asymptomatic", 3)
            ], format_func=lambda x: x[0])[1]
            trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
            chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        
        with col2:
            restecg = st.selectbox("Resting ECG", [
                ("Normal", 0),
                ("ST-T Wave Abnormality", 1),
                ("Left Ventricular Hypertrophy", 2)
            ], format_func=lambda x: x[0])[1]
            thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise", [
                ("Upsloping", 0),
                ("Flat", 1),
                ("Downsloping", 2)
            ], format_func=lambda x: x[0])[1]
            ca = st.number_input("Number of Major Vessels", min_value=0, max_value=3, value=0)
            thal = st.selectbox("Thalassemia", [
                ("Normal", 0),
                ("Fixed Defect", 1),
                ("Reversible Defect", 2)
            ], format_func=lambda x: x[0])[1]
        
        submitted = st.form_submit_button("🔮 Predict Risk")
        
        if submitted and st.session_state.best_model:
            # Prepare features
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            
            # Make prediction
            if st.session_state.best_model_name in ['Logistic Regression', 'SVM', 'Neural Network']:
                features_scaled = st.session_state.scaler.transform(features)
                prediction = st.session_state.best_model.predict(features_scaled)[0]
                prediction_proba = st.session_state.best_model.predict_proba(features_scaled)[0]
            else:
                prediction = st.session_state.best_model.predict(features)[0]
                prediction_proba = st.session_state.best_model.predict_proba(features)[0]
            
            risk_probability = prediction_proba[1]
            
            # Calculate risk score
            risk_assessment = risk_calc.get_risk_category(risk_probability * 100)
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", "Heart Disease Risk" if prediction == 1 else "No Risk")
            
            with col2:
                st.metric("Probability", f"{risk_probability * 100:.2f}%")
            
            with col3:
                st.metric("Risk Category", risk_assessment['category'])
            
            # Risk assessment
            st.subheader("📊 Risk Assessment")
            risk_color = risk_assessment['color']
            st.markdown(f"""
            <div style="background-color: {risk_color}; padding: 1rem; border-radius: 0.5rem; color: white;">
                <h3>{risk_assessment['category']}</h3>
                <p>{risk_assessment['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Save prediction
            features_dict = {
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
                'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
                'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
            }
            
            db.save_prediction(
                st.session_state.user_id,
                int(prediction),
                float(risk_probability),
                risk_score=risk_probability * 100,
                risk_category=risk_assessment['category'],
                features=features_dict,
                model_used=st.session_state.best_model_name
            )
            
            st.success("Prediction saved to history!")

def train_models_with_sample_data():
    """Train models with sample data"""
    # Load sample data
    df = load_sample_data()
    
    # Clean data
    cleaner = DataCleaner(df)
    cleaner.remove_duplicates()
    cleaner.handle_missing_values()
    df_clean = cleaner.get_cleaned_data()
    
    # Prepare features
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X = df_clean[feature_cols]
    y = df_clean['target']
    
    st.session_state.feature_names = feature_cols
    
    # Train models
    trainer = ModelTrainer(X, y)
    trainer.train_all_models()
    trainer.evaluate_all_models()
    
    # Get best model
    best_name, best_model, best_results = trainer.get_best_model()
    
    st.session_state.models_trained = True
    st.session_state.trained_models = trainer.models
    st.session_state.best_model = best_model
    st.session_state.best_model_name = best_name
    st.session_state.scaler = trainer.scaler
    st.session_state.trainer = trainer
    
    st.success(f"✅ Models trained! Best model: {best_name}")

def show_model_training():
    """Show model training page"""
    st.title("📊 Model Training & Comparison")
    
    if st.button("🔄 Train All Models"):
        with st.spinner("Training models... This may take a few minutes."):
            train_models_with_sample_data()
    
    if st.session_state.models_trained and 'trainer' in st.session_state:
        trainer = st.session_state.trainer
        
        # Comparison table
        st.subheader("📈 Model Comparison")
        comparison_df = trainer.get_comparison_dataframe()
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curves")
            fig_roc = trainer.plot_roc_curves()
            st.pyplot(fig_roc)
        
        with col2:
            st.subheader("Confusion Matrices")
            fig_cm = trainer.plot_confusion_matrices()
            st.pyplot(fig_cm)
        
        # Best model info
        st.subheader("🏆 Best Model")
        st.info(f"Best Model: **{st.session_state.best_model_name}**")
        
        if st.session_state.best_model_name in trainer.results:
            best_results = trainer.results[st.session_state.best_model_name]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{best_results['accuracy']:.3f}")
            col2.metric("Precision", f"{best_results['precision']:.3f}")
            col3.metric("Recall", f"{best_results['recall']:.3f}")
            col4.metric("F1-Score", f"{best_results['f1_score']:.3f}")

def show_xai_page():
    """Show Explainable AI page"""
    st.title("🧠 Explainable AI (XAI)")
    
    if not st.session_state.models_trained or 'trainer' not in st.session_state:
        st.warning("Please train models first!")
        return
    
    model_name = st.selectbox("Select Model to Explain", list(st.session_state.trained_models.keys()))
    model = st.session_state.trained_models[model_name]
    
    # Get training data
    trainer = st.session_state.trainer
    
    # Initialize XAI explainer
    explainer = XAIExplainer(
        model,
        trainer.X_train,
        trainer.X_test,
        st.session_state.feature_names,
        model_name
    )
    
    # Feature importance
    st.subheader("📊 Feature Importance")
    importance_df = explainer.get_feature_importance()
    if importance_df is not None:
        st.dataframe(importance_df, use_container_width=True)
        
        fig_importance = explainer.plot_feature_importance()
        st.pyplot(fig_importance)
    
    # SHAP values
    st.subheader("🔍 SHAP Values")
    if st.button("Calculate SHAP Values"):
        with st.spinner("Calculating SHAP values..."):
            fig_shap = explainer.plot_shap_summary()
            if fig_shap:
                st.pyplot(fig_shap)
    
    # LIME explanation
    st.subheader("🍋 LIME Explanation")
    test_size = len(trainer.X_test) if hasattr(trainer, 'X_test') else 10
    instance_idx = st.number_input("Select Test Instance", min_value=0, max_value=max(0, test_size-1), value=0)
    
    if st.button("Generate LIME Explanation"):
        if hasattr(trainer, 'X_test'):
            if hasattr(trainer.X_test, 'iloc'):
                instance = trainer.X_test.iloc[instance_idx]
            else:
                instance = trainer.X_test[instance_idx]
            fig_lime = explainer.plot_lime_explanation(instance)
            if fig_lime:
                st.pyplot(fig_lime)
        else:
            st.warning("Test data not available")

def show_risk_calculator():
    """Show risk calculator page"""
    st.title("📈 Risk Score Calculator")
    
    with st.form("risk_calculator_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
            bp_systolic = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120)
            bp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80)
        
        with col2:
            cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
            hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20, max_value=100, value=50)
            smoking = st.selectbox("Smoking", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            diabetes = st.selectbox("Diabetes", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        
        submitted = st.form_submit_button("Calculate Risk")
        
        if submitted:
            risk_score, risk_percentage = risk_calc.calculate_framingham_risk(
                age, sex, bp_systolic, bp_diastolic, cholesterol, hdl, smoking, diabetes
            )
            
            risk_assessment = risk_calc.get_risk_category(risk_percentage)
            
            st.success("Risk calculated!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Score", f"{risk_score:.0f}")
            col2.metric("Risk Percentage", f"{risk_percentage:.1f}%")
            col3.metric("Category", risk_assessment['category'])
            
            st.markdown(f"""
            <div style="background-color: {risk_assessment['color']}; padding: 1rem; border-radius: 0.5rem; color: white;">
                <h3>{risk_assessment['category']}</h3>
                <p>{risk_assessment['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk factors breakdown
            st.subheader("Risk Factors Breakdown")
            factors = risk_calc.get_risk_factors_breakdown(
                age, sex, bp_systolic, bp_diastolic, cholesterol, hdl, smoking, diabetes
            )
            if factors:
                st.dataframe(pd.DataFrame(factors), use_container_width=True)

def show_symptom_analyzer():
    """Show symptom analyzer page"""
    st.title("💬 Symptom Analyzer")
    
    symptom_text = st.text_area(
        "Describe your symptoms",
        placeholder="e.g., Chest pain while walking, shortness of breath, feeling dizzy..."
    )
    
    if st.button("Analyze Symptoms"):
        if symptom_text:
            analysis = symptom_analyzer.analyze_symptoms(symptom_text)
            
            st.subheader("Analysis Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Score", f"{analysis['risk_score']:.1f}")
            with col2:
                st.metric("Risk Category", analysis['risk_category'])
            
            st.subheader("Symptoms Detected")
            st.write(f"Found {analysis['symptom_count']} symptom(s)")
            for symptom in analysis['symptoms_found']:
                st.write(f"- **{symptom['category']}**: {symptom['keyword']} (Severity: {symptom['severity']})")
            
            st.subheader("Urgency Assessment")
            urgency = analysis['urgency']
            urgency_color = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}[urgency['level']]
            st.markdown(f"""
            <div style="background-color: {urgency_color}; padding: 1rem; border-radius: 0.5rem; color: white;">
                <h3>{urgency['level']} Urgency</h3>
                <p><strong>Timeframe:</strong> {urgency['timeframe']}</p>
                <p><strong>Recommendation:</strong> {urgency['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Possible Causes")
            for cause in analysis['possible_causes']:
                st.write(f"**{cause['condition']}** ({cause['probability']} probability)")
                st.write(cause['description'])
            
            st.subheader("Recommendations")
            for rec in analysis['recommendations']:
                st.write(f"• {rec}")

def show_lifestyle_recommendations():
    """Show lifestyle recommendations page"""
    st.title("💡 Lifestyle Recommendations")
    
    with st.form("lifestyle_form"):
        st.subheader("Your Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
            bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
            bp_systolic = st.number_input("Systolic BP", min_value=50, max_value=250, value=120)
            bp_diastolic = st.number_input("Diastolic BP", min_value=30, max_value=150, value=80)
        
        with col2:
            cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
            hdl = st.number_input("HDL", min_value=20, max_value=100, value=50)
            smoking = st.selectbox("Smoking", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            diabetes = st.selectbox("Diabetes", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            physical_activity = st.number_input("Physical Activity (min/week)", min_value=0, max_value=1000, value=150)
            sleep_hours = st.number_input("Sleep Hours/Night", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            stress_level = st.slider("Stress Level (0-10)", min_value=0, max_value=10, value=5)
        
        submitted = st.form_submit_button("Get Recommendations")
        
        if submitted:
            recommendations = lifestyle_recommender.analyze_user_profile(
                age, sex, bmi, bp_systolic, bp_diastolic, cholesterol, hdl,
                smoking, diabetes, physical_activity, sleep_hours, stress_level
            )
            
            for category, items in recommendations.items():
                if items:
                    st.subheader(category.replace('_', ' ').title())
                    for item in items:
                        if isinstance(item, dict):
                            st.write(f"**{item.get('category', '')}** ({item.get('priority', '')} Priority)")
                            for rec in item.get('recommendations', []):
                                st.write(f"• {rec}")


def show_prediction_history():
    """Show prediction history"""
    st.title("📁 Prediction History")
    
    predictions = db.get_user_predictions(st.session_state.user_id)
    
    if predictions:
        df = pd.DataFrame(predictions)
        st.dataframe(df, use_container_width=True)
        
        # Statistics
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(df))
        if df['risk_score'].notna().any():
            col2.metric("Average Risk", f"{df['risk_score'].mean():.1f}%")
            col3.metric("Latest Risk", f"{df.iloc[0]['risk_score']:.1f}%" if df.iloc[0]['risk_score'] else "N/A")
    else:
        st.info("No predictions yet. Make your first prediction!")

def show_chatbot():
    """Show chatbot page"""
    st.title("🤖 AI Health Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about heart health, predictions, or recommendations..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get chatbot response
        response = chatbot.process_query(prompt)
        
        # Add assistant response
        assistant_message = response.get('response', 'I apologize, I did not understand that.')
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        with st.chat_message("assistant"):
            st.markdown(assistant_message)
            
            if 'suggestions' in response:
                st.write("**Suggestions:**")
                for suggestion in response['suggestions']:
                    st.write(f"• {suggestion}")

def show_voice_analysis():
    """Show voice analysis page"""
    st.title("🎤 Voice Stress Analysis")
    
    st.info("Upload an audio file (WAV, MP3) to analyze stress levels.")
    
    audio_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
    
    if audio_file:
        st.audio(audio_file)
        
        if st.button("Analyze Stress"):
            with st.spinner("Analyzing voice patterns..."):
                # Save uploaded file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    analysis = voice_analyzer.analyze_audio_file(tmp_path)
                    
                    if 'error' not in analysis:
                        st.subheader("Stress Analysis Results")
                        col1, col2 = st.columns(2)
                        col1.metric("Stress Score", f"{analysis['stress_score']:.1f}/100")
                        col2.metric("Stress Level", analysis['stress_level'])
                        
                        st.subheader("Recommendations")
                        for rec in analysis['recommendations']:
                            st.write(f"• {rec}")
                    else:
                        st.error(analysis['error'])
                finally:
                    os.unlink(tmp_path)

def show_pdf_generator():
    """Show PDF generator page"""
    st.title("📄 Generate PDF Report")
    
    st.info("Generate a comprehensive PDF report of your latest prediction.")
    
    if st.button("Generate Report"):
        predictions = db.get_user_predictions(st.session_state.user_id, limit=1)
        
        if predictions:
            latest = predictions[0]
            features = latest.get('features', {})
            
            # Generate PDF
            pdf_path = f"report_{st.session_state.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            pdf_generator.generate_report(
                pdf_path,
                features,
                {
                    'prediction': latest['prediction_result'],
                    'probability': latest['prediction_probability']
                },
                {
                    'category': latest.get('risk_category', 'Unknown'),
                    'risk_percentage': latest.get('risk_score', 0),
                    'recommendation': "Consult with healthcare provider"
                },
                {},
                {'model_name': latest.get('model_used', 'Unknown')}
            )
            
            st.success("PDF report generated!")
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("Download PDF", pdf_file, file_name=pdf_path)
        else:
            st.warning("No predictions found. Make a prediction first!")

def show_data_cleaning():
    """Show data cleaning module"""
    st.title("🧹 Data Cleaning Module")
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Original Data")
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape}")
        
        cleaner = DataCleaner(df)
        
        # Duplicates
        duplicates = cleaner.detect_duplicates()
        st.metric("Duplicates Found", duplicates)
        
        if st.button("Remove Duplicates"):
            df_clean = cleaner.remove_duplicates()
            st.success("Duplicates removed!")
        
        # Missing values
        missing_df = cleaner.detect_missing_values()
        if not missing_df.empty:
            st.subheader("Missing Values")
            st.dataframe(missing_df)
        
        # Outliers
        if st.button("Detect Outliers (Z-score)"):
            outliers = cleaner.detect_outliers_zscore()
            st.json(outliers)
        
        # Correlation
        if st.button("Show Correlation Heatmap"):
            fig = cleaner.plot_correlation_heatmap()
            st.pyplot(fig)

if __name__ == "__main__":
    main()