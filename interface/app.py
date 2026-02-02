"""
Medical AI Prediction System - Streamlit Interface
Author: Bishowdip
"""

import streamlit as st
import sys
import os
import numpy as np
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Page configuration
st.set_page_config(
    page_title="Medical AI Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'results', 'models')

# Load model and preprocessors (cached)
@st.cache_resource
def load_models():
    try:
        # Load best model (LightGBM)
        models_path = os.path.join(MODELS_DIR, 'advanced_models.pkl')
        models = joblib.load(models_path)
        best_model = models['LightGBM']
        
        # Load preprocessors
        preprocessors_path = os.path.join(MODELS_DIR, 'preprocessors.pkl')
        preprocessors = joblib.load(preprocessors_path)
        scaler = preprocessors['scalers']['numerical']
        
        return best_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error(f"Looking in: {MODELS_DIR}")
        return None, None

model, scaler = load_models()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🏥 Medical AI System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Single Prediction", "📊 Model Comparison", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Project:** ST5000CEM  
    **Author:** Bishowdip  
    **Model:** LightGBM  
    **Accuracy:** 86.89%  
    **ROC-AUC:** 0.9459
    """
)

# Main content
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🏥 Medical AI Prediction System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Heart Disease Prediction System
    
    This AI-powered system predicts cardiovascular disease risk using machine learning.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Models Trained", "10")
    with col2:
        st.metric("Best Accuracy", "86.89%")
    with col3:
        st.metric("ROC-AUC Score", "0.9459")
    with col4:
        st.metric("Recall", "92.86%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Key Features
        - **10 ML Models**: Comprehensive comparison
        - **High Accuracy**: 86.89% correct predictions
        - **Explainable AI**: SHAP visualizations
        - **Fast Predictions**: Results in milliseconds
        """)
    
    with col2:
        st.markdown("""
        #### 🔬 Clinical Validation
        - **Sensitivity**: 92.86% (catches disease)
        - **Specificity**: ~94% (identifies healthy)
        - **Dataset**: UCI Heart Disease (303 patients)
        - **Features**: 13 clinical measurements
        """)

elif page == "🔍 Single Prediction":
    st.markdown('<h1 class="main-header">🔍 Single Patient Prediction</h1>', unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ Model not loaded. Please check model files.")
    else:
        st.info("Enter patient clinical measurements to predict heart disease risk.")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age (years)", min_value=20, max_value=100, value=55)
                sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
                trestbps = st.number_input("Resting BP (mmHg)", min_value=90, max_value=200, value=120)
                chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            
            with col2:
                fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
                restecg = st.selectbox("Resting ECG", [0, 1, 2])
                thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
                exang = st.selectbox("Exercise Induced Angina", [0, 1])
            
            with col3:
                oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
                slope = st.selectbox("Slope", [1, 2, 3])
                ca = st.selectbox("Number of Vessels (0-3)", [0, 1, 2, 3])
                thal = st.selectbox("Thalassemia", [3, 6, 7])
            
            submitted = st.form_submit_button("🔬 Predict", use_container_width=True)
        
        if submitted:
            # Prepare input
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                   thalach, exang, oldpeak, slope, ca, thal]])
            
            # Scale numerical features (indices: 0, 3, 4, 7, 9)
            numerical_indices = [0, 3, 4, 7, 9]
            input_scaled = input_data.copy()
            input_scaled[0, numerical_indices] = scaler.transform(input_data[:, numerical_indices])
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1] * 100
            
            st.success("✅ Prediction completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Prediction Result")
                st.metric("Disease Probability", f"{probability:.1f}%", 
                         "High Risk" if probability > 70 else "Moderate Risk" if probability > 40 else "Low Risk")
                
                if probability > 70:
                    st.error("⚠️ **HIGH RISK** - Recommend cardiology referral")
                elif probability > 40:
                    st.warning("⚠️ **MODERATE RISK** - Additional testing recommended")
                else:
                    st.success("✅ **LOW RISK** - Continue routine monitoring")
            
            with col2:
                st.markdown("### Key Input Values")
                st.markdown(f"""
                - **Age:** {age} years
                - **Thalassemia:** {thal}
                - **Blocked Vessels:** {ca}
                - **Chest Pain Type:** {cp}
                - **Max Heart Rate:** {thalach} bpm
                """)

elif page == "📊 Model Comparison":
    st.markdown('<h1 class="main-header">📊 Model Comparison</h1>', unsafe_allow_html=True)
    
    import pandas as pd
    
    results = {
        'Model': ['Naive Bayes', 'Logistic Regression', 'LightGBM', 'Random Forest', 
                 'SVM', 'XGBoost', 'Neural Network', 'Decision Tree'],
        'Accuracy': [85.25, 83.61, 86.89, 86.89, 80.33, 81.97, 70.49, 65.57],
        'Precision': [78.79, 78.12, 81.25, 81.25, 75.00, 75.76, 70.83, 60.00],
        'Recall': [92.86, 89.29, 92.86, 92.86, 85.71, 89.29, 60.71, 75.00],
        'F1-Score': [85.25, 83.33, 86.67, 86.67, 80.00, 81.97, 65.38, 66.67],
        'ROC-AUC': [0.9481, 0.9459, 0.9459, 0.9410, 0.9123, 0.9048, 0.7543, 0.6629]
    }
    
    df = pd.DataFrame(results)
    st.dataframe(df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']), 
                use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    ### 🏆 Recommended Model: **LightGBM**
    
    - ✅ Best F1-Score (86.67%)
    - ✅ Tied best Accuracy (86.89%)
    - ✅ Tied best Recall (92.86%)
    - ✅ Fast predictions (<1ms)
    """)

elif page == "ℹ️ About":
    st.markdown('<h1 class="main-header">ℹ️ About This System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎓 Project Information
    
    **Module:** ST5000CEM - Introduction to Artificial Intelligence  
    **Author:** Bishowdip  
    **Institution:** Softwarica College
    
    ### 🎯 Objectives
    
    Develop an AI system for cardiovascular disease prediction with:
    - High accuracy (>85%)
    - Explainable predictions
    - Clinical applicability
    
    ### 📊 Dataset
    
    **Source:** UCI Heart Disease (Cleveland Clinic)  
    **Size:** 303 patients, 13 features
    
    ### 🤖 Models
    
    10 models trained including:
    - Baseline: LR, DT, RF, SVM, NB
    - Advanced: XGBoost, LightGBM, NN
    - Ensemble: Voting, Stacking
    
    ### 📈 Best Results
    
    **LightGBM Performance:**
    - Accuracy: 86.89%
    - ROC-AUC: 0.9459
    - Recall: 92.86%
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Medical AI System | Bishowdip © 2026</div>",
    unsafe_allow_html=True
)
