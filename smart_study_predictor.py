import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Study Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2563eb;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
    }
    .recommendation-box {
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 15px;
        border-radius: 4px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# --- Synthetic Dataset & Model Generator ---
@st.cache_data
def generate_synthetic_data(samples=600):
    np.random.seed(42)
    study_hours = np.random.uniform(1.0, 10.0, samples)
    sleep_hours = np.random.uniform(4.0, 9.5, samples)
    attendance = np.random.uniform(50.0, 100.0, samples)
    past_score = np.random.uniform(40.0, 98.0, samples)
    distraction_level = np.random.uniform(1.0, 10.0, samples)  # 1 low, 10 high
    revision_freq = np.random.randint(1, 6, samples)  # 1 to 5 days a week

    # Synthetic target score calculation with non-linear relationships
    base_score = (
        (study_hours * 4.2) +
        (sleep_hours * 2.5) +
        (attendance * 0.25) +
        (past_score * 0.35) -
        (distraction_level * 1.8) +
        (revision_freq * 1.5)
    )
    # Add noise & clamp target score between 0 and 100
    noise = np.random.normal(0, 3.5, samples)
    predicted_score = np.clip(base_score + noise, 35.0, 100.0)

    df = pd.DataFrame({
        'Study_Hours_Per_Day': np.round(study_hours, 1),
        'Sleep_Hours_Per_Night': np.round(sleep_hours, 1),
        'Attendance_Percentage': np.round(attendance, 1),
        'Past_Exam_Score': np.round(past_score, 1),
        'Distraction_Level': np.round(distraction_level, 1),
        'Revision_Days_Per_Week': revision_freq,
        'Final_Exam_Score': np.round(predicted_score, 1)
    })
    return df

@st.cache_resource
def train_model(df):
    X = df.drop(columns=['Final_Exam_Score'])
    y = df['Final_Exam_Score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return model, r2, list(X.columns)

# Load data and train ML model
df_data = generate_synthetic_data()
model, model_r2, feature_names = train_model(df_data)

# --- Initialize Session State for Daily Logs ---
if 'logs' not in st.session_state:
    st.session_state['logs'] = pd.DataFrame([
        {'Date': (datetime.date.today() - datetime.timedelta(days=i)).strftime('%Y-%m-%d'), 
         'Hours Studied': np.random.randint(2, 7), 
         'Tasks Completed': np.random.randint(3, 8),
         'Focus Quality': np.random.choice(['High', 'Medium', 'Low'])}
        for i in range(5, -1, -1)
    ])

# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/illustrations/100/learning.png", width=80)
st.sidebar.title("Student Profile")
st.sidebar.markdown("Adjust your study parameters to evaluate expected performance.")

study_hours = st.sidebar.slider("Daily Study Hours", 0.5, 12.0, 5.0, 0.5)
sleep_hours = st.sidebar.slider("Nightly Sleep Hours", 3.0, 10.0, 7.5, 0.5)
attendance = st.sidebar.slider("Class Attendance (%)", 40, 100, 85, 1)
past_score = st.sidebar.slider("Past Exam Average (%)", 30, 100, 75, 1)
distraction_level = st.sidebar.slider("Distraction Level (1=Low, 10=High)", 1, 10, 4, 1)
revision_freq = st.sidebar.selectbox("Revision Days Per Week", [1, 2, 3, 4, 5, 6, 7], index=3)

# Build input row for model prediction
input_data = pd.DataFrame([[
    study_hours,
    sleep_hours,
    attendance,
    past_score,
    distraction_level,
    revision_freq
]], columns=feature_names)

# Make Prediction
predicted_score = model.predict(input_data)[0]
pass_probability = min(100.0, max(0.0, (predicted_score / 100.0) * 105))
study_efficiency = min(100.0, (study_hours / (distraction_level * 0.5 + 1.0)) * 20)

# --- Header Section ---
st.title("🎓 Smart Study Performance Predictor")
st.markdown("An AI-driven dashboard that analyzes study habits, forecasts exam performance, and provides tailored feedback.")

# --- Top Key Metrics ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Exam Score</div>
            <div class="metric-value">{predicted_score:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    status_color = "#16a34a" if pass_probability >= 70 else "#d97706" if pass_probability >= 50 else "#dc2626"
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pass Confidence</div>
            <div class="metric-value" style="color: {status_color};">{pass_probability:.0f}%</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Efficiency Index</div>
            <div class="metric-value">{study_efficiency:.0f}/100</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy (R²)</div>
            <div class="metric-value" style="color: #475569;">{model_r2*100:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Tabs Navigation ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predictions & Insights", 
    "🧪 What-If Simulator", 
    "📅 Daily Habit Tracker", 
    "📊 Model Analytics"
])

# --- TAB 1: Predictions & AI Recommendations ---
with tab1:
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.subheader("Performance Breakdown")
        
        # Gauge chart for predicted score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Final Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2563eb"},
                'steps': [
                    {'range': [0, 50], 'color': "#fee2e2"},
                    {'range': [50, 75], 'color': "#fef3c7"},
                    {'range': [75, 100], 'color': "#dcfce7"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_b:
        st.subheader("💡 Personalized Recommendations")
        
        recommendations = []
        if sleep_hours < 7.0:
            recommendations.append("😴 **Sleep Deficit**: Sleeping under 7 hours reduces cognitive retention. Increasing sleep to 7.5 hours could boost test scores by ~3-5%.")
        if distraction_level > 5:
            recommendations.append("📱 **High Distraction Rate**: Minimize phone usage during study blocks. Using techniques like Pomodoro can significantly improve your score.")
        if revision_freq < 3:
            recommendations.append("🔁 **Low Revision Frequency**: Spaced repetition builds long-term memory. Try adding 1-2 more revision sessions each week.")
        if study_hours < 3.0:
            recommendations.append("⏱️ **Study Volume**: Increasing active study time by just 1 hour daily can produce substantial grade improvements.")
            
        if not recommendations:
            recommendations.append("🌟 **Great Balance!**: Your study routines and habits are optimized. Keep up the consistent work!")
            
        for rec in recommendations:
            st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)

# --- TAB 2: What-If Simulator ---
with tab2:
    st.subheader("🧪 'What-If' Parameter Sensitivity Simulator")
    st.write("See how tweaking individual study factors impacts your predicted score in real-time.")

    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        st.markdown("##### Study Hours Sensitivity")
        hours_range = np.linspace(1, 12, 20)
        temp_input = input_data.copy()
        scores_by_hours = []
        
        for h in hours_range:
            temp_input['Study_Hours_Per_Day'] = h
            scores_by_hours.append(model.predict(temp_input)[0])
            
        fig_hours = px.line(x=hours_range, y=scores_by_hours, labels={'x': 'Daily Study Hours', 'y': 'Predicted Score'}, title="Impact of Study Hours")
        fig_hours.add_scatter(x=[study_hours], y=[predicted_score], mode='markers', marker=dict(color='red', size=12), name='Current State')
        st.plotly_chart(fig_hours, use_container_width=True)

    with sim_col2:
        st.markdown("##### Distraction Level Sensitivity")
        distraction_range = np.linspace(1, 10, 20)
        scores_by_dist = []
        
        for d in distraction_range:
            temp_input['Distraction_Level'] = d
            scores_by_dist.append(model.predict(temp_input)[0])
            
        fig_dist = px.line(x=distraction_range, y=scores_by_dist, labels={'x': 'Distraction Level', 'y': 'Predicted Score'}, title="Impact of Distractions")
        fig_dist.add_scatter(x=[distraction_level], y=[predicted_score], mode='markers', marker=dict(color='red', size=12), name='Current State')
        st.plotly_chart(fig_dist, use_container_width=True)

# --- TAB 3: Daily Habit Tracker ---
with tab3:
    st.subheader("📅 Daily Study Habit Log")
    
    # Input Form to Add New Entry
    with st.expander("➕ Log Today's Study Performance", expanded=False):
        with st.form("log_form"):
            log_date = st.date_input("Date", datetime.date.today())
            log_hours = st.number_input("Hours Studied", min_value=0.0, max_value=16.0, value=4.0, step=0.5)
            log_tasks = st.number_input("Tasks Completed", min_value=0, max_value=20, value=5)
            log_quality = st.selectbox("Focus Quality", ["High", "Medium", "Low"])
            
            submit_btn = st.form_submit_button("Save Log Entry")
            if submit_btn:
                new_entry = pd.DataFrame([{
                    'Date': log_date.strftime('%Y-%m-%d'),
                    'Hours Studied': log_hours,
                    'Tasks Completed': log_tasks,
                    'Focus Quality': log_quality
                }])
                st.session_state['logs'] = pd.concat([st.session_state['logs'], new_entry], ignore_index=True)
                st.success("Log added successfully!")

    # Display Logs & Trends
    log_df = st.session_state['logs']
    
    col_log1, col_log2 = st.columns([1, 1.2])
    
    with col_log1:
        st.markdown("##### Recent Activity Logs")
        st.dataframe(log_df.sort_values(by='Date', ascending=False), use_container_width=True, height=250)
        
    with col_log2:
        st.markdown("##### Study Hours Trend")
        fig_trend = px.bar(log_df, x='Date', y='Hours Studied', color='Focus Quality',
                           color_discrete_map={'High': '#22c55e', 'Medium': '#f59e0b', 'Low': '#ef4444'},
                           title="Daily Study Hours by Focus Quality")
        fig_trend.update_layout(height=280)
        st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 4: Model Analytics ---
with tab4:
    st.subheader("📊 Machine Learning Feature Importance")
    st.write("Which factors contribute most to student academic outcome according to the trained model?")
    
    # Feature Importances
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': [f.replace('_', ' ') for f in feature_names],
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    fig_importance = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                            title="Feature Weights in Score Prediction",
                            color='Importance', color_continuous_scale='Viridis')
    st.plotly_chart(fig_importance, use_container_width=True)

    with st.expander("🔍 View Synthetic Training Dataset"):
        st.dataframe(df_data.head(100), use_container_width=True)