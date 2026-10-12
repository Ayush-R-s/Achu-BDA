"""
Big Data Analytics (BDA) Platform - Global Socio-Demographic & Suicide Analytics
Author: A
Course: Big Data Analytics (BDA)
Features:
- Big Data Ingestion & MapReduce Simulation
- Complete Preprocessing & Feature Engineering Pipeline
- Exploratory Big Data Analytics (OLAP slicing & Plotly visualisations)
- Supervised ML: Multi-model Regression & Classification with Metrics
- Unsupervised ML: K-Means Clustering & PCA 2D Decomposition
- Real-Time Interactive What-If Scenario Predictor
- Formal Academic BDA Project Documentation & Report Generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import time
import io
import os

from src.data_loader import load_raw_data, simulate_mapreduce_processing
from src.preprocessing import clean_data, detect_outliers_iqr, engineer_features, prepare_ml_data
from src.ml_models import BdaMLSuite, RISK_CLASSES
from src.bda_analytics import generate_data_profile, compute_demographic_vulnerability_index, compute_olap_cube

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BDA Platform | Global Socio-Demographic Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN DESIGN STYLING ---
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 0;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.4rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.1);
        border-color: #38bdf8;
    }
    
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    
    .metric-sub {
        font-size: 0.8rem;
        color: #059669;
        font-weight: 600;
    }
    
    /* Badge styling */
    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        text-align: center;
    }
    
    .risk-low {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    
    .risk-moderate {
        background-color: #fef9c3;
        color: #a16207;
        border: 1px solid #fde047;
    }
    
    .risk-high {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }
    
    /* Code/Info Cards */
    .info-box {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 1.2rem;
        border-radius: 0 10px 10px 0;
        margin: 1.2rem 0;
    }
    
    .pipeline-step {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA PIPELINE ---
def get_processed_data():
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    featured_df = engineer_features(cleaned_df)
    ml_data = prepare_ml_data(featured_df)
    return raw_df, cleaned_df, featured_df, ml_data

def get_ml_suite(ml_data, test_size=0.25, random_state=42):
    suite = BdaMLSuite(ml_data, test_size=test_size, random_state=random_state)
    reg_results = suite.train_regression_models()
    clf_results = suite.train_classification_models()
    clustering_results = suite.train_clustering(k=3)
    return suite, reg_results, clf_results, clustering_results

# Load pipeline data
try:
    raw_df, cleaned_df, featured_df, ml_data = get_processed_data()
    ml_suite, reg_results, clf_results, clustering_results = get_ml_suite(ml_data)
except Exception as e:
    st.error(f"Error initializing data pipeline: {str(e)}")
    st.stop()

# --- SIDEBAR NAVIGATION & FILTERS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/data-configuration.png", width=64)
    st.markdown("### **BDA Project Control**")
    st.markdown("**Big Data Analytics & ML**")
    st.markdown("---")
    
    selected_page = option_menu(
        menu_title="Navigation",
        options=[
            "Overview & KPIs",
            "BDA Preprocessing",
            "Exploratory Analytics",
            "ML Supervised Models",
            "Unsupervised Clustering",
            "What-If Predictor",
            "BDA Documentation"
        ],
        icons=[
            "speedometer2",
            "gear-wide-connected",
            "bar-chart-line",
            "cpu",
            "diagram-3",
            "magic",
            "file-earmark-text"
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#0284c7", "font-size": "15px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "2px 0px",
                "--hover-color": "#f1f5f9",
            },
            "nav-link-selected": {"background-color": "#0284c7", "color": "white", "font-weight": "600"},
        }
    )
    
    st.markdown("---")
    st.markdown("#### **Dataset Scope**")
    st.markdown(f"- **File:** `master.csv`")
    st.markdown(f"- **Total Rows:** `{len(raw_df)} records`")
    st.markdown(f"- **Countries:** `{', '.join(raw_df['country'].unique())}`")
    st.markdown(f"- **Years Active:** `{raw_df['year'].min()} - {raw_df['year'].max()}`")
    
    st.markdown("---")
    st.caption("Big Data Analytics Course Project\nDepartment of Computer Science & Engineering")

# ==============================================================================
# 1. OVERVIEW & KPIS
# ==============================================================================
if selected_page == "Overview & KPIs":
    st.markdown("""
    <div class="main-header">
        <h1>Global Socio-Demographic & Suicide Analytics</h1>
        <p>Big Data Engineering, Exploratory Analytics, and Predictive Machine Learning Pipeline</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Metrics
    total_suicides = featured_df['suicides_no'].sum()
    total_population = featured_df['population'].sum()
    avg_rate = featured_df['suicides_per_100k'].mean()
    highest_rate_row = featured_df.loc[featured_df['suicides_per_100k'].idxmax()]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Suicides Recorded</div>
            <div class="metric-value">{total_suicides:,}</div>
            <div class="metric-sub">Across {featured_df['country'].nunique()} Major Economies</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Population Analyzed</div>
            <div class="metric-value">{total_population/1e6:.1f}M</div>
            <div class="metric-sub">Aggregated Demographic Cohorts</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Rate per 100k</div>
            <div class="metric-value">{avg_rate:.2f}</div>
            <div class="metric-sub">Global Benchmark Standard</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Peak Vulnerability Cohort</div>
            <div class="metric-value" style="font-size:1.35rem;">{highest_rate_row['country']}</div>
            <div class="metric-sub">{highest_rate_row['age']} | {highest_rate_row['sex'].capitalize()} ({highest_rate_row['suicides_per_100k']:.1f}/100k)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual Analytics Row
    chart_col1, chart_col2 = st.columns([3, 2])
    
    with chart_col1:
        st.subheader("📈 Country-Level Suicide Rates by Year")
        yearly_country = featured_df.groupby(['country', 'year'])['suicides_per_100k'].mean().reset_index()
        fig_trend = px.line(
            yearly_country,
            x='year',
            y='suicides_per_100k',
            color='country',
            markers=True,
            line_shape='spline',
            labels={'suicides_per_100k': 'Mean Suicides / 100k Pop', 'year': 'Year', 'country': 'Country'},
            title="Temporal Suicide Rate Trends (2012 - 2016)"
        )
        fig_trend.update_layout(template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with chart_col2:
        st.subheader("🚻 Gender Disparity Ratio")
        gender_agg = featured_df.groupby('sex')['suicides_no'].sum().reset_index()
        fig_gender = px.pie(
            gender_agg,
            values='suicides_no',
            names='sex',
            color='sex',
            color_discrete_map={'male': '#3b82f6', 'female': '#ec4899'},
            hole=0.55,
            title="Total Suicide Incidents by Gender"
        )
        fig_gender.update_layout(template="plotly_white", legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_gender, use_container_width=True)
        
    # Demographic Breakdown Matrix
    st.subheader("📊 Age Bracket Vulnerability Heatmap")
    age_country_pivot = featured_df.pivot_table(
        index='age',
        columns='country',
        values='suicides_per_100k',
        aggfunc='mean'
    ).round(2)
    
    fig_heat = px.imshow(
        age_country_pivot,
        labels=dict(x="Country", y="Age Bracket", color="Suicide Rate (per 100k)"),
        color_continuous_scale="Viridis",
        text_auto=True,
        aspect="auto",
        title="Cross-Demographic Heatmap: Age Group vs Country"
    )
    fig_heat.update_layout(template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

# ==============================================================================
# 2. BDA PREPROCESSING & PIPELINE ARCHITECTURE
# ==============================================================================
elif selected_page == "BDA Preprocessing":
    st.markdown("""
    <div class="main-header">
        <h1>Big Data Preprocessing & Pipeline Architecture</h1>
        <p>End-to-End Scalable ETL Pipeline, Outlier Detection, and Feature Transformation</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_arch, tab_profile, tab_cleaning, tab_outliers, tab_mapreduce = st.tabs([
        "🏗️ Architecture Pipeline",
        "📋 Data Profiling",
        "🧹 Cleaning & Transformation",
        "🎯 Outlier Diagnostics",
        "⚡ MapReduce Simulation"
    ])
    
    with tab_arch:
        st.markdown("### 🔄 End-to-End BDA Pipeline Workflow")
        
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin: 1rem 0 1.5rem 0;">
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #38bdf8; border-radius: 8px; padding: 1rem;">
                <div style="font-weight: 700; color: #0284c7; font-size: 0.85rem; text-transform: uppercase;">Step 1: Ingestion</div>
                <div style="font-weight: 600; color: #0f172a; margin-top: 0.2rem;">Raw Data (master.csv)</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.3rem;">Batch file loading & schema inspection</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #818cf8; border-radius: 8px; padding: 1rem;">
                <div style="font-weight: 700; color: #6366f1; font-size: 0.85rem; text-transform: uppercase;">Step 2: Cleaning</div>
                <div style="font-weight: 600; color: #0f172a; margin-top: 0.2rem;">Normalization & Imputation</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.3rem;">Standardize columns & median imputation</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #c084fc; border-radius: 8px; padding: 1rem;">
                <div style="font-weight: 700; color: #a855f7; font-size: 0.85rem; text-transform: uppercase;">Step 3: Diagnostics</div>
                <div style="font-weight: 600; color: #0f172a; margin-top: 0.2rem;">IQR Outlier Detection</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.3rem;">Flag demographic anomalies</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #f472b6; border-radius: 8px; padding: 1rem;">
                <div style="font-weight: 700; color: #ec4899; font-size: 0.85rem; text-transform: uppercase;">Step 4: Features</div>
                <div style="font-weight: 600; color: #0f172a; margin-top: 0.2rem;">Feature Engineering</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.3rem;">Rate per 100k & Risk Category tiers</div>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #10b981; border-radius: 8px; padding: 1rem;">
                <div style="font-weight: 700; color: #059669; font-size: 0.85rem; text-transform: uppercase;">Step 5: ML Models</div>
                <div style="font-weight: 600; color: #0f172a; margin-top: 0.2rem;">Encoding & Scaling</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.3rem;">OneHotEncoder + StandardScaler for ML</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ```mermaid
        flowchart LR
            A["Raw Ingestion: master.csv"] --> B["Data Profiling & Validation"]
            B --> C["Missing Imputation & Cleansing"]
            C --> D["IQR Outlier Detection"]
            D --> E["Feature Engineering & Risk Labelling"]
            E --> F["One-Hot & StandardScaler"]
            F --> G["ML Regression, Classification & Clustering"]
        ```
        """)
        st.markdown("""
        #### **Core Preprocessing Responsibilities:**
        1. **Schema Standardization:** Stripped special characters (`$`, `/`, parentheses), unified column naming convention to snake_case.
        2. **Type Safety & Cleansing:** Handled currency string conversions, parsed population & suicide count integers.
        3. **Missing Value Treatment:** Imputed missing socioeconomic indicators (HDI) with median distribution.
        4. **Feature Engineering:** Calculated normalized rates per 100k, generated categorical risk boundaries (`Low`, `Moderate`, `High`).
        5. **Scaling & Encoders:** Standardized continuous features using Z-score `StandardScaler` and applied `OneHotEncoder` for demographic categories.
        """)
        
    with tab_profile:
        st.subheader("📋 Dataset Health & Structural Profile")
        profile_df = generate_data_profile(raw_df)
        st.dataframe(profile_df, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(raw_df))
        c2.metric("Total Attributes", len(raw_df.columns))
        c3.metric("Memory Footprint", f"{raw_df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    with tab_cleaning:
        st.subheader("🔄 Before vs. After Preprocessing Comparison")
        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown("##### **Original Raw Schema & Samples**")
            st.dataframe(raw_df.head(5), use_container_width=True)
        with col_after:
            st.markdown("##### **Engineered & Standardized Features**")
            st.dataframe(featured_df[['country', 'year', 'sex', 'age', 'suicides_per_100k', 'risk_category']].head(5), use_container_width=True)

        st.markdown("##### **Feature Engineering Summary Table**")
        st.dataframe(featured_df[['country', 'year', 'sex', 'age', 'suicides_no', 'population', 'calculated_rate_100k', 'risk_category', 'risk_label']], use_container_width=True)

    with tab_outliers:
        st.subheader("🎯 Outlier Diagnostics (Interquartile Range Method)")
        col_iqr1, col_iqr2 = st.columns([2, 1])
        
        with col_iqr1:
            outliers_df, lower, upper, count = detect_outliers_iqr(featured_df, column='suicides_no')
            fig_box = px.box(
                featured_df,
                x='country',
                y='suicides_no',
                color='country',
                points="all",
                title="Suicide Volume Distribution with Outlier Boundaries"
            )
            fig_box.update_layout(template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)
            
        with col_iqr2:
            st.markdown("#### **IQR Boundary Thresholds**")
            st.metric("Lower Bound (Q1 - 1.5*IQR)", f"{lower:,.0f}")
            st.metric("Upper Bound (Q3 + 1.5*IQR)", f"{upper:,.0f}")
            st.metric("Detected Outlier Records", count)
            if count > 0:
                st.caption("Identified outliers represent high-population cohorts in the United States and India.")

    with tab_mapreduce:
        st.subheader("⚡ Big Data Distributed MapReduce Simulation")
        st.markdown("Demonstration of distributed chunking, parallel mapping, and group reduction across the dataset.")
        
        chunk_size = st.slider("Select Processing Chunk Size", min_value=5, max_value=15, value=10, step=5)
        
        if st.button("🚀 Execute MapReduce Job", type="primary"):
            with st.spinner("Processing distributed chunks..."):
                summary_mapreduce, metrics = simulate_mapreduce_processing(featured_df, chunk_size=chunk_size)
                time.sleep(0.3)
                
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Input Records", metrics['total_rows'])
            col_m2.metric("Distributed Chunks", metrics['num_chunks'])
            col_m3.metric("Execution Latency", f"{metrics['execution_time_ms']} ms")
            col_m4.metric("Reduced Partitions", metrics['unique_groups'])
            
            st.markdown("##### **Reduced Group Aggregations Output (Map -> Shuffle -> Reduce)**")
            st.dataframe(summary_mapreduce, use_container_width=True)

# ==============================================================================
# 3. EXPLORATORY ANALYTICS (EDA)
# ==============================================================================
elif selected_page == "Exploratory Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>Exploratory Big Data Analytics (EDA)</h1>
        <p>Interactive OLAP Cubes, Multi-Dimensional Slicing & Demographic Patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter Controls
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        sel_countries = st.multiselect("Select Countries", options=featured_df['country'].unique(), default=featured_df['country'].unique())
    with filter_col2:
        sel_genders = st.multiselect("Select Gender", options=featured_df['sex'].unique(), default=featured_df['sex'].unique())
    with filter_col3:
        sel_ages = st.multiselect("Select Age Brackets", options=featured_df['age'].unique(), default=featured_df['age'].unique())
        
    filtered_df = featured_df[
        (featured_df['country'].isin(sel_countries)) &
        (featured_df['sex'].isin(sel_genders)) &
        (featured_df['age'].isin(sel_ages))
    ]
    
    st.caption(f"Displaying **{len(filtered_df)}** records matching active criteria.")
    
    # Analytical Visuals
    c1, c2 = st.columns(2)
    with c1:
        fig_bar = px.bar(
            filtered_df,
            x='age',
            y='suicides_per_100k',
            color='country',
            barmode='group',
            labels={'suicides_per_100k': 'Suicide Rate per 100k', 'age': 'Age Bracket'},
            title="Age-Specific Suicide Rates Across Selected Nations"
        )
        fig_bar.update_layout(template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        fig_scatter = px.scatter(
            filtered_df,
            x='population',
            y='suicides_no',
            color='country',
            size='suicides_per_100k',
            hover_data=['year', 'sex', 'age'],
            labels={'population': 'Cohort Population', 'suicides_no': 'Suicide Count'},
            title="Population Scale vs Suicide Volume (Bubble Size = Rate/100k)"
        )
        fig_scatter.update_layout(template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    # Demographic Vulnerability Index Table
    st.subheader("🛡️ Demographic Vulnerability Index (DVI) Ranking")
    dvi_df = compute_demographic_vulnerability_index(filtered_df)
    st.dataframe(dvi_df, use_container_width=True)

# ==============================================================================
# 4. ML SUPERVISED MODELS
# ==============================================================================
elif selected_page == "ML Supervised Models":
    st.markdown("""
    <div class="main-header">
        <h1>Supervised Machine Learning Studio</h1>
        <p>Comparative Benchmarking: Regression for Rate Prediction & Multi-Class Risk Classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_reg, tab_clf = st.tabs(["📈 Supervised Regression Suite", "🎯 Supervised Classification Suite"])
    
    with tab_reg:
        st.subheader("📈 Predicting Suicide Rates per 100,000 Inhabitants")
        st.markdown("Benchmarking **Linear Regression**, **Decision Tree Regressor**, and **Random Forest Regressor**.")
        
        # Performance summary metrics table
        reg_summary = []
        for name, res in reg_results.items():
            reg_summary.append({
                'Algorithm': name,
                'R² Score (Variance Explained)': res['r2'],
                'RMSE (Root Mean Sq Error)': res['rmse'],
                'MAE (Mean Absolute Error)': res['mae']
            })
        reg_summary_df = pd.DataFrame(reg_summary).sort_values(by='R² Score (Variance Explained)', ascending=False)
        st.dataframe(reg_summary_df, use_container_width=True)
        
        # Regression Visualisation: Actual vs Predicted
        sel_reg_model = st.selectbox("Select Model for Visual Inspection", list(reg_results.keys()))
        selected_res = reg_results[sel_reg_model]
        
        comparison_df = pd.DataFrame({
            'Actual Rate': selected_res['y_test'],
            'Predicted Rate': selected_res['y_pred']
        })
        
        fig_reg = px.scatter(
            comparison_df,
            x='Actual Rate',
            y='Predicted Rate',
            title=f"Actual vs Predicted Suicide Rates ({sel_reg_model})",
            labels={'Actual Rate': 'Actual Rate / 100k', 'Predicted Rate': 'Predicted Rate / 100k'}
        )
        # Add diagonal ideal fit reference line
        min_val = min(comparison_df['Actual Rate'].min(), comparison_df['Predicted Rate'].min())
        max_val = max(comparison_df['Actual Rate'].max(), comparison_df['Predicted Rate'].max())
        fig_reg.add_shape(
            type="line",
            x0=min_val, y0=min_val,
            x1=max_val, y1=max_val,
            line=dict(color="#ef4444", dash="dash", width=2)
        )
        fig_reg.update_layout(template="plotly_white")
        st.plotly_chart(fig_reg, use_container_width=True)
        
    with tab_clf:
        st.subheader("🎯 Classifying Suicide Risk Tiers (Low / Moderate / High)")
        st.markdown("Benchmarking **Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier**.")
        
        clf_summary = []
        for name, res in clf_results.items():
            clf_summary.append({
                'Algorithm': name,
                'Accuracy (%)': f"{res['accuracy']}%",
                'Precision (Weighted)': res['precision'],
                'Recall (Weighted)': res['recall'],
                'F1-Score': res['f1_score']
            })
        clf_summary_df = pd.DataFrame(clf_summary)
        st.dataframe(clf_summary_df, use_container_width=True)
        
        # Confusion Matrix Heatmap
        sel_clf_model = st.selectbox("Inspect Confusion Matrix for Model", list(clf_results.keys()))
        cm = clf_results[sel_clf_model]['confusion_matrix']
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Class Index", y="True Class Index", color="Instances"),
            x=['Low', 'Moderate', 'High'][:cm.shape[1]],
            y=['Low', 'Moderate', 'High'][:cm.shape[0]],
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix: {sel_clf_model}"
        )
        fig_cm.update_layout(template="plotly_white")
        st.plotly_chart(fig_cm, use_container_width=True)

# ==============================================================================
# 5. UNSUPERVISED CLUSTERING
# ==============================================================================
elif selected_page == "Unsupervised Clustering":
    st.markdown("""
    <div class="main-header">
        <h1>Unsupervised Machine Learning: K-Means Risk Clustering</h1>
        <p>Demographic Segmentation and Vulnerability Pattern Discovery via Principal Component Analysis (PCA)</p>
    </div>
    """, unsafe_allow_html=True)
    
    k_val = st.slider("Select Number of Clusters (k)", min_value=2, max_value=4, value=3)
    cl_results = ml_suite.train_clustering(k=k_val)
    
    df_cl = cl_results['df_clustered']
    var_exp = cl_results['pca_variance_ratio']
    
    st.caption(f"Principal Component Analysis: PC1 explains **{var_exp[0]*100:.1f}%** variance, PC2 explains **{var_exp[1]*100:.1f}%**.")
    
    col_c1, col_c2 = st.columns([3, 2])
    with col_c1:
        fig_pca = px.scatter(
            df_cl,
            x='pca_x',
            y='pca_y',
            color=df_cl['cluster'].astype(str),
            hover_data=['country', 'sex', 'age', 'suicides_per_100k'],
            labels={'pca_x': 'Principal Component 1', 'pca_y': 'Principal Component 2', 'color': 'Cluster ID'},
            title="2D PCA Projection of Demographic & Socioeconomic Clusters"
        )
        fig_pca.update_layout(template="plotly_white")
        st.plotly_chart(fig_pca, use_container_width=True)
        
    with col_c2:
        st.markdown("#### **Cluster Vulnerability Profiles**")
        st.dataframe(cl_results['cluster_summary'], use_container_width=True)
        st.markdown("""
        **BDA Cluster Insights:**
        - **Cluster 0:** High-risk senior cohorts (75+ years) with elevated rates per 100k.
        - **Cluster 1:** Working-age male demographics with large aggregate volume.
        - **Cluster 2:** Low-to-moderate vulnerability youth & female demographics.
        """)

# ==============================================================================
# 6. WHAT-IF PREDICTOR
# ==============================================================================
elif selected_page == "What-If Predictor":
    st.markdown("""
    <div class="main-header">
        <h1>Real-Time What-If Scenario Predictor</h1>
        <p>Interactive Predictive Inference Engine for Demographic Risk Estimation</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Select demographic parameters below to compute real-time model inferences:")
    
    with st.form("prediction_form"):
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            in_country = st.selectbox("Target Country", options=featured_df['country'].unique())
            in_sex = st.selectbox("Biological Sex", options=['Male', 'Female'])
        with p_col2:
            in_age = st.selectbox("Age Bracket", options=featured_df['age'].unique())
            in_year = st.slider("Target Year", min_value=2012, max_value=2030, value=2017)
        with p_col3:
            in_pop = st.number_input("Cohort Population", min_value=100000, max_value=100000000, value=25000000, step=500000)
            
        submitted = st.form_submit_button("🔮 Generate ML Prediction", type="primary")
        
    if submitted:
        pred = ml_suite.predict_sample(
            country=in_country,
            sex=in_sex,
            age=in_age,
            year=in_year,
            population=in_pop
        )
        
        st.markdown("---")
        st.subheader("🎯 Machine Learning Inferences")
        
        res1, res2, res3 = st.columns(3)
        with res1:
            st.metric("Predicted Suicide Rate", f"{pred['predicted_rate_per_100k']} / 100k")
            st.caption(f"Engine: {pred['model_used_reg']}")
        with res2:
            st.metric("Estimated Case Volume", f"{pred['predicted_suicides_no']:,} cases")
            st.caption("Extrapolated across cohort population")
        with res3:
            badge_class = "risk-high" if pred['predicted_risk_level'] == 'High Risk' else ("risk-moderate" if pred['predicted_risk_level'] == 'Moderate Risk' else "risk-low")
            st.markdown(f"**Assessed Risk Category:**")
            st.markdown(f"<div class='risk-badge {badge_class}'>{pred['predicted_risk_level']}</div>", unsafe_allow_html=True)
            st.caption(f"Engine: {pred['model_used_clf']}")
            
        if pred['probabilities'] is not None:
            prob_df = pd.DataFrame({
                'Risk Class': RISK_CLASSES[:len(pred['probabilities'])],
                'Probability': pred['probabilities']
            })
            fig_prob = px.bar(
                prob_df,
                x='Risk Class',
                y='Probability',
                color='Risk Class',
                color_discrete_map={'Low Risk': '#10b981', 'Moderate Risk': '#f59e0b', 'High Risk': '#ef4444'},
                title="Classification Probability Distribution"
            )
            fig_prob.update_layout(template="plotly_white", yaxis_range=[0, 1])
            st.plotly_chart(fig_prob, use_container_width=True)

# ==============================================================================
# 7. BDA DOCUMENTATION & EXPORT
# ==============================================================================
elif selected_page == "BDA Documentation":
    st.markdown("""
    <div class="main-header">
        <h1>BDA Project Report & Documentation</h1>
        <p>Big Data Analytics Formal Specifications, Methodology, and Analytical Exports</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 1. Problem Statement
    Suicide is a complex socio-demographic crisis influenced by economic status, age vulnerability, and healthcare accessibility.
    The primary goal of this **Big Data Analytics (BDA)** project is to construct a scalable data preprocessing, exploratory OLAP, and predictive machine learning pipeline using multi-national demographic records (`master.csv`) to uncover risk disparities and forecast intervention priorities.

    ### 2. Big Data Lifecycle & Methodology
    - **Data Ingestion:** Batch and chunked distributed streaming simulation.
    - **Data Preprocessing & Cleaning:**
      - Normalization of text labels and numeric columns.
      - Median imputation for missing socioeconomic indices.
      - Interquartile Range (IQR) detection for extreme demographic anomalies.
    - **Feature Engineering:**
      - Derived rate verification: $\\text{Rate} = \\frac{\\text{Suicides}}{\\text{Population}} \\times 100,000$
      - Ordinal age mapping and binary gender encoding.
      - Stratified risk category labeling into *Low*, *Moderate*, and *High* vulnerability tiers.
    - **Machine Learning Suite:**
      - Supervised Regression: Linear Regression, Decision Tree, Random Forest.
      - Supervised Classification: Logistic Regression, Decision Tree, Random Forest.
      - Unsupervised Clustering: K-Means with PCA 2D dimensionality reduction.

    ### 3. Key Findings & Empirical Insights
    - **Senior Vulnerability:** Older adults (75+ years) exhibit the highest mortality rates per 100,000 population across all analyzed nations (reaching up to 100/100k in Japan).
    - **Gender Gap:** Males display substantially higher rates across all age brackets compared to female counterparts.
    - **Intervention Recommendations:** Targeted mental healthcare programs should be customized by demographic vulnerability segments.
    """)
    
    st.markdown("---")
    st.subheader("💾 Export Processed Artifacts")
    
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        csv_data = featured_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned & Engineered Dataset (CSV)",
            data=csv_data,
            file_name="bda_preprocessed_master.csv",
            mime="text/csv",
            type="primary"
        )
    with exp_col2:
        reg_summary = []
        for name, res in reg_results.items():
            reg_summary.append({'Algorithm': name, 'R2': res['r2'], 'RMSE': res['rmse'], 'MAE': res['mae']})
        report_csv = pd.DataFrame(reg_summary).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Model Benchmark Report (CSV)",
            data=report_csv,
            file_name="bda_model_benchmarks.csv",
            mime="text/csv"
        )
