import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="EuroBank Churn Analytics Hub",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injected custom CSS for a beautiful, modern, sleek dashboard appearance
st.markdown("""
<style>
    /* Executive Color Scheme & Clean Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Background & Main Content Padding */
    .stApp {
        background-color: #f8fafc !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        color: #94a3b8 !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] label {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #f1f5f9;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05), 0 1px 2px -1px rgb(0 0 0 / 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.05), 0 4px 6px -4px rgb(0 0 0 / 0.05);
    }
    .metric-title {
        font-size: 0.875rem;
        font-weight: 500;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.25rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .metric-delta {
        font-size: 0.8125rem;
        font-weight: 600;
        margin-top: 0.5rem;
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
    }
    .delta-up {
        color: #047857;
        background-color: #d1fae5;
    }
    .delta-down {
        color: #b91c1c;
        background-color: #fee2e2;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px 8px 0px 0px;
        color: #64748b;
        font-weight: 500;
        padding: 0px 16px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #2563eb !important;
    }

    /* Quick Insights Box Styling */
    .insight-box {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .insight-title {
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.35rem;
        font-size: 1rem;
    }
    .insight-desc {
        color: #2563eb;
        font-size: 0.95rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA INGESTION & PIPELINE PREPROCESSING
# ==========================================
@st.cache_data
def get_default_data():
    """Generates elegant, realistic fallback/demo data in case the csv is not found."""
    np.random.seed(42)
    n_samples = 1000
    
    customer_ids = np.arange(15600000, 15600000 + n_samples)
    surnames = ["Smith", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor"] * 100
    credit_scores = np.random.normal(650, 95, n_samples).astype(int)
    credit_scores = np.clip(credit_scores, 350, 850)
    
    geographies = np.random.choice(["France", "Germany", "Spain"], size=n_samples, p=[0.50, 0.25, 0.25])
    genders = np.random.choice(["Female", "Male"], size=n_samples, p=[0.45, 0.55])
    ages = np.random.normal(38.9, 10.5, n_samples).astype(int)
    ages = np.clip(ages, 18, 92)
    
    tenures = np.random.randint(0, 11, size=n_samples)
    balances = np.random.choice([0.0, 85000.0, 120000.0, 155000.0], size=n_samples, p=[0.35, 0.15, 0.30, 0.20])
    balances = np.where(balances > 0, balances + np.random.normal(0, 15000, n_samples), 0.0)
    balances = np.clip(balances, 0.0, None)
    
    num_products = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.51, 0.45, 0.03, 0.01])
    has_cr_card = np.random.choice([1, 0], size=n_samples, p=[0.70, 0.30])
    is_active_member = np.random.choice([1, 0], size=n_samples, p=[0.51, 0.49])
    estimated_salaries = np.random.uniform(15000, 200000, n_samples)
    
    # Logic for exited (higher churn for Germany, higher age 46-60, inactive members, and balance/products interactions)
    exited = []
    for i in range(n_samples):
        prob = 0.05
        if geographies[i] == "Germany": prob += 0.15
        if 46 <= ages[i] <= 60: prob += 0.35
        elif ages[i] > 60: prob += 0.15
        if is_active_member[i] == 0: prob += 0.15
        if num_products[i] >= 3: prob += 0.40
        if balances[i] > 100000: prob += 0.10
        
        prob = np.clip(prob, 0.02, 0.98)
        exited.append(np.random.choice([1, 0], p=[prob, 1 - prob]))
        
    df = pd.DataFrame({
        "CustomerId": customer_ids,
        "Surname": surnames,
        "CreditScore": credit_scores,
        "Geography": geographies,
        "Gender": genders,
        "Age": ages,
        "Tenure": tenures,
        "Balance": balances,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salaries,
        "Exited": exited
    })
    return df

def clean_dataframe_columns(df):
    """Clean up byte order marks (BOM) and trailing spaces to prevent KeyErrors."""
    if df is not None:
        # Clean BOM and whitespace from column headers
        df.columns = [str(col).replace('\ufeff', '').strip() for col in df.columns]
        
        # Build robust case-insensitive map to standard keys
        rename_dict = {}
        for col in df.columns:
            col_clean = col.lower().replace('_', '').replace(' ', '')
            if col_clean == 'customerid': rename_dict[col] = 'CustomerId'
            elif col_clean == 'surname': rename_dict[col] = 'Surname'
            elif col_clean == 'creditscore': rename_dict[col] = 'CreditScore'
            elif col_clean == 'geography': rename_dict[col] = 'Geography'
            elif col_clean == 'gender': rename_dict[col] = 'Gender'
            elif col_clean == 'age': rename_dict[col] = 'Age'
            elif col_clean == 'tenure': rename_dict[col] = 'Tenure'
            elif col_clean == 'balance': rename_dict[col] = 'Balance'
            elif col_clean == 'numofproducts': rename_dict[col] = 'NumOfProducts'
            elif col_clean == 'hascrcard': rename_dict[col] = 'HasCrCard'
            elif col_clean == 'isactivemember': rename_dict[col] = 'IsActiveMember'
            elif col_clean == 'estimatedsalary': rename_dict[col] = 'EstimatedSalary'
            elif col_clean == 'exited': rename_dict[col] = 'Exited'
        
        df = df.rename(columns=rename_dict)
    return df

@st.cache_data
def process_data(df):
    """Creates derived segment columns for visualizations."""
    if df is None:
        return None
    
    # Create copy to avoid mutating cache
    df = df.copy()
    
    # 1. Geographic Segment
    df['GeographicSegment'] = df['Geography']
    
    # 2. Age Segment (<30, 30–45, 46–60, 60+)
    age_bins = [0, 30, 46, 61, np.inf]
    age_labels = ['<30', '30–45', '46–60', '60+']
    df['AgeSegment'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    
    # 3. Credit Score Bands (Low, Medium, High)
    cs_bins = [0, 550, 701, np.inf]
    cs_labels = ['Low (<550)', 'Medium (550-700)', 'High (700+)']
    df['CreditScoreBand'] = pd.cut(df['CreditScore'], bins=cs_bins, labels=cs_labels, right=False)
    
    # 4. Tenure Groups (New, Mid-term, Long-term)
    tenure_bins = [0, 3, 8, np.inf]
    tenure_labels = ['New (0-2 yr)', 'Mid-term (3-7 yr)', 'Long-term (8+ yr)']
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=tenure_bins, labels=tenure_labels, right=False)
    
    # 5. Balance Segments (Zero-balance, Low-balance, High-balance)
    bal_bins = [-np.inf, 1, 100000, np.inf]
    bal_labels = ['Zero Balance', 'Low Balance (<$100k)', 'High Balance (>= $100k)']
    df['BalanceSegment'] = pd.cut(df['Balance'], bins=bal_bins, labels=bal_labels, right=False)
    
    # Labeling Binary flags nicely for dashboard filters
    df['ActiveStatus'] = df['IsActiveMember'].map({1: 'Active', 0: 'Inactive'})
    df['CreditCardStatus'] = df['HasCrCard'].map({1: 'Has Credit Card', 0: 'No Credit Card'})
    df['ChurnStatus'] = df['Exited'].map({1: 'Churned', 0: 'Retained'})
    
    return df

# Initialize data container
df_raw = None

# Sidebar file uploader to allow seamless custom CSV analysis
st.sidebar.subheader("📥 Upload Custom Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload Churn CSV File (Optional)",
    type=["csv"],
    help="Upload your custom European bank customer churn dataset to populate the dashboard dynamically."
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        df_raw = clean_dataframe_columns(df_raw)
        st.sidebar.success("✅ Custom Dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error parsing uploaded CSV: {e}. Falling back to default.")
        df_raw = None

# If no file uploaded, look for local csv file
if df_raw is None:
    if os.path.exists("churn_data.csv"):
        try:
            df_raw = pd.read_csv("churn_data.csv")
            df_raw = clean_dataframe_columns(df_raw)
        except Exception:
            df_raw = clean_dataframe_columns(get_default_data())
    else:
        df_raw = clean_dataframe_columns(get_default_data())

df_clean = process_data(df_raw)

# ==========================================
# 3. SIDEBAR NAVIGATION & DYNAMIC FILTERS
# ==========================================
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; padding: 0.5rem 0.25rem;">
    <div style="width: 36px; height: 36px; background-color: #2563eb; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 800; color: #ffffff; font-size: 1.25rem; box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);">B</div>
    <div style="display: flex; flex-direction: column;">
        <span style="font-weight: 800; font-size: 1.15rem; letter-spacing: -0.02em; color: #ffffff; line-height: 1.2;">EuroBank AI</span>
        <span style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Churn Intelligence</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div style="font-weight: 700; font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;">📊 Filter Segments</div>', unsafe_allow_html=True)

if df_clean is not None:
    # 1. Geographic Segment Filter
    all_geographies = sorted(df_clean['Geography'].unique().tolist())
    selected_geo = st.sidebar.multiselect(
        "Country / Geography",
        options=all_geographies,
        default=all_geographies,
        key="sb_geo"
    )
    
    # 2. Gender Filter
    all_genders = sorted(df_clean['Gender'].unique().tolist())
    selected_gender = st.sidebar.multiselect(
        "Gender Selection",
        options=all_genders,
        default=all_genders,
        key="sb_gender"
    )
    
    # 3. Age Segment Filter
    all_ages = sorted(df_clean['AgeSegment'].unique().tolist())
    selected_age = st.sidebar.multiselect(
        "Age Segments",
        options=all_ages,
        default=all_ages,
        key="sb_age"
    )

    # 4. Active Status Filter
    all_actives = sorted(df_clean['ActiveStatus'].unique().tolist())
    selected_active = st.sidebar.multiselect(
        "Engagement Status (Active/Inactive)",
        options=all_actives,
        default=all_actives,
        key="sb_active"
    )
    
    # 5. Products Offering Filter (1, 2, 3, 4, etc.)
    all_products = sorted(df_clean['NumOfProducts'].unique().tolist())
    selected_products = st.sidebar.multiselect(
        "Product Holdings (Num Of Products)",
        options=all_products,
        default=all_products,
        key="sb_prod"
    )
    
    # 6. Credit Card Ownership Filter
    all_cr_status = sorted(df_clean['CreditCardStatus'].unique().tolist())
    selected_cr_status = st.sidebar.multiselect(
        "Credit Card ownership",
        options=all_cr_status,
        default=all_cr_status,
        key="sb_crcard"
    )

    # 7. Balance Filters
    st.sidebar.subheader("💰 Financial Balance Segmentation")
    balance_choice = st.sidebar.radio(
        "Account Balance Focus",
        options=["All Balances", "Below €100,000", "Above €100,000"],
        index=0,
        key="sb_bal"
    )
    
    # 8. Numeric Credit Score Slider
    st.sidebar.subheader("💳 Creditworthiness Bounds")
    min_score, max_score = int(df_clean['CreditScore'].min()), int(df_clean['CreditScore'].max())
    selected_cs = st.sidebar.slider(
        "Credit Score Range",
        min_value=min_score,
        max_value=max_score,
        value=(min_score, max_score),
        key="sb_cs"
    )
    
    # Analyst profile card at bottom
    st.sidebar.markdown("""
    <div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #1e293b; display: flex; align-items: center; gap: 12px;">
        <div style="width: 40px; height: 40px; border-radius: 50%; background-color: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.4); color: #60a5fa; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.95rem; text-align: center;">AK</div>
        <div>
            <p style="font-size: 0.875rem; font-weight: 600; color: #ffffff; margin: 0;">Aksh Kumar Jha</p>
            <p style="font-size: 0.75rem; color: #64748b; margin: 0;">Lead Analyst</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Apply filters systematically
    filtered_df = df_clean[
        (df_clean['Geography'].isin(selected_geo)) &
        (df_clean['Gender'].isin(selected_gender)) &
        (df_clean['AgeSegment'].isin(selected_age)) &
        (df_clean['ActiveStatus'].isin(selected_active)) &
        (df_clean['NumOfProducts'].isin(selected_products)) &
        (df_clean['CreditCardStatus'].isin(selected_cr_status)) &
        (df_clean['CreditScore'].between(selected_cs[0], selected_cs[1]))
    ]
    
    # Apply Balance radio filters
    if balance_choice == "Below €100,000":
        filtered_df = filtered_df[filtered_df['Balance'] < 100000]
    elif balance_choice == "Above €100,000":
        filtered_df = filtered_df[filtered_df['Balance'] >= 100000]
        
    premium_balance_threshold = 100000.0
else:
    filtered_df = pd.DataFrame()

# ==========================================
# 4. EXECUTIVE METRICS & KPI ENGINE
# ==========================================
def render_metric_card(title, value, delta_val=None, is_positive=False, suffix=""):
    delta_class = "delta-up" if is_positive else "delta-down"
    delta_symbol = "▲" if is_positive else "▼"
    
    delta_html = ""
    if delta_val is not None:
        delta_html = f'<div class="metric-delta {delta_class}">{delta_symbol} {delta_val}</div>'
        
    card_html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}{suffix}</div>
        {delta_html}
    </div>
    """
    return st.markdown(card_html, unsafe_allow_html=True)

# Main Title & Subtitle
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #e2e8f0; padding-bottom: 1.25rem; margin-bottom: 2rem; margin-top: 1rem;">
  <div style="display: flex; flex-direction: column;">
    <span style="font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Projects / EuroBank Churn Hub</span>
    <h1 style="font-size: 2.25rem; font-weight: 800; letter-spacing: -0.03em; color: #0f172a; margin: 0.2rem 0 0 0; line-height: 1.1;">Churn Analytics Dashboard</h1>
  </div>
  <div style="display: flex; align-items: center; gap: 12px;">
    <span style="font-size: 0.75rem; background-color: #d1fae5; color: #065f46; font-weight: 700; padding: 0.35rem 0.6rem; border-radius: 6px; letter-spacing: 0.05em; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">LIVE DATA</span>
  </div>
</div>
""", unsafe_allow_html=True)

if filtered_df is not None and not filtered_df.empty:
    
    # Dynamic KPI Calculations
    total_customers = len(filtered_df)
    churn_count = len(filtered_df[filtered_df['Exited'] == 1])
    overall_churn_rate = (churn_count / total_customers * 100) if total_customers > 0 else 0.0
    
    # Premium / High-Value Customers (Balances >= €100k)
    high_value_df = filtered_df[filtered_df['Balance'] >= premium_balance_threshold]
    total_hv = len(high_value_df)
    churn_hv = len(high_value_df[high_value_df['Exited'] == 1])
    hv_churn_rate = (churn_hv / total_hv * 100) if total_hv > 0 else 0.0
    
    # Highest Regional Risk Exposure
    regional_shares = filtered_df.groupby('Geography')['Exited'].mean() * 100
    highest_risk_region = "N/A"
    highest_risk_rate = 0.0
    if not regional_shares.empty:
        highest_risk_region = regional_shares.idxmax()
        highest_risk_rate = regional_shares.max()
        
    # Active/Inactive Engagement Risk Gap
    churn_active = filtered_df[filtered_df['IsActiveMember'] == 1]['Exited'].mean() * 100
    churn_inactive = filtered_df[filtered_df['IsActiveMember'] == 0]['Exited'].mean() * 100
    engagement_gap = churn_inactive - churn_active
    
    # Display top metric KPIs
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        render_metric_card(
            title="Overall Churn Rate",
            value=f"{overall_churn_rate:.2f}",
            delta_val="Baseline standard: 20.37%" if len(filtered_df) == len(df_clean) else f"Filtered size: {total_customers}",
            is_positive=(overall_churn_rate < 20.37),
            suffix="%"
        )
    with kpi_col2:
        render_metric_card(
            title="High-Value Churn Ratio",
            value=f"{hv_churn_rate:.1f}",
            delta_val=f"Premium Cust: {total_hv}",
            is_positive=(hv_churn_rate < overall_churn_rate),
            suffix="%"
        )
    with kpi_col3:
        render_metric_card(
            title="Highest Risk Region",
            value=highest_risk_region,
            delta_val=f"Churn rate: {highest_risk_rate:.1f}%",
            is_positive=False
        )
    with kpi_col4:
        render_metric_card(
            title="Engagement Risk Gap",
            value=f"{engagement_gap:.1f}",
            delta_val="Inactive vs Active Churn",
            is_positive=(engagement_gap < 10.0),
            suffix=" pp"
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==========================================
    # 5. CORE TAB MODULE NAVIGATION
    # ==========================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Executive Summary",
        "🌍 Geographic & Demographic Analytics",
        "💳 Financial Profiling & Segments",
        "💎 Premium Cust Explorer",
        "🔮 AI Churn Risk Predictor"
    ])
    
    # ------------------------------------------
    # TAB 1: EXECUTIVE SUMMARY
    # ------------------------------------------
    with tab1:
        st.subheader("Executive Intelligence Summary")
        
        # Grid of insights and visual summaries
        col_summary_1, col_summary_2 = st.columns([2, 3])
        
        with col_summary_1:
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">🔑 Critical Strategic Insight</div>
                <div class="insight-desc">
                    Overall Churn rate at <b>{overall_churn_rate:.2f}%</b> represents a major threat to customer lifetime value (LTV).
                    The Geographic risk remains heavily concentrated in <b>Germany</b>, where churn rates exceed 32%, nearly double the rates of France and Spain.
                    Active product development and tailored marketing must target this exposure.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge charts for Churn Rate comparison
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = overall_churn_rate,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Customer Churn Status Gauge", 'font': {'size': 18}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#f43f5e"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 15], 'color': '#d1fae5'},
                        {'range': [15, 25], 'color': '#fef3c7'},
                        {'range': [25, 100], 'color': '#ffe4e6'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 20.37}
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=30, b=0, l=10, r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_summary_2:
            # Main visual representation of the active segments
            df_counts = filtered_df['ChurnStatus'].value_counts().reset_index()
            fig_pie = px.pie(
                df_counts,
                values='count',
                names='ChurnStatus',
                title="Proportion of Retained vs Churned Customers",
                color='ChurnStatus',
                color_discrete_map={'Retained': '#10b981', 'Churned': '#f43f5e'},
                hole=0.4
            )
            fig_pie.update_layout(height=360, margin=dict(t=50, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # High-Risk Identification Model recommendation banner matching Sleek Interface
        st.markdown("""
        <div style="background-color: #1e293b; padding: 1.5rem; border-radius: 16px; border: 1px solid #334155; color: #ffffff; display: flex; align-items: center; justify-content: space-between; margin-top: 1rem; margin-bottom: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
          <div style="display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;">
            <div style="padding: 0.75rem; background-color: rgba(59, 130, 246, 0.2); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.3); display: flex; align-items: center; justify-content: center; width: 48px; height: 48px; box-sizing: border-box;">
              <svg style="width: 1.75rem; height: 1.75rem; color: #60a5fa;" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>
            </div>
            <div style="display: flex; flex-direction: column;">
              <h4 style="font-size: 1.125rem; font-weight: 700; margin: 0; color: #ffffff; line-height: 1.3;">High-Risk Identification Model</h4>
              <p style="color: #94a3b8; font-size: 0.875rem; margin: 0.25rem 0 0 0; line-height: 1.4;">Decision support system identifies 46-60 year old high-balance members in Germany as #1 retention priority.</p>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualizing Churn across multiple main dimensions directly
        st.subheader("Key Demographic Segments Breakdown")
        col_seg1, col_seg2 = st.columns(2)
        
        with col_seg1:
            # Active status vs Churn Status
            df_act_churn = filtered_df.groupby(['ActiveStatus', 'ChurnStatus']).size().reset_index(name='Count')
            fig_act_churn = px.bar(
                df_act_churn,
                x='ActiveStatus',
                y='Count',
                color='ChurnStatus',
                title="Customer Engagement (Active Status) Impact on Churn Status",
                color_discrete_map={'Retained': '#10b981', 'Churned': '#f43f5e'},
                barmode='group'
            )
            st.plotly_chart(fig_act_churn, use_container_width=True)
            
        with col_seg2:
            # Credit score distribution with Churn
            fig_score_dist = px.histogram(
                filtered_df,
                x='CreditScore',
                color='ChurnStatus',
                title='Credit Score Distribution by Churn Status',
                color_discrete_map={'Retained': '#10b981', 'Churned': '#f43f5e'},
                opacity=0.8,
                barmode='overlay'
            )
            st.plotly_chart(fig_score_dist, use_container_width=True)
            
    # ------------------------------------------
    # TAB 2: GEOGRAPHIC & DEMOGRAPHIC ANALYTICS
    # ------------------------------------------
    with tab2:
        st.subheader("Geographic and Demographic Insights")
        
        col_geo_1, col_geo_2 = st.columns(2)
        
        with col_geo_1:
            # Geographic wise churn rate
            geo_churn = filtered_df.groupby('Geography')['Exited'].mean().reset_index()
            geo_churn['Churn Rate (%)'] = geo_churn['Exited'] * 100
            
            fig_geo_churn = px.bar(
                geo_churn,
                x='Geography',
                y='Churn Rate (%)',
                title="Churn Rate Across European Regions",
                labels={'Geography': 'Region'},
                color='Geography',
                color_discrete_sequence=['#4f46e5', '#3b82f6', '#10b981']
            )
            fig_geo_churn.add_hline(y=20.37, line_dash="dot", annotation_text="Global Baseline Average (20.37%)", annotation_position="top left", line_color="red")
            fig_geo_churn.update_layout(yaxis_range=[0, 45])
            st.plotly_chart(fig_geo_churn, use_container_width=True)
            
        with col_geo_2:
            # Gender breakdown Churn differences
            gender_churn = filtered_df.groupby('Gender')['Exited'].mean().reset_index()
            gender_churn['Churn Rate (%)'] = gender_churn['Exited'] * 100
            
            fig_gender_churn = px.bar(
                gender_churn,
                x='Gender',
                y='Churn Rate (%)',
                title="Churn Rate by Customer Gender",
                color='Gender',
                color_discrete_sequence=['#ec4899', '#3b82f6']
            )
            fig_gender_churn.update_layout(yaxis_range=[0, 35])
            st.plotly_chart(fig_gender_churn, use_container_width=True)
            
        st.markdown("---")
        
        st.subheader("Demographic Interactions & Age Dynamics")
        col_demo_1, col_demo_2 = st.columns(2)
        
        with col_demo_1:
            # Age group Churn rate
            age_churn = filtered_df.groupby('AgeSegment', observed=False)['Exited'].mean().reset_index()
            age_churn['Churn Rate (%)'] = age_churn['Exited'] * 100
            
            fig_age_churn = px.bar(
                age_churn,
                x='AgeSegment',
                y='Churn Rate (%)',
                title="Customer Age Segments vs Churn Rate",
                color_discrete_sequence=['#8b5cf6']
            )
            st.plotly_chart(fig_age_churn, use_container_width=True)
            
        with col_demo_2:
            # Geography and Age Interaction
            geo_age_churn = filtered_df.groupby(['Geography', 'AgeSegment'], observed=False)['Exited'].mean().reset_index()
            geo_age_churn['Churn Rate (%)'] = geo_age_churn['Exited'] * 100
            
            fig_geo_age = px.bar(
                geo_age_churn,
                x='AgeSegment',
                y='Churn Rate (%)',
                color='Geography',
                barmode='group',
                title="Inter-sectional Analysis: Region vs Age Churn Rates",
                color_discrete_sequence=['#4f46e5', '#3b82f6', '#10b981']
            )
            st.plotly_chart(fig_geo_age, use_container_width=True)

    # ------------------------------------------
    # TAB 3: FINANCIAL PROFILING & SEGMENTS
    # ------------------------------------------
    with tab3:
        st.subheader("Financial stability & Product portfolio correlation")
        
        col_fin_1, col_geo_2 = st.columns(2)
        
        with col_fin_1:
            # Churn rate by credit band
            cs_churn = filtered_df.groupby('CreditScoreBand', observed=False)['Exited'].mean().reset_index()
            cs_churn['Churn Rate (%)'] = cs_churn['Exited'] * 100
            
            fig_cs_churn = px.bar(
                cs_churn,
                x='CreditScoreBand',
                y='Churn Rate (%)',
                title="Churn Rate Across Credit Score Bands",
                color_discrete_sequence=['#06b6d4']
            )
            st.plotly_chart(fig_cs_churn, use_container_width=True)
            
        with col_geo_2:
            # Churn by Num of Products
            prod_churn = filtered_df.groupby('NumOfProducts')['Exited'].mean().reset_index()
            prod_churn['Churn Rate (%)'] = prod_churn['Exited'] * 100
            
            fig_prod_churn = px.line(
                prod_churn,
                x='NumOfProducts',
                y='Churn Rate (%)',
                title="Impact of Product Holdings on Customer Churn Rate",
                markers=True,
                line_shape='linear',
                color_discrete_sequence=['#f59e0b']
            )
            st.plotly_chart(fig_prod_churn, use_container_width=True)
            
        st.markdown("---")
        
        col_ten_1, col_ten_2 = st.columns(2)
        
        with col_ten_1:
            # Tenure Group churn rate
            tenure_churn = filtered_df.groupby('TenureGroup', observed=False)['Exited'].mean().reset_index()
            tenure_churn['Churn Rate (%)'] = tenure_churn['Exited'] * 100
            
            fig_ten_churn = px.bar(
                tenure_churn,
                x='TenureGroup',
                y='Churn Rate (%)',
                title="Customer Tenure Group vs Churn Rate",
                color_discrete_sequence=['#10b981']
            )
            st.plotly_chart(fig_ten_churn, use_container_width=True)
            
        with col_ten_2:
            # Balance segment churn rate
            bal_churn = filtered_df.groupby('BalanceSegment', observed=False)['Exited'].mean().reset_index()
            bal_churn['Churn Rate (%)'] = bal_churn['Exited'] * 100
            
            fig_bal_churn = px.bar(
                bal_churn,
                x='BalanceSegment',
                y='Churn Rate (%)',
                title="Financial Risk: Balance Segment vs Churn Rate",
                color_discrete_sequence=['#6366f1']
            )
            st.plotly_chart(fig_bal_churn, use_container_width=True)

    # ------------------------------------------
    # TAB 4: PREMIUM CUSTOMER EXPLORER
    # ------------------------------------------
    with tab4:
        st.subheader("High-Value Customer Churn Analysis")
        
        # High value filter box
        st.markdown(f"""
        <div class="insight-box" style="background-color: #f0fdf4; border-left-color: #10b981;">
            <div class="insight-title" style="color: #065f46;">💎 Premium Customer Segment Baseline</div>
            <div class="insight-desc" style="color: #047857;">
                High-Value customers are classified as those with account balances of <b>€100,000</b> or above. 
                These customers represent major financial liquidity and assets under management (AUM). Their loss poses significant revenue and capital risks to the banking institution.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_hv_1, col_hv_2 = st.columns([3, 2])
        
        with col_hv_1:
            # Scatter Plot Salary vs Balance for Premium Customers
            hv_filtered = filtered_df[filtered_df['Balance'] >= premium_balance_threshold]
            
            if not hv_filtered.empty:
                fig_scatter = px.scatter(
                    hv_filtered,
                    x='Balance',
                    y='EstimatedSalary',
                    color='ChurnStatus',
                    title="Financial Map of Premium Customers: Balance vs Estimated Salary",
                    color_discrete_map={'Retained': '#10b981', 'Churned': '#f43f5e'},
                    hover_data=['CustomerId', 'Age', 'Geography', 'NumOfProducts'],
                    size='Age',
                    size_max=15
                )
                fig_scatter.update_layout(xaxis_title="Account Balance (€)", yaxis_title="Estimated Annual Salary (€)")
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("⚠️ No High-Value Customer records match your active filtering parameters.")
                
        with col_hv_2:
            # Revenue Risk metric gauge
            total_premium_aum = filtered_df[filtered_df['Exited'] == 1]['Balance'].sum()
            st.markdown("### 💸 Revenue Assets Under Risk")
            st.metric(
                label="Total Lost Capital Assets (Churned Balance)",
                value=f"€{total_premium_aum:,.2f}",
                delta=f"Impact across {len(filtered_df[filtered_df['Exited'] == 1])} churned customers",
                delta_color="inverse"
            )
            
            # Premium Customers Churn across Credit Scores
            if not hv_filtered.empty:
                hv_cs_churn = hv_filtered.groupby('CreditScoreBand', observed=False)['Exited'].mean().reset_index()
                hv_cs_churn['Churn Rate (%)'] = hv_cs_churn['Exited'] * 100
                fig_hv_cs = px.bar(
                    hv_cs_churn,
                    x='CreditScoreBand',
                    y='Churn Rate (%)',
                    title="Premium Churn Across Credit Scores",
                    color_discrete_sequence=['#f43f5e']
                )
                st.plotly_chart(fig_hv_cs, use_container_width=True)

    # ------------------------------------------
    # TAB 5: AI CHURN RISK PREDICTOR
    # ------------------------------------------
    with tab5:
        st.subheader("🔮 Predictive Machine Learning Risk Engine")
        st.write("Train a Random Forest classifier in real-time on your filtered segment data to estimate individual churn risk profile probabilities.")
        
        ml_df = filtered_df.copy()
        
        if len(ml_df) > 50:
            
            # Encode Categorical Fields
            le_geo = LabelEncoder()
            le_gen = LabelEncoder()
            
            ml_df['Geography_Encoded'] = le_geo.fit_transform(ml_df['Geography'])
            ml_df['Gender_Encoded'] = le_gen.fit_transform(ml_df['Gender'])
            
            X_cols = ['CreditScore', 'Geography_Encoded', 'Gender_Encoded', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
            X = ml_df[X_cols]
            y = ml_df['Exited']
            
            # Train Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Fit Model
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            train_acc = rf_model.score(X_test, y_test) * 100
            
            st.success(f"⚡ Machine Learning Model successfully trained! Model Test Accuracy: **{train_acc:.2f}%**")
            
            # Interactive Interface for Input
            st.markdown("---")
            st.markdown("### Input Customer Profile Parameters")
            
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                input_geo = st.selectbox("Geography Region", options=le_geo.classes_)
                input_gen = st.selectbox("Gender", options=le_gen.classes_)
                input_age = st.slider("Customer Age", min_value=18, max_value=100, value=35)
                
            with pred_col2:
                input_score = st.slider("Creditworthiness Score", min_value=300, max_value=850, value=650)
                input_balance = st.number_input("Account Balance (€)", min_value=0.0, value=75000.0, step=1000.0)
                input_salary = st.number_input("Estimated Salary (€)", min_value=0.0, value=85000.0, step=1000.0)
                
            with pred_col3:
                input_tenure = st.slider("Tenure (Years)", min_value=0, max_value=15, value=5)
                input_products = st.selectbox("Number of Products", options=[1, 2, 3, 4], index=1)
                input_active = st.selectbox("Active Engagement status", options=["Active", "Inactive"], index=0)
                input_card = st.selectbox("Credit Card Ownership", options=["Has Credit Card", "No Credit Card"], index=0)
                
            # Formatting Input Vector
            active_flag = 1 if input_active == "Active" else 0
            card_flag = 1 if input_card == "Has Credit Card" else 0
            geo_enc = le_geo.transform([input_geo])[0]
            gen_enc = le_gen.transform([input_gen])[0]
            
            input_vector = pd.DataFrame([[
                input_score, geo_enc, gen_enc, input_age, input_tenure, input_balance, input_products, card_flag, active_flag, input_salary
            ]], columns=X_cols)
            
            # Predict Churn probability
            risk_probability = rf_model.predict_proba(input_vector)[0][1] * 100
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Churn Risk Assessment Result")
            
            if risk_probability < 30.0:
                risk_status = "Low Risk 🟢"
                risk_color = "green"
            elif risk_probability < 60.0:
                risk_status = "Moderate Risk 🟡"
                risk_color = "orange"
            else:
                risk_status = "High Risk 🔴"
                risk_color = "red"
                
            st.markdown(f"""
            <div style="background-color: #f8fafc; border: 2px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; text-align: center;">
                <div style="font-size: 1.1rem; color: #64748b; font-weight: 500;">Calculated Customer Churn Probability</div>
                <div style="font-size: 3.5rem; font-weight: 800; color: {risk_color}; margin: 0.5rem 0;">{risk_probability:.1f}%</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: {risk_color}; text-transform: uppercase;">{risk_status}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature Importance Visualizing
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Model Feature Importance Hierarchy")
            importances = rf_model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Risk Feature Factor': ['Credit Score', 'Region', 'Gender', 'Age', 'Tenure', 'Account Balance', 'Num Products', 'Credit Card Flag', 'Active Member Flag', 'Estimated Salary'],
                'Relative Importance Impact': importances
            }).sort_values(by='Relative Importance Impact', ascending=True)
            
            fig_imp = px.bar(
                feat_imp_df,
                x='Relative Importance Impact',
                y='Risk Feature Factor',
                orientation='h',
                title="Factors Influencing Predictive Churn Decision Scores",
                color='Relative Importance Impact',
                color_continuous_scale=px.colors.sequential.Agsunset
            )
            fig_imp.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
            
        else:
            st.warning("⚠️ Insufficient dynamic data volume for training. Select broader sidebar filters to compile at least 50 target customer entries.")
            
else:
    st.warning("⚠️ No active dataset matches found for current selections. Please adjust sidebar filter bounds.")
