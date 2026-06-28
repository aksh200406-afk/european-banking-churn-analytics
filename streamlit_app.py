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
    page_title="European Bank Churn Analytics Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for high-contrast executive appearance
st.markdown("""
<style>
    /* Executive Color Scheme & Clean Fonts */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background & Main Content Padding */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Elegant Title and Header styling */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
        letter-spacing: -0.05em;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 2rem;
    }
    
    /* Custom CSS styled metric cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    }
    .metric-title {
        font-size: 0.875rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1;
    }
    .metric-delta {
        font-size: 0.8125rem;
        font-weight: 500;
        margin-top: 0.5rem;
        display: flex;
        align-items: center;
    }
    .delta-up {
        color: #10b981;
    }
    .delta-down {
        color: #f43f5e;
    }
    
    /* Quick Insights Box Styling */
    .insight-box {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
    }
    .insight-title {
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 0.25rem;
    }
    .insight-desc {
        color: #2563eb;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA INGESTION & PIPELINE PREPROCESSING
# ==========================================
@st.cache_data
def load_and_prepare_data(file_path="churn_data.csv"):
    if not os.path.exists(file_path):
        st.error(f"❌ Error: Required dataset file '{file_path}' not found in root directory!")
        return None
    
    df = pd.read_csv(file_path)
    
    # 2.1 Remove Non-analytical fields
    if "Surname" in df.columns:
        df = df.drop(columns=["Surname"])
    if "Year" in df.columns:
        df = df.drop(columns=["Year"])
        
    # 2.2 Create Derived Segments as per specs
    
    # 1. Geographic Segments (France, Spain, Germany)
    df['GeographicSegment'] = df['Geography']
    
    # 2. Age Segments (<30, 30–45, 46–60, 60+)
    age_bins = [0, 30, 46, 61, np.inf]
    age_labels = ['<30', '30–45', '46–60', '60+']
    df['AgeSegment'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    
    # 3. Credit Score Bands (Low, Medium, High)
    cs_bins = [0, 550, 701, np.inf]
    cs_labels = ['Low (<550)', 'Medium (550-700)', 'High (700+)']
    df['CreditScoreBand'] = pd.cut(df['CreditScore'], bins=cs_bins, labels=cs_labels, right=False)
    
    # 4. Tenure Groups (New, Mid-term, Long-term)
    # New: 0-2 years, Mid-term: 3-7 years, Long-term: 8+ years
    tenure_bins = [0, 3, 8, np.inf]
    tenure_labels = ['New (0-2 yr)', 'Mid-term (3-7 yr)', 'Long-term (8+ yr)']
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=tenure_bins, labels=tenure_labels, right=False)
    
    # 5. Balance Segments (Zero-balance, Low-balance, High-balance)
    bal_bins = [-np.inf, 1, 100000, np.inf]
    bal_labels = ['Zero Balance', 'Low Balance (<$100k)', 'High Balance (>= $100k)']
    df['BalanceSegment'] = pd.cut(df['Balance'], bins=bal_bins, labels=bal_labels, right=False)
    
    # Labeling Binary flags nicely for dashboards
    df['ActiveStatus'] = df['IsActiveMember'].map({1: 'Active', 0: 'Inactive'})
    df['CreditCardStatus'] = df['HasCrCard'].map({1: 'Has Credit Card', 0: 'No Credit Card'})
    df['ChurnStatus'] = df['Exited'].map({1: 'Churned', 0: 'Retained'})
    
    return df

# Load the data
df_clean = load_and_prepare_data()

# ==========================================
# 3. SIDEBAR NAVIGATION & DYNAMIC FILTERS
# ==========================================
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h2 style="color: #4f46e5; font-size: 1.7rem; font-weight: 800; margin-bottom: 0.2rem;">Pyrrhia Bank</h2>
    <p style="color: #64748b; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em;">Churn Analytics Hub</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("📊 Filter Segments")

if df_clean is not None:
    # 1. Geographic Segment Filter
    all_geographies = df_clean['Geography'].unique().tolist()
    selected_geo = st.sidebar.multiselect(
        "Geographic Region",
        options=all_geographies,
        default=all_geographies
    )
    
    # 2. Gender Filter
    all_genders = df_clean['Gender'].unique().tolist()
    selected_gender = st.sidebar.multiselect(
        "Gender Selection",
        options=all_genders,
        default=all_genders
    )
    
    # 3. Age Segment Filter
    all_ages = df_clean['AgeSegment'].unique().tolist()
    selected_age = st.sidebar.multiselect(
        "Age Segments",
        options=all_ages,
        default=all_ages
    )

    # 4. Active Status Filter
    all_actives = df_clean['ActiveStatus'].unique().tolist()
    selected_active = st.sidebar.multiselect(
        "Engagement Status",
        options=all_actives,
        default=all_actives
    )

    # 5. Financial Profile Filter Sliders
    st.sidebar.subheader("💳 Financial Bounds")
    min_score, max_score = int(df_clean['CreditScore'].min()), int(df_clean['CreditScore'].max())
    selected_cs = st.sidebar.slider(
        "Credit Score Range",
        min_value=min_score,
        max_value=max_score,
        value=(min_score, max_score)
    )
    
    # Apply filters systematically
    filtered_df = df_clean[
        (df_clean['Geography'].isin(selected_geo)) &
        (df_clean['Gender'].isin(selected_gender)) &
        (df_clean['AgeSegment'].isin(selected_age)) &
        (df_clean['ActiveStatus'].isin(selected_active)) &
        (df_clean['CreditScore'].between(selected_cs[0], selected_cs[1]))
    ]
    
    # Calculate premium baseline for High-Value customer stats
    premium_balance_threshold = 100000.0  # standard high-balance criteria
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
st.markdown('<div class="main-title">Customer Segmentation & Churn Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">European Central Banking Group — Dynamic Executive Intelligence Dashboard</div>', unsafe_allow_html=True)

if filtered_df is not None and not filtered_df.empty:
    
    # Dynamic calculations
    total_customers = len(filtered_df)
    churn_count = len(filtered_df[filtered_df['Exited'] == 1])
    overall_churn_rate = (churn_count / total_customers * 100) if total_customers > 0 else 0.0
    
    # High Value Customer Churn Metrics (Balance >= $100k)
    high_value_df = filtered_df[filtered_df['Balance'] >= premium_balance_threshold]
    total_hv = len(high_value_df)
    churn_hv = len(high_value_df[high_value_df['Exited'] == 1])
    hv_churn_rate = (churn_hv / total_hv * 100) if total_hv > 0 else 0.0
    
    # Regional Churn stats
    regional_shares = filtered_df.groupby('Geography')['Exited'].mean() * 100
    highest_risk_region = "N/A"
    
    # render metric grids
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Customer Base", f"{total_customers:,}")
    with col2:
        # Churn rate benchmark is around 20%
        render_metric_card("Overall Churn Rate", f"{overall_churn_rate:.2f}", delta_val="Avg. Benchmark: 20.3%", is_positive=(overall_churn_rate < 20.3), suffix="%")
    with col3:
        # High Value Churn delta
        render_metric_card("High-Value Churn Ratio", f"{hv_churn_rate:.2f}", delta_val="Target: < 15.0%", is_positive=(hv_churn_rate < 15.0), suffix="%")
    with col4:
        # Engagement Gap (Active vs Inactive Churn delta)
        churn_active = filtered_df[filtered_df['IsActiveMember'] == 1]['Exited'].mean() * 100
        churn_inactive = filtered_df[filtered_df['IsActiveMember'] == 0]['Exited'].mean() * 100
        engagement_gap = churn_inactive - churn_active
        render_metric_card("Engagement Risk Gap", f"{engagement_gap:.1f}", delta_val="Inactive vs Active delta", is_positive=(engagement_gap < 12.0), suffix=" pp")
        
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
        
        col_summary_1, col_summary_2 = st.columns([2, 3])
        
        with col_summary_1:
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">🔑 Critical Strategic Insight</div>
                <div class="insight-desc">
                    Overall Churn rate at <b>{overall_churn_rate:.1f}%</b> represents a major threat to customer lifetime value (LTV).
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
                        'value': 20.4}
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
            
        st.markdown("---")
        
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
            fig_geo_churn.add_hline(y=20.4, line_dash="dot", annotation_text="Global Baseline Average (20.4%)", annotation_position="top left", line_color="red")
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
        
        st.markdown(f"""
        <div class="insight-box" style="background-color: #f0fdf4; border-left-color: #10b981;">
            <div class="insight-title" style="color: #065f46;">💎 Premium Customer Segment Baseline</div>
            <div class="insight-desc" style="color: #047857;">
                High-Value customers are classified as those with account balances of <b>${premium_balance_threshold:,.2f}</b> or above. 
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
                fig_scatter.update_layout(xaxis_title="Account Balance ($)", yaxis_title="Estimated Annual Salary ($)")
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("⚠️ No High-Value Customer records match your active filtering parameters.")
                
        with col_hv_2:
            # Revenue Risk metric gauge
            total_premium_aum = filtered_df[filtered_df['Exited'] == 1]['Balance'].sum()
            st.markdown("### 💸 Revenue Assets Under Risk")
            st.metric(
                label="Total Lost Capital Assets (Churned Balance)",
                value=f"${total_premium_aum:,.2f}",
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
        
        # Preparation of machine learning framework
        ml_df = filtered_df.copy()
        
        if len(ml_df) > 50: # Ensure enough data exists to train the predictive model
            
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
                input_balance = st.number_input("Account Balance ($)", min_value=0.0, value=75000.0, step=1000.0)
                input_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=85000.0, step=1000.0)
                
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
