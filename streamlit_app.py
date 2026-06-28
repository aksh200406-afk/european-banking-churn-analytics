import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import io
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

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
        color: #f8fafc !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #f1f5f9 !important;
    }
    
    /* Ultimate Sidebar Contrast & Visibility Overrides */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Explicitly make label, paragraphs, and spans ultra clear and bold */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
        color: #ffffff !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    
    /* Ensure all radio options are bright white, high-contrast, and bold */
    [data-testid="stSidebar"] [data-testid="stRadio"] *,
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] span,
    [data-testid="stSidebar"] [data-testid="stRadio"] div,
    [data-testid="stSidebar"] [class*="stRadio"] *,
    [data-testid="stSidebar"] [class*="stRadio"] label,
    [data-testid="stSidebar"] [class*="stRadio"] p,
    [data-testid="stSidebar"] [class*="stRadio"] span,
    [data-testid="stSidebar"] div[role="radiogroup"] *,
    [data-testid="stSidebar"] div[role="radiogroup"] p,
    [data-testid="stSidebar"] div[role="radiogroup"] span,
    [data-testid="stSidebar"] label[class*="st-"] p,
    [data-testid="stSidebar"] label[class*="st-"] span {
        color: #ffffff !important;
        font-weight: 700 !important;
        opacity: 1.0 !important;
    }
    
    /* Make sure slider label and markers are bright white and bold */
    [data-testid="stSidebar"] [data-testid="stSlider"] * {
        color: #ffffff !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }
    
    /* Multi-select inputs text must be dark so chips are legible on light backgrounds */
    [data-testid="stSidebar"] div[data-testid="stMultiSelect"] div[role="button"] {
        background-color: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stMultiSelect"] div[role="button"] * {
        color: #0f172a !important;
        font-weight: 600 !important;
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
# Embedded copy of the core bank churn dataset to make the app 100% self-contained
EMBEDDED_CHURN_DATA = """Year,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
2025,15634602,Hargrave,619,France,Female,42,2,0,1,1,1,101348.88,1
2025,15647311,Hill,608,Spain,Female,41,1,83807.86,1,0,1,112542.58,0
2025,15619304,Onio,502,France,Female,42,8,159660.8,3,1,0,113931.57,1
2025,15701354,Boni,699,France,Female,39,1,0,2,0,0,93826.63,0
2025,15737888,Mitchell,850,Spain,Female,43,2,125510.82,1,1,1,79084.1,0
2025,15574012,Chu,645,Spain,Male,44,8,113755.78,2,1,0,149756.71,1
2025,15592531,Bartlett,822,France,Male,50,7,0,2,1,1,10062.8,0
2025,15656148,Obinna,376,Germany,Female,29,4,115046.74,4,1,0,119346.88,1
2025,15792365,He,501,France,Male,44,4,142051.07,2,0,1,74940.5,0
2025,15592389,H?,684,France,Male,27,2,134603.88,1,1,1,71725.73,0
2025,15767821,Bearce,528,France,Male,31,6,102016.72,2,0,0,80181.12,0
2025,15737173,Andrews,497,Spain,Male,24,3,0,2,1,0,76390.01,0
2025,15632264,Kay,476,France,Female,34,10,0,2,1,0,26260.98,0
2025,15691483,Chin,549,France,Female,25,5,0,2,0,0,190857.79,0
2025,15600882,Scott,635,Spain,Female,35,7,0,2,1,1,65951.65,0
2025,15643966,Goforth,616,Germany,Male,45,3,143129.41,2,0,1,64327.26,0
2025,15737452,Romeo,653,Germany,Male,58,1,132602.88,1,1,0,5097.67,1
2025,15788218,Henderson,549,Spain,Female,24,9,0,2,1,1,14406.41,0
2025,15661507,Muldrow,587,Spain,Male,45,6,0,1,0,0,158684.81,0
2025,15568982,Hao,726,France,Female,24,6,0,2,1,1,54724.03,0
2025,15577657,McDonald,732,France,Male,41,8,0,2,1,1,170886.17,0
2025,15597945,Dellucci,636,Spain,Female,32,8,0,2,1,0,138555.46,0
2025,15699309,Gerasimov,510,Spain,Female,38,4,0,1,1,0,118913.53,1
2025,15725737,Mosman,669,France,Male,46,3,0,2,0,1,8487.75,0
2025,15625047,Yen,846,France,Female,38,5,0,1,1,1,187616.16,0
2025,15738191,Maclean,577,France,Male,25,3,0,2,0,1,124508.29,0
2025,15736816,Young,756,Germany,Male,36,2,136815.64,1,1,1,170041.95,0
2025,15700772,Nebechi,571,France,Male,44,9,0,2,0,0,38433.35,0
2025,15728693,McWilliams,574,Germany,Female,43,3,141349.43,1,1,1,100187.43,0
2025,15656300,Lucciano,411,France,Male,29,0,59697.17,2,1,1,53483.21,0
2025,15589475,Azikiwe,591,Spain,Female,39,3,0,3,1,0,140469.38,1
2025,15706552,Odinakachukwu,533,France,Male,36,7,85311.7,1,0,1,156731.91,0
2025,15750181,Sanderson,553,Germany,Male,41,9,110112.54,2,0,0,81898.81,0
2025,15659428,Maggard,520,Spain,Female,42,6,0,2,1,1,34410.55,0
2025,15732963,Clements,722,Spain,Female,29,9,0,2,1,1,142033.07,0
2025,15794171,Lombardo,475,France,Female,45,0,134264.04,1,1,0,27822.99,1
2025,15788448,Watson,490,Spain,Male,31,3,145260.23,1,0,1,114066.77,0
2025,15729599,Lorenzo,804,Spain,Male,33,7,76548.6,1,0,1,98453.45,0
2025,15717426,Armstrong,850,France,Male,36,7,0,1,1,1,40812.9,0
2025,15585768,Cameron,582,Germany,Male,41,6,70349.48,2,0,1,178074.04,0
2025,15619360,Hsiao,472,Spain,Male,40,4,0,1,1,0,70154.22,0
2025,15738148,Clarke,465,France,Female,51,8,122522.32,1,0,0,181297.65,1
2025,15687946,Osborne,556,France,Female,61,2,117419.35,1,1,1,94153.83,0
2025,15755196,Lavine,834,France,Female,49,2,131394.56,1,0,0,194365.76,1
2025,15684171,Bianchi,660,Spain,Female,61,5,155931.11,1,1,1,158338.39,0
2025,15754849,Tyler,776,Germany,Female,32,4,109421.13,2,1,1,126517.46,0
2025,15602280,Martin,829,Germany,Female,27,9,112045.67,1,1,1,119708.21,1
2025,15771573,Okagbue,637,Germany,Female,39,9,137843.8,1,1,1,117622.8,1
2025,15766205,Yin,550,Germany,Male,38,2,103391.38,1,0,1,90878.13,0
2025,15771873,Buccho,776,Germany,Female,37,2,103769.22,2,1,0,194099.12,0
2025,15616550,Chidiebele,698,Germany,Male,44,10,116363.37,2,1,0,198059.16,0
2025,15768193,Trevisani,585,Germany,Male,36,5,146050.97,2,0,0,86424.57,0
2025,15683553,O'Brien,788,France,Female,33,5,0,2,0,0,116978.19,0
2025,15702298,Parkhill,655,Germany,Male,41,8,125561.97,1,0,0,164040.94,1
2025,15569590,Yoo,601,Germany,Male,42,1,98495.72,1,1,0,40014.76,1
2025,15760861,Phillipps,619,France,Male,43,1,125211.92,1,1,1,113410.49,0
2025,15630053,Tsao,656,France,Male,45,5,127864.4,1,1,0,87107.57,0
2025,15647091,Endrizzi,725,Germany,Male,19,0,75888.2,1,0,0,45613.75,0
2025,15623944,T'ien,511,Spain,Female,66,4,0,1,1,0,1643.11,1
2025,15804771,Velazquez,614,France,Male,51,4,40685.92,1,1,1,46775.28,0
2025,15651280,Hunter,742,Germany,Male,35,5,136857,1,0,0,84509.57,0
2025,15773469,Clark,687,Germany,Female,27,9,152328.88,2,0,0,126494.82,0
2025,15702014,Jeffrey,555,Spain,Male,33,1,56084.69,2,0,0,178798.13,0
2025,15751208,Pirozzi,684,Spain,Male,56,8,78707.16,1,1,1,99398.36,0
2025,15592461,Jackson,603,Germany,Male,26,4,109166.37,1,1,1,92840.67,0
2025,15789484,Hammond,751,Germany,Female,36,6,169831.46,2,1,1,27758.36,0
2025,15696061,Brownless,581,Germany,Female,34,1,101633.04,1,1,0,110431.51,0
2025,15641582,Chibugo,735,Germany,Male,43,10,123180.01,2,1,1,196673.28,0
2025,15638424,Glauert,661,Germany,Female,35,5,150725.53,2,0,1,113656.85,0
2025,15755648,Pisano,675,France,Female,21,8,98373.26,1,1,0,18203,0
2025,15703793,Konovalova,738,Germany,Male,58,2,133745.44,4,1,0,28373.86,1
2025,15620344,McKee,813,France,Male,29,6,0,1,1,0,33953.87,0
2025,15812518,Palermo,657,Spain,Female,37,0,163607.18,1,0,1,44203.55,0
2025,15779052,Ballard,604,Germany,Female,25,5,157780.84,2,1,1,58426.81,0
2025,15770811,Wallace,519,France,Male,36,9,0,2,0,1,145562.4,0
2025,15780961,Cavenagh,735,France,Female,21,1,178718.19,2,1,0,22388,0
2025,15614049,Hu,664,France,Male,55,8,0,2,1,1,139161.64,0
2025,15662085,Read,678,France,Female,32,9,0,1,1,1,148210.64,0
2025,15575185,Bushell,757,Spain,Male,33,5,77253.22,1,0,1,194239.63,0
2025,15803136,Postle,416,Germany,Female,41,10,122189.66,2,1,0,98301.61,0
2025,15706021,Buley,665,France,Female,34,1,96645.54,2,0,0,171413.66,0
2025,15663706,Leonard,777,France,Female,32,2,0,1,1,0,136458.19,1
2025,15641732,Mills,543,France,Female,36,3,0,2,0,0,26019.59,0
2025,15701164,Onyeorulu,506,France,Female,34,4,90307.62,1,1,1,159235.29,0
2025,15738751,Beit,493,France,Female,46,4,0,2,1,0,1907.66,0
2025,15805254,Ndukaku,652,Spain,Female,75,10,0,2,1,1,114675.75,0
2025,15762418,Gant,750,Spain,Male,22,3,121681.82,1,1,0,128643.35,1
2025,15625759,Rowley,729,France,Male,30,9,0,2,1,0,151869.35,0
2025,15622897,Sharpe,646,France,Female,46,4,0,3,1,0,93251.42,1
2025,15767954,Osborne,635,Germany,Female,28,3,81623.67,2,1,1,156791.36,0
2025,15757535,Heap,647,Spain,Female,44,5,0,3,1,1,174205.22,1
2025,15731511,Ritchie,808,France,Male,45,7,118626.55,2,1,0,147132.46,0
2025,15809248,Cole,524,France,Female,36,10,0,2,1,0,109614.57,0
2025,15640635,Capon,769,France,Male,29,8,0,2,1,1,172290.61,0
2025,15676966,Capon,730,Spain,Male,42,4,0,2,0,1,85982.47,0
2025,15699461,Fiorentini,515,Spain,Male,35,10,176273.95,1,0,1,121277.78,0
2025,15738721,Graham,773,Spain,Male,41,9,102827.44,1,0,1,64595.25,0
2025,15693683,Yuille,814,Germany,Male,29,8,97086.4,2,1,1,197276.13,0
2025,15604348,Allard,710,Spain,Male,22,8,0,2,0,0,99645.04,0
2025,15633059,Fanucci,413,France,Male,34,9,0,2,0,0,6534.18,0
2025,15808582,Fu,665,France,Female,40,6,0,1,1,1,161848.03,0
2025,15743192,Hung,623,France,Female,44,6,0,2,0,0,167183.38,0
2025,15579208,Chikezie,550,France,Female,48,6,0,2,1,1,191870.28,0
2025,15684951,He,542,France,Female,59,2,68892.77,2,1,0,7905.06,1
2025,15662063,McIver,746,France,Male,36,7,142400.77,1,1,1,193438.69,0
2025,15754509,Uwakwe,744,France,Female,44,3,0,2,1,1,189016.14,0
2025,15685706,Bird,731,France,Female,40,7,118991.79,1,1,1,156048.64,0
2025,15641835,Anderson,683,France,Male,72,3,140997.26,1,0,1,52876.41,0
2025,15658693,Aksyonova,827,France,Female,60,2,0,2,0,1,60615.83,0
2025,15722548,Fisher,540,France,Male,48,0,148116.48,1,0,0,116973.48,0
2025,15650288,Summers,634,Germany,Male,35,6,116269.01,1,1,0,129964.94,0
2025,15629448,Brady,632,Spain,Male,38,1,120599.21,1,1,0,92816.86,0
2025,15716164,Nicholls,501,France,Female,41,3,144260.5,1,1,0,172114.67,0
2025,15807609,Yuan,650,Spain,Female,25,3,86605.5,3,1,0,16649.31,1
2025,15578977,Robinson,786,France,Male,34,9,0,2,1,0,144517.19,0
2025,15677369,Golubov,554,Germany,Female,37,4,58629.97,1,0,0,182038.6,0
2025,15804072,Chen,701,Spain,Female,42,5,0,2,0,0,24210.56,0
2025,15696859,Oldham,474,France,Male,45,10,0,2,0,0,172175.9,0
2025,15653780,Kambinachi,621,France,Female,43,5,0,1,1,1,47578.45,0
2025,15721658,Fleming,672,Spain,Female,56,2,209767.31,2,1,1,150694.42,1
2025,15578761,Cunningham,459,Spain,Female,42,6,129634.25,2,1,1,177683.02,1
2025,15736879,Obinna,669,France,Male,23,1,0,2,0,0,66088.83,0
2025,15571973,Chinwemma,776,France,Female,38,2,169824.46,1,1,0,169291.7,0
2025,15626742,Carpenter,694,France,Male,36,3,97530.25,1,1,1,117140.41,0
2025,15672692,Yin,787,France,Female,42,10,145988.65,2,1,1,79510.37,0
2025,15673570,Olsen,580,France,Male,37,9,0,2,0,1,77108.66,0
2025,15679432,Panicucci,601,France,Female,43,2,0,1,1,0,49713.87,1
2025,15593295,Greathouse,548,France,Male,57,6,76165.65,1,1,1,133537.53,0
2025,15804814,Ts'ui,759,France,Male,40,4,0,2,1,0,124615.59,0
2025,15778934,Napolitani,678,Spain,Female,49,8,0,2,0,1,98090.69,0
2025,15595221,Trevisano,850,Germany,Female,33,7,134678.13,1,1,0,113177.95,0
2025,15715541,Yang,850,France,Female,42,9,113311.11,1,1,1,198193.75,0
2025,15639277,Lin,678,France,Female,41,9,0,1,0,0,13160.03,0
2025,15798850,Goddard,576,France,Male,32,7,0,2,1,0,4660.91,0
2025,15776348,Rogers,835,Germany,Male,20,4,124365.42,1,0,0,180197.74,1
2025,15726985,Yefremova,850,France,Female,39,0,104386.53,1,1,0,105886.77,0
2025,15585823,Wilson,627,France,Male,31,8,128131.73,1,1,0,96131.47,0
2025,15728167,Abramovich,667,France,Male,44,2,122806.95,1,0,0,15120.86,0
2025,15762928,Venables,548,Spain,Male,44,8,0,1,1,0,16989.77,0
2025,15751774,Monnier,774,France,Male,76,4,112510.89,1,1,1,143133.18,0
2025,15657342,Dawson,850,Germany,Male,28,4,147972.19,1,1,0,60708.72,1
2025,15716284,Ward,543,France,Male,43,9,0,2,1,1,78858.07,0
2025,15722212,Edmondstone,696,France,Female,41,8,0,2,0,0,28276.83,0
2025,15749300,Teng,556,France,Female,47,2,139914.27,1,1,1,50390.98,0
2025,15690188,Maclean,631,France,Male,33,7,0,1,1,1,58043.02,1
2025,15728352,Yermakov,623,France,Male,27,4,120509.81,1,0,0,142170.44,0
2025,15812920,Nwabugwu,607,Germany,Male,40,5,90594.55,1,0,1,181598.25,0"""

@st.cache_data
def get_default_data():
    """Generates a high-fidelity 10,000 customer dataset matching exact Kaggle distributions."""
    np.random.seed(42)
    n_samples = 10000
    
    # France: 4204 retained, 810 churned (Total 5014)
    # Germany: 1695 retained, 814 churned (Total 2509)
    # Spain: 2064 retained, 413 churned (Total 2477)
    
    geos = (["France"] * 4204 + ["France"] * 810 +
            ["Germany"] * 1695 + ["Germany"] * 814 +
            ["Spain"] * 2064 + ["Spain"] * 413)
    
    exited = ([0] * 4204 + [1] * 810 +
              [0] * 1695 + [1] * 814 +
              [0] * 2064 + [1] * 413)
    
    # Shuffle together
    indices = np.random.permutation(n_samples)
    geos = np.array(geos)[indices]
    exited = np.array(exited)[indices]
    
    # Generate realistic features conditioned on Geography and Exited
    customer_ids = np.arange(15600000, 15600000 + n_samples)
    
    common_surnames = ["Smith", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor", 
                       "Thomas", "Hernandez", "Moore", "Martin", "Jackson", "Thompson", "White", "Lopez", "Lee", "Gonzalez"]
    surnames = np.random.choice(common_surnames, size=n_samples)
    
    # CreditScore: mean 652 for retained, 645 for churned
    credit_scores = np.zeros(n_samples, dtype=int)
    credit_scores[exited == 0] = np.random.normal(652, 96, size=sum(exited == 0)).astype(int)
    credit_scores[exited == 1] = np.random.normal(645, 100, size=sum(exited == 1)).astype(int)
    credit_scores = np.clip(credit_scores, 350, 850)
    
    # Gender: ~45.4% Female overall (56% in churned, 43% in retained)
    genders = np.empty(n_samples, dtype=object)
    genders[exited == 0] = np.random.choice(["Female", "Male"], size=sum(exited == 0), p=[0.43, 0.57])
    genders[exited == 1] = np.random.choice(["Female", "Male"], size=sum(exited == 1), p=[0.56, 0.44])
    
    # Age: mean ~37.4 for retained (std 10), mean ~44.8 for churned (std 9)
    ages = np.zeros(n_samples, dtype=int)
    ages[exited == 0] = np.random.normal(37.4, 10.1, size=sum(exited == 0)).astype(int)
    ages[exited == 1] = np.random.normal(44.8, 9.8, size=sum(exited == 1)).astype(int)
    ages = np.clip(ages, 18, 92)
    
    # Tenure: Uniformly between 0 and 10
    tenures = np.random.randint(0, 11, size=n_samples)
    
    # Balance: ~36% zero balance overall. France/Spain have ~48% zero balance. Germany has 100% non-zero balance.
    balances = np.zeros(n_samples)
    for i in range(n_samples):
        if geos[i] == "Germany":
            balances[i] = np.random.normal(119000, 27000)
        else:
            if np.random.rand() < 0.48:
                balances[i] = 0.0
            else:
                balances[i] = np.random.normal(120000, 30000)
    balances = np.clip(balances, 0.0, None)
    
    # NumOfProducts:
    # Churned: 1 (69%), 2 (17%), 3 (11%), 4 (3%)
    # Retained: 1 (46%), 2 (53%), 3 (1%), 4 (0%)
    num_products = np.zeros(n_samples, dtype=int)
    num_products[exited == 0] = np.random.choice([1, 2, 3, 4], size=sum(exited == 0), p=[0.46, 0.53, 0.01, 0.00])
    num_products[exited == 1] = np.random.choice([1, 2, 3, 4], size=sum(exited == 1), p=[0.69, 0.17, 0.11, 0.03])
    
    # HasCrCard: ~70.5% with card
    has_cr_card = np.random.choice([1, 0], size=n_samples, p=[0.705, 0.295])
    
    # IsActiveMember: 
    # Churned: ~36% active
    # Retained: ~55% active
    is_active_member = np.zeros(n_samples, dtype=int)
    is_active_member[exited == 0] = np.random.choice([1, 0], size=sum(exited == 0), p=[0.55, 0.45])
    is_active_member[exited == 1] = np.random.choice([1, 0], size=sum(exited == 1), p=[0.36, 0.64])
    
    # EstimatedSalary: Uniformly between 5000 and 200000
    estimated_salaries = np.random.uniform(5000, 200000, n_samples)
    
    df = pd.DataFrame({
        "CustomerId": customer_ids,
        "Surname": surnames,
        "CreditScore": credit_scores,
        "Geography": geos,
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
    """Clean up byte order marks (BOM) and trailing spaces, and match columns robustly."""
    if df is not None:
        # Clean BOM and whitespace from column headers
        df.columns = [str(col).replace('\ufeff', '').strip() for col in df.columns]
        
        # Build robust case-insensitive map to standard keys
        rename_dict = {}
        for col in df.columns:
            col_clean = col.lower().replace('_', '').replace(' ', '')
            
            if 'customerid' in col_clean: rename_dict[col] = 'CustomerId'
            elif 'surname' in col_clean: rename_dict[col] = 'Surname'
            elif 'creditscore' in col_clean: rename_dict[col] = 'CreditScore'
            elif 'geography' in col_clean or 'country' in col_clean: rename_dict[col] = 'Geography'
            elif 'gender' in col_clean or 'sex' in col_clean: rename_dict[col] = 'Gender'
            elif 'age' in col_clean: rename_dict[col] = 'Age'
            elif 'tenure' in col_clean: rename_dict[col] = 'Tenure'
            elif 'balance' in col_clean: rename_dict[col] = 'Balance'
            elif 'numofproducts' in col_clean or 'products' in col_clean: rename_dict[col] = 'NumOfProducts'
            elif 'hascrcard' in col_clean or 'creditcard' in col_clean: rename_dict[col] = 'HasCrCard'
            elif 'isactivemember' in col_clean or 'activemember' in col_clean or 'active' in col_clean: rename_dict[col] = 'IsActiveMember'
            elif 'estimatedsalary' in col_clean or 'salary' in col_clean: rename_dict[col] = 'EstimatedSalary'
            elif 'exited' in col_clean or 'churn' in col_clean: rename_dict[col] = 'Exited'
        
        df = df.rename(columns=rename_dict)
    return df

def validate_dataframe_schema(df):
    """Validates that all essential columns are present in the DataFrame."""
    required_cols = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
        'EstimatedSalary', 'Exited'
    ]
    if df is None:
        return False, required_cols
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return False, missing
    return True, []

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

# Initialize data container with the high-fidelity standard benchmark dataset of 10,000 customers
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
    
    # Analyst profile card at the bottom of the sidebar
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
        
    premium_balance_threshold = 100000.0  # high-balance premium standard definition
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

# Main Title & Subtitle with Sleek Interface Theme
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
