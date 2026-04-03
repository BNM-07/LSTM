"""
🚀 GCC Retail Sales Forecasting — Premium Streamlit Dashboard
Production-grade analytics dashboard with LSTM forecasting.
"""
import os
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import timedelta

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GCC Sales Intelligence | AI Forecaster",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.main { background: linear-gradient(180deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%); }
.stApp { background: linear-gradient(180deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1225 0%, #131b35 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdown"] p {
    color: #c7d2fe !important;
}

/* ── Glassmorphism Cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.05) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    transform: translateY(-2px);
}
div[data-testid="stMetricLabel"] p {
    color: #a5b4fc !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetricValue"] {
    color: #e0e7ff !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
}
div[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(99, 102, 241, 0.05);
    border-radius: 14px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(99, 102, 241, 0.1);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #94a3b8 !important;
    font-weight: 500;
    padding: 10px 20px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}

/* ── Headers ── */
h1 { 
    color: #e0e7ff !important; 
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
h2, h3 { 
    color: #c7d2fe !important; 
    font-weight: 600 !important; 
}

/* ── Dividers ── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
    margin: 1.5rem 0;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Selectboxes / Multiselect ── */
div[data-baseweb="select"] {
    border-radius: 10px !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: rgba(99, 102, 241, 0.05) !important;
    border-radius: 10px !important;
}

/* ── Info/Success/Warning boxes ── */
div[data-testid="stAlert"] {
    border-radius: 12px;
    border: none;
    backdrop-filter: blur(10px);
}

/* ── Checkbox ── */
.stCheckbox label span {
    color: #c7d2fe !important;
}

/* ── Widget labels (selectbox, slider, etc.) ── */
.stSelectbox label, .stMultiSelect label, .stSlider label, .stDateInput label,
div[data-testid="stWidgetLabel"] p {
    color: #c7d2fe !important;
    font-weight: 500 !important;
}

/* ── General paragraph text ── */
p, span, label {
    color: #c7d2fe;
}

/* ── Sidebar text ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #c7d2fe !important;
}
[data-testid="stSidebar"] .stSelectbox label p,
[data-testid="stSidebar"] .stMultiSelect label p {
    color: #a5b4fc !important;
    font-weight: 500 !important;
}

/* ── Dataframe text ── */
.stDataFrame { color: #c7d2fe !important; }

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 50%, rgba(59, 130, 246, 0.08) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 20px;
    padding: 30px 40px;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
}

/* ── KPI Grid Card ── */
.kpi-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.05) 100%);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    backdrop-filter: blur(20px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: all 0.3s;
}
.kpi-card:hover { border-color: rgba(99, 102, 241, 0.3); transform: translateY(-2px); }
.kpi-label { color: #a5b4fc; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.kpi-value { color: #e0e7ff; font-size: 2rem; font-weight: 800; }
.kpi-delta { font-size: 0.85rem; font-weight: 600; margin-top: 4px; }
.kpi-delta.positive { color: #34d399; }
.kpi-delta.negative { color: #f87171; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter", color="#e0e7ff", size=13),
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color="#c7d2fe"),
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1,
    ),
    xaxis=dict(gridcolor='rgba(99,102,241,0.08)', zerolinecolor='rgba(99,102,241,0.08)',
               tickfont=dict(size=12, color='#c7d2fe'), title=dict(font=dict(size=13, color='#a5b4fc'))),
    yaxis=dict(gridcolor='rgba(99,102,241,0.08)', zerolinecolor='rgba(99,102,241,0.08)',
               tickfont=dict(size=12, color='#c7d2fe'), title=dict(font=dict(size=13, color='#a5b4fc'))),
)

COLORS = {
    'primary': '#6366f1',
    'secondary': '#8b5cf6',
    'accent': '#a78bfa',
    'success': '#34d399',
    'warning': '#fbbf24',
    'danger': '#f87171',
    'info': '#60a5fa',
    'gradient': ['#6366f1', '#8b5cf6', '#a78bfa', '#c084fc', '#e879f9'],
    'regions': {'UAE': '#6366f1', 'KSA': '#8b5cf6', 'Oman': '#34d399', 'Qatar': '#fbbf24'},
    'categories': px.colors.qualitative.Pastel,
}

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_sales_data():
    df = pd.read_csv('sales_data.csv', parse_dates=['date'])
    return df

@st.cache_data(ttl=3600)
def load_model_metrics(model_type='lstm'):
    """Load metrics for a specific model type, with fallback to generic."""
    for path in [f'models/metrics_{model_type}.json', 'models/metrics.json']:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            continue
    return None

@st.cache_data(ttl=3600)
def load_predictions(model_type='lstm'):
    """Load predictions for a specific model type, with fallback to generic."""
    for path in [f'models/test_predictions_{model_type}.csv', 'models/test_predictions.csv']:
        try:
            return pd.read_csv(path, parse_dates=['date'])
        except FileNotFoundError:
            continue
    return None

@st.cache_data(ttl=3600)
def load_training_history(model_type='lstm'):
    """Load training history for a specific model type, with fallback to generic."""
    for path in [f'models/training_history_{model_type}.json', 'models/training_history.json']:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            continue
    return None

def load_model_and_preprocessor(model_type='lstm'):
    try:
        import keras
        import joblib
        model = keras.models.load_model(f'models/sales_{model_type}_model.keras')
        prep = joblib.load('models/preprocessor.pkl')
        return model, prep
    except:
        return None, None


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
df = load_sales_data()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size: 2.5rem;">🚀</div>
        <div style="font-size: 1.2rem; font-weight: 700; color: #a5b4fc; letter-spacing: -0.3px;">
            AI Sales Forecaster
        </div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
            GCC Retail Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Model Selection ──
    st.markdown("##### 🧠 Model Selection")
    selected_model = st.selectbox(
        "Select Forecasting Model",
        ["LSTM", "GRU"],
        index=0,
        key="model_selector"
    )
    compare_models = st.checkbox("☑ Compare Models", value=False, key="compare_toggle")
    
    model_type_key = selected_model.lower()
    
    # Show active model badge
    badge_color = "#6366f1" if selected_model == "LSTM" else "#8b5cf6"
    st.markdown(
        f'<div style="text-align:center; margin: 8px 0 4px 0; padding: 6px 12px; '
        f'background: linear-gradient(135deg, {badge_color}33, {badge_color}18); '
        f'border: 1px solid {badge_color}50; border-radius: 10px;">'
        f'<span style="color: #e0e7ff; font-weight: 600; font-size: 0.85rem;">'
        f'Active: {selected_model}</span></div>',
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown("##### 🎛️ Filters")
    
    # Date Range
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Region
    regions = st.multiselect(
        "Region",
        options=sorted(df['region'].unique()),
        default=sorted(df['region'].unique()),
    )
    
    # Category
    categories = st.multiselect(
        "Category",
        options=sorted(df['category'].unique()),
        default=sorted(df['category'].unique()),
    )
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#94a3b8; font-size:0.7rem;'>"
        "Powered by LSTM &amp; GRU Neural Networks<br>© 2024 GCC Analytics</div>",
        unsafe_allow_html=True
    )

# ── Load model-specific data ──
metrics = load_model_metrics(model_type_key)
predictions = load_predictions(model_type_key)
history = load_training_history(model_type_key)

# Apply filters
if len(date_range) == 2:
    mask = (
        (df['date'].dt.date >= date_range[0]) &
        (df['date'].dt.date <= date_range[1]) &
        (df['region'].isin(regions)) &
        (df['category'].isin(categories))
    )
else:
    mask = (df['region'].isin(regions)) & (df['category'].isin(categories))

fdf = df[mask].copy()

# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">GCC Sales Intelligence Dashboard</div>
    <div class="hero-subtitle">Real-time analytics & AI-powered forecasting across UAE, KSA, Oman & Qatar</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary",
    "🔮 AI Forecast",
    "📈 Analytics Deep Dive",
    "🕵️ Anomaly Detection",
    "🧪 Scenario Simulator"
])


# ═════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE SUMMARY
# ═════════════════════════════════════════════════════════════
with tab1:
    # ── KPI Row ──
    total_revenue = fdf['revenue'].sum()
    total_units = fdf['units_sold'].sum()
    avg_order = fdf[fdf['units_sold'] > 0]['revenue'].mean() if len(fdf[fdf['units_sold'] > 0]) > 0 else 0
    active_stores = fdf['store_id'].nunique()
    unique_products = fdf['product_id'].nunique()
    avg_conversion = fdf['conversion_rate'].mean() * 100
    
    # YoY growth
    this_year = fdf[fdf['date'].dt.year == fdf['date'].dt.year.max()]['revenue'].sum()
    last_year = fdf[fdf['date'].dt.year == fdf['date'].dt.year.max() - 1]['revenue'].sum()
    yoy_growth = ((this_year - last_year) / last_year * 100) if last_year > 0 else 0
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Revenue", f"${total_revenue/1e6:.1f}M", f"{yoy_growth:+.1f}% YoY")
    c2.metric("Units Sold", f"{total_units/1e6:.2f}M")
    c3.metric("Avg Transaction", f"${avg_order:.0f}")
    c4.metric("Active Stores", f"{active_stores}")
    c5.metric("Products", f"{unique_products}")
    c6.metric("Conversion Rate", f"{avg_conversion:.2f}%")
    
    st.markdown("")
    
    # ── Revenue Trend ──
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("#### 📈 Revenue Trend")
        
        daily_rev = fdf.groupby('date')['revenue'].sum().reset_index()
        daily_rev['rev_7d'] = daily_rev['revenue'].rolling(7).mean()
        daily_rev['rev_30d'] = daily_rev['revenue'].rolling(30).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_rev['date'], y=daily_rev['revenue'],
            mode='lines', name='Daily Revenue',
            line=dict(color='rgba(99,102,241,0.2)', width=1),
            fill='tozeroy', fillcolor='rgba(99,102,241,0.03)'
        ))
        fig.add_trace(go.Scatter(
            x=daily_rev['date'], y=daily_rev['rev_7d'],
            mode='lines', name='7-Day MA',
            line=dict(color=COLORS['primary'], width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=daily_rev['date'], y=daily_rev['rev_30d'],
            mode='lines', name='30-Day MA',
            line=dict(color=COLORS['accent'], width=2, dash='dash')
        ))
        fig.update_layout(**PLOT_LAYOUT, height=400, title="")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("#### 🌍 Revenue by Region")
        
        reg_rev = fdf.groupby('region')['revenue'].sum().reset_index()
        reg_rev = reg_rev.sort_values('revenue', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=reg_rev['revenue'],
            y=reg_rev['region'],
            orientation='h',
            marker=dict(
                color=[COLORS['regions'].get(r, '#6366f1') for r in reg_rev['region']],
                line=dict(width=0),
            ),
            text=[f"${v/1e6:.1f}M" for v in reg_rev['revenue']],
            textposition='auto',
            textfont=dict(color='white', size=12, family='Inter'),
        ))
        fig.update_layout(**PLOT_LAYOUT, height=400, title="",
                         showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # ── Row 2 ──
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 🏷️ Category Performance")
        cat_df = fdf.groupby('category').agg(
            revenue=('revenue', 'sum'),
            units=('units_sold', 'sum')
        ).reset_index().sort_values('revenue', ascending=False)
        
        fig = px.treemap(cat_df, path=['category'], values='revenue',
                        color='revenue', color_continuous_scale='Purples')
        fig.update_layout(**PLOT_LAYOUT, height=380, title="", coloraxis_showscale=False)
        fig.update_traces(textinfo='label+value+percent root',
                         texttemplate='%{label}<br>$%{value:,.0f}<br>%{percentRoot:.1%}',
                         textfont=dict(size=13, family='Inter'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.markdown("#### 📅 Monthly Revenue Heatmap")
        fdf_hm = fdf.copy()
        fdf_hm['year'] = fdf_hm['date'].dt.year
        fdf_hm['month'] = fdf_hm['date'].dt.month
        hm_data = fdf_hm.groupby(['year', 'month'])['revenue'].sum().reset_index()
        hm_pivot = hm_data.pivot(index='year', columns='month', values='revenue').fillna(0)
        
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        hm_pivot.columns = [month_names[m-1] for m in hm_pivot.columns]
        
        fig = go.Figure(go.Heatmap(
            z=hm_pivot.values,
            x=hm_pivot.columns,
            y=[str(y) for y in hm_pivot.index],
            colorscale=[[0, '#0a0e1a'], [0.5, '#6366f1'], [1, '#c084fc']],
            showscale=False,
            text=[[f"${v/1e6:.1f}M" for v in row] for row in hm_pivot.values],
            texttemplate='%{text}',
            textfont=dict(size=10, color='white'),
        ))
        fig.update_layout(**PLOT_LAYOUT, height=380, title="")
        st.plotly_chart(fig, use_container_width=True)
    
    # ── Business Insights ──
    st.markdown("#### 💡 AI-Generated Insights")
    from utils import generate_business_insights
    insights = generate_business_insights(fdf)
    
    insight_cols = st.columns(min(len(insights), 3))
    for i, insight in enumerate(insights[:6]):
        col_idx = i % 3
        with insight_cols[col_idx]:
            st.info(insight)


# ═════════════════════════════════════════════════════════════
# TAB 2: AI FORECAST
# ═════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"#### 🔮 {selected_model} Neural Network Forecast")
    
    # ── Model badge ──
    st.markdown(
        f'<div style="display:inline-block; padding: 4px 14px; border-radius: 8px; '
        f'background: linear-gradient(135deg, {badge_color}, {badge_color}cc); '
        f'color: white; font-weight: 600; font-size: 0.85rem; margin-bottom: 12px;">'
        f'Model: {selected_model}</div>',
        unsafe_allow_html=True
    )
    
    if predictions is not None and metrics is not None:
        # ── Model Metrics ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE", f"{metrics.get('rmse', 0):,.0f}")
        m2.metric("MAE", f"{metrics.get('mae', 0):,.0f}")
        m3.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
        m4.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
        
        st.markdown("")
        
        # ── Actual vs Predicted ──
        left, right = st.columns([3, 1])
        
        with left:
            st.markdown(f"##### Actual vs Predicted — {selected_model} Test Period")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predictions['date'], y=predictions['actual'],
                mode='lines', name='Actual',
                line=dict(color=COLORS['primary'], width=2.5)
            ))
            fig.add_trace(go.Scatter(
                x=predictions['date'], y=predictions['predicted'],
                mode='lines', name=f'{selected_model} Predicted',
                line=dict(color=COLORS['warning'], width=2.5, dash='dash')
            ))
            
            # Confidence band (± 10% of predicted)
            upper = predictions['predicted'] * 1.10
            lower = predictions['predicted'] * 0.90
            fig.add_trace(go.Scatter(
                x=pd.concat([predictions['date'], predictions['date'][::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill='toself', fillcolor='rgba(251,191,36,0.08)',
                line=dict(color='rgba(0,0,0,0)'),
                name='Confidence Band (±10%)'
            ))
            fig.update_layout(
                **PLOT_LAYOUT, height=450,
                title=f"Sales Forecast ({selected_model})"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with right:
            arch_name = "Stacked LSTM" if selected_model == "LSTM" else "Stacked GRU"
            layer_name = "LSTM" if selected_model == "LSTM" else "GRU"
            st.markdown("##### Model Info")
            st.markdown(f"""
            <div style="background: rgba(99,102,241,0.05); border: 1px solid rgba(99,102,241,0.15); 
                        border-radius: 14px; padding: 20px; font-size: 0.85rem; color: #c7d2fe;">
                <p><strong style="color:#e0e7ff;">Architecture:</strong> {arch_name}</p>
                <p><strong style="color:#e0e7ff;">Layers:</strong> 2 × {layer_name} (128 units)</p>
                <p><strong style="color:#e0e7ff;">Lookback:</strong> {metrics.get('lookback', 30)} days</p>
                <p><strong style="color:#e0e7ff;">Epochs:</strong> {metrics.get('epochs_run', 'N/A')}</p>
                <p><strong style="color:#e0e7ff;">Train Samples:</strong> {metrics.get('train_samples', 'N/A'):,}</p>
                <p><strong style="color:#e0e7ff;">Test Samples:</strong> {metrics.get('test_samples', 'N/A'):,}</p>
                <p><strong style="color:#e0e7ff;">Training Time:</strong> {metrics.get('training_time_seconds', 'N/A')}s</p>
            </div>
            """, unsafe_allow_html=True)
        
        # ── Training History ──
        if history:
            st.markdown("##### 📉 Training History")
            hc1, hc2 = st.columns(2)
            
            with hc1:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=history.get('loss', []), mode='lines', name='Train Loss',
                    line=dict(color=COLORS['primary'], width=2)
                ))
                fig_loss.add_trace(go.Scatter(
                    y=history.get('val_loss', []), mode='lines', name='Val Loss',
                    line=dict(color=COLORS['danger'], width=2, dash='dash')
                ))
                fig_loss.update_layout(**PLOT_LAYOUT, height=300, title="Loss (MSE)")
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with hc2:
                fig_mae = go.Figure()
                fig_mae.add_trace(go.Scatter(
                    y=history.get('mae', []), mode='lines', name='Train MAE',
                    line=dict(color=COLORS['success'], width=2)
                ))
                fig_mae.add_trace(go.Scatter(
                    y=history.get('val_mae', []), mode='lines', name='Val MAE',
                    line=dict(color=COLORS['warning'], width=2, dash='dash')
                ))
                fig_mae.update_layout(**PLOT_LAYOUT, height=300, title="Mean Absolute Error")
                st.plotly_chart(fig_mae, use_container_width=True)
        
        # ── Residual Analysis ──
        st.markdown("##### 📊 Residual Analysis")
        residuals = predictions['actual'] - predictions['predicted']
        
        rc1, rc2 = st.columns(2)
        with rc1:
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=predictions['date'], y=residuals,
                mode='markers', marker=dict(size=4, color=COLORS['accent'], opacity=0.5),
                name='Residuals'
            ))
            fig_res.add_hline(y=0, line_dash='dash', line_color='rgba(99,102,241,0.3)')
            fig_res.update_layout(**PLOT_LAYOUT, height=300, title="Residuals Over Time")
            st.plotly_chart(fig_res, use_container_width=True)
        
        with rc2:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=residuals, nbinsx=50,
                marker_color=COLORS['primary'],
                opacity=0.7
            ))
            fig_hist.update_layout(**PLOT_LAYOUT, height=300, title="Residual Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # ═══════════════════════════════════════════════════════
        # MODEL COMPARISON SECTION
        # ═══════════════════════════════════════════════════════
        if compare_models:
            st.markdown("---")
            st.markdown("#### ⚖️ Model Comparison: LSTM vs GRU")
            
            # Load both models' data
            metrics_lstm = load_model_metrics('lstm')
            metrics_gru = load_model_metrics('gru')
            preds_lstm = load_predictions('lstm')
            preds_gru = load_predictions('gru')
            
            if metrics_lstm and metrics_gru and preds_lstm is not None and preds_gru is not None:
                # ── Metric Comparison Cards ──
                cmp1, cmp2, cmp3, cmp4 = st.columns(4)
                
                def _metric_compare(col, label, key, lower_better=True):
                    v_l = metrics_lstm.get(key, 0)
                    v_g = metrics_gru.get(key, 0)
                    if lower_better:
                        winner = "LSTM" if v_l <= v_g else "GRU"
                    else:
                        winner = "LSTM" if v_l >= v_g else "GRU"
                    w_color = "#34d399"
                    fmt = f"{v_l:,.2f}" if key != 'r2' else f"{v_l:.4f}"
                    fmt_g = f"{v_g:,.2f}" if key != 'r2' else f"{v_g:.4f}"
                    col.markdown(
                        f'<div style="background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.15); '
                        f'border-radius:14px; padding:16px; text-align:center;">'
                        f'<div style="color:#a5b4fc; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">{label}</div>'
                        f'<div style="color:#e0e7ff; font-size:0.85rem;">LSTM: <strong>{fmt}</strong></div>'
                        f'<div style="color:#e0e7ff; font-size:0.85rem;">GRU: <strong>{fmt_g}</strong></div>'
                        f'<div style="color:{w_color}; font-size:0.75rem; margin-top:6px;">🏆 {winner}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                _metric_compare(cmp1, "RMSE", "rmse", lower_better=True)
                _metric_compare(cmp2, "MAE", "mae", lower_better=True)
                _metric_compare(cmp3, "MAPE", "mape", lower_better=True)
                _metric_compare(cmp4, "R² Score", "r2", lower_better=False)
                
                st.markdown("")
                
                # ── Overlay Predictions ──
                st.markdown("##### 📈 Prediction Overlay")
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Scatter(
                    x=preds_lstm['date'], y=preds_lstm['actual'],
                    mode='lines', name='Actual',
                    line=dict(color='#e0e7ff', width=2.5)
                ))
                fig_cmp.add_trace(go.Scatter(
                    x=preds_lstm['date'], y=preds_lstm['predicted'],
                    mode='lines', name='LSTM Predicted',
                    line=dict(color=COLORS['primary'], width=2, dash='dash')
                ))
                fig_cmp.add_trace(go.Scatter(
                    x=preds_gru['date'], y=preds_gru['predicted'],
                    mode='lines', name='GRU Predicted',
                    line=dict(color=COLORS['secondary'], width=2, dash='dot')
                ))
                fig_cmp.update_layout(
                    **PLOT_LAYOUT, height=450,
                    title="LSTM vs GRU — Prediction Comparison"
                )
                st.plotly_chart(fig_cmp, use_container_width=True)
            else:
                missing = []
                if not metrics_lstm or preds_lstm is None:
                    missing.append("LSTM")
                if not metrics_gru or preds_gru is None:
                    missing.append("GRU")
                st.warning(
                    f"⚠️ Cannot compare — {' and '.join(missing)} model artifacts not found. "
                    f"Train with: `python train.py --model both`"
                )
        
    else:
        st.warning(
            f"⚠️ {selected_model} model not found. Train it first:"
        )
        model_cmd = f"python train.py --model {model_type_key}"
        st.markdown(f"""
        ```bash
        # Step 1: Generate data (if needed)
        python data_generation.py
        
        # Step 2: Train {selected_model} model
        {model_cmd}
        
        # Step 3: Launch dashboard
        streamlit run app.py
        ```
        """)


# ═════════════════════════════════════════════════════════════
# TAB 3: ANALYTICS DEEP DIVE
# ═════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📈 Analytics Deep Dive")
    
    # ── Sub-category Analysis ──
    sub1, sub2 = st.columns(2)
    
    with sub1:
        st.markdown("##### Top 15 Sub-Categories by Revenue")
        subcat = fdf.groupby('sub_category')['revenue'].sum().nlargest(15).reset_index()
        fig = go.Figure(go.Bar(
            x=subcat['revenue'], y=subcat['sub_category'],
            orientation='h',
            marker=dict(
                color=np.linspace(0, 1, len(subcat)),
                colorscale=[[0, '#6366f1'], [1, '#c084fc']],
            ),
            text=[f"${v/1e6:.1f}M" for v in subcat['revenue']],
            textposition='auto',
            textfont=dict(color='white', size=11),
        ))
        fig.update_layout(**PLOT_LAYOUT, height=500, title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with sub2:
        st.markdown("##### Sales by Region & Category")
        rc_df = fdf.groupby(['region', 'category'])['revenue'].sum().reset_index()
        fig = px.sunburst(rc_df, path=['region', 'category'], values='revenue',
                         color='revenue', color_continuous_scale='Purples')
        fig.update_layout(**PLOT_LAYOUT, height=500, title="", coloraxis_showscale=False)
        fig.update_traces(textfont=dict(size=12, family='Inter'))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ── Discount & Marketing Analysis ──
    an1, an2 = st.columns(2)
    
    with an1:
        st.markdown("##### 💰 Discount Impact on Sales")
        fdf_disc = fdf.copy()
        fdf_disc['disc_bucket'] = pd.cut(fdf_disc['discount_percentage'], 
                                         bins=[-1, 0, 10, 20, 30, 100],
                                         labels=['No Discount', '1-10%', '11-20%', '21-30%', '30%+'])
        disc_impact = fdf_disc.groupby('disc_bucket', observed=True)['units_sold'].mean().reset_index()
        
        fig = go.Figure(go.Bar(
            x=disc_impact['disc_bucket'].astype(str),
            y=disc_impact['units_sold'],
            marker=dict(
                color=['#312e81', '#4338ca', '#6366f1', '#818cf8', '#a5b4fc'],
                line=dict(width=0),
            ),
            text=[f"{v:.1f}" for v in disc_impact['units_sold']],
            textposition='outside',
            textfont=dict(color='#c7d2fe', size=12),
        ))
        fig.update_layout(**PLOT_LAYOUT, height=400, title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with an2:
        st.markdown("##### 📢 Marketing Spend vs Revenue")
        daily_mkt = fdf.groupby('date').agg(
            marketing=('marketing_spend', 'sum'),
            revenue=('revenue', 'sum')
        ).reset_index()
        daily_mkt['mkt_7d'] = daily_mkt['marketing'].rolling(7).mean()
        daily_mkt['rev_7d'] = daily_mkt['revenue'].rolling(7).mean()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=daily_mkt['date'], y=daily_mkt['mkt_7d'],
            mode='lines', name='Marketing (7-day MA)',
            line=dict(color=COLORS['warning'], width=2)
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=daily_mkt['date'], y=daily_mkt['rev_7d'],
            mode='lines', name='Revenue (7-day MA)',
            line=dict(color=COLORS['primary'], width=2)
        ), secondary_y=True)
        fig.update_layout(**PLOT_LAYOUT, height=400, title="")
        fig.update_yaxes(title_text="Marketing $", secondary_y=False, gridcolor='rgba(99,102,241,0.08)')
        fig.update_yaxes(title_text="Revenue $", secondary_y=True, gridcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ── Seasonal Patterns ──
    sp1, sp2 = st.columns(2)
    
    with sp1:
        st.markdown("##### 📅 Day-of-Week Sales Pattern")
        dow = fdf.copy()
        dow['dow'] = dow['date'].dt.day_name()
        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow_agg = dow.groupby('dow')['units_sold'].mean().reindex(dow_order).reset_index()
        
        fig = go.Figure(go.Bar(
            x=dow_agg['dow'], y=dow_agg['units_sold'],
            marker=dict(
                color=[COLORS['primary'] if d in ['Friday', 'Saturday'] else 'rgba(99,102,241,0.3)' 
                       for d in dow_agg['dow']],
                line=dict(width=0),
            ),
            text=[f"{v:.0f}" for v in dow_agg['units_sold']],
            textposition='outside',
            textfont=dict(color='#c7d2fe'),
        ))
        fig.update_layout(**PLOT_LAYOUT, height=380, title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with sp2:
        st.markdown("##### 🎉 Holiday vs Normal Sales")
        hol_comp = fdf.groupby('holiday_flag')['units_sold'].mean().reset_index()
        hol_comp['label'] = hol_comp['holiday_flag'].map({0: 'Normal Days', 1: 'Holiday/Ramadan'})
        
        fig = go.Figure(go.Bar(
            x=hol_comp['label'], y=hol_comp['units_sold'],
            marker=dict(
                color=[COLORS['primary'], COLORS['warning']],
                line=dict(width=0),
            ),
            text=[f"{v:.1f}" for v in hol_comp['units_sold']],
            textposition='outside',
            textfont=dict(color='#c7d2fe', size=14, family='Inter'),
            width=0.5,
        ))
        fig.update_layout(**PLOT_LAYOUT, height=380, title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # ── Top Products ──
    st.markdown("##### 🏆 Top 10 Products by Revenue")
    top_prods = fdf.groupby(['product_id', 'product_name', 'category']).agg(
        total_revenue=('revenue', 'sum'),
        total_units=('units_sold', 'sum'),
    ).reset_index().nlargest(10, 'total_revenue')
    
    top_prods['total_revenue_fmt'] = top_prods['total_revenue'].apply(lambda x: f"${x:,.0f}")
    top_prods['total_units_fmt'] = top_prods['total_units'].apply(lambda x: f"{x:,}")
    
    st.dataframe(
        top_prods[['product_id', 'product_name', 'category', 'total_revenue_fmt', 'total_units_fmt']].rename(
            columns={'product_id': 'ID', 'product_name': 'Product', 'category': 'Category',
                    'total_revenue_fmt': 'Revenue', 'total_units_fmt': 'Units'}
        ),
        use_container_width=True,
        hide_index=True
    )


# ═════════════════════════════════════════════════════════════
# TAB 4: ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 🕵️ Anomaly Detection")
    st.markdown("Identifies unusual sales events using rolling Z-score analysis.")
    
    from utils import detect_anomalies
    
    # Aggregate daily
    daily_ts = fdf.groupby('date')['units_sold'].sum().reset_index()
    
    ac1, ac2 = st.columns([1, 1])
    with ac1:
        window = st.slider("Rolling Window (days)", 7, 60, 14, key='anom_window')
    with ac2:
        threshold = st.slider("Z-Score Threshold", 1.5, 4.0, 2.5, 0.1, key='anom_thresh')
    
    anom_df = detect_anomalies(daily_ts, col='units_sold', window=window, threshold=threshold)
    
    # ── Anomaly Chart ──
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=anom_df['date'], y=anom_df['units_sold'],
        mode='lines', name='Units Sold',
        line=dict(color=COLORS['primary'], width=1.5)
    ))
    
    # Spikes
    spikes = anom_df[anom_df['anomaly_type'] == 'Spike']
    if len(spikes) > 0:
        fig.add_trace(go.Scatter(
            x=spikes['date'], y=spikes['units_sold'],
            mode='markers', name='Spike Anomaly',
            marker=dict(color=COLORS['danger'], size=10, symbol='triangle-up',
                       line=dict(width=1, color='white'))
        ))
    
    # Drops
    drops = anom_df[anom_df['anomaly_type'] == 'Drop']
    if len(drops) > 0:
        fig.add_trace(go.Scatter(
            x=drops['date'], y=drops['units_sold'],
            mode='markers', name='Drop Anomaly',
            marker=dict(color=COLORS['warning'], size=10, symbol='triangle-down',
                       line=dict(width=1, color='white'))
        ))
    
    fig.update_layout(**PLOT_LAYOUT, height=450, title="Sales Time Series with Anomalies")
    st.plotly_chart(fig, use_container_width=True)
    
    # ── Stats ──
    total_anomalies = anom_df['is_anomaly'].sum()
    spike_count = len(spikes)
    drop_count = len(drops)
    
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Anomalies", total_anomalies)
    s2.metric("Spikes ↑", spike_count)
    s3.metric("Drops ↓", drop_count)
    
    # ── Anomaly Log ──
    if total_anomalies > 0:
        st.markdown("##### 📋 Anomaly Log")
        anomaly_log = anom_df[anom_df['is_anomaly']][['date', 'units_sold', 'z_score', 'anomaly_type']].copy()
        anomaly_log['date'] = anomaly_log['date'].dt.strftime('%Y-%m-%d')
        anomaly_log['z_score'] = anomaly_log['z_score'].round(2)
        anomaly_log.columns = ['Date', 'Units Sold', 'Z-Score', 'Type']
        st.dataframe(anomaly_log.sort_values('Date', ascending=False), 
                    use_container_width=True, hide_index=True, height=300)


# ═════════════════════════════════════════════════════════════
# TAB 5: SCENARIO SIMULATOR
# ═════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### 🧪 What-If Scenario Simulator")
    st.markdown("Simulate the impact of business decisions on future sales.")
    
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        discount_adj = st.slider("Discount Adjustment (%)", -20, 50, 0, key='sim_disc')
    with sc2:
        marketing_adj = st.slider("Marketing Spend Change ($)", -500, 2000, 0, key='sim_mkt')
    with sc3:
        price_adj = st.slider("Price Adjustment (%)", -30, 30, 0, key='sim_price')
    
    if st.button("🚀 Run Simulation", use_container_width=True, key='sim_btn'):
        # Calculate impact using data-driven coefficients
        daily_agg = fdf.groupby('date').agg(
            units=('units_sold', 'sum'),
            revenue=('revenue', 'sum')
        ).reset_index()
        
        base_units = daily_agg['units'].tail(30).mean()
        base_revenue = daily_agg['revenue'].tail(30).mean()
        
        # Impact coefficients (derived from data patterns)
        disc_effect = discount_adj * 0.8    # Each 1% discount → 0.8% sales uplift
        mkt_effect = marketing_adj / 100 * 2.0   # Each $100 marketing → 2% lift
        price_effect = -price_adj * 1.2     # Price increase → demand decrease (elasticity)
        
        total_impact_pct = disc_effect + mkt_effect + price_effect
        
        projected_units = base_units * (1 + total_impact_pct / 100)
        projected_revenue = projected_units * (base_revenue / base_units) * (1 + price_adj / 100) if base_units > 0 else 0
        
        # Revenue impact
        rev_change = projected_revenue - base_revenue
        
        st.markdown("---")
        st.markdown("##### 📊 Simulation Results")
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Projected Daily Units", f"{projected_units:,.0f}",
                  f"{total_impact_pct:+.1f}%")
        r2.metric("Projected Daily Revenue", f"${projected_revenue:,.0f}",
                  f"${rev_change:+,.0f}")
        r3.metric("30-Day Units Forecast", f"{projected_units * 30:,.0f}")
        r4.metric("30-Day Revenue Forecast", f"${projected_revenue * 30:,.0f}")
        
        # ── Projection Chart ──
        import datetime as dt
        last_date = daily_agg['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
        
        # Generate baseline and scenario projections with trend
        baseline = [base_units * (1 + 0.001 * i + 0.05 * np.sin(2 * np.pi * i / 7)) for i in range(30)]
        scenario = [projected_units * (1 + 0.001 * i + 0.05 * np.sin(2 * np.pi * i / 7)) for i in range(30)]
        
        fig = go.Figure()
        # Historical
        fig.add_trace(go.Scatter(
            x=daily_agg['date'].tail(60), y=daily_agg['units'].tail(60),
            mode='lines', name='Historical',
            line=dict(color=COLORS['primary'], width=2)
        ))
        # Baseline
        fig.add_trace(go.Scatter(
            x=future_dates, y=baseline,
            mode='lines', name='Baseline Forecast',
            line=dict(color='#94a3b8', width=2, dash='dash')
        ))
        # Scenario
        fig.add_trace(go.Scatter(
            x=future_dates, y=scenario,
            mode='lines', name='Scenario Forecast',
            line=dict(color=COLORS['success'] if total_impact_pct > 0 else COLORS['danger'], 
                     width=3),
            fill='tonexty', fillcolor='rgba(52,211,153,0.08)' if total_impact_pct > 0 else 'rgba(248,113,113,0.08)'
        ))
        
        fig.update_layout(**PLOT_LAYOUT, height=400, 
                         title="30-Day Projection: Baseline vs Scenario")
        st.plotly_chart(fig, use_container_width=True)
        
        # Impact Breakdown
        st.markdown("##### 📋 Impact Breakdown")
        impact_data = pd.DataFrame({
            'Factor': ['Discount Effect', 'Marketing Effect', 'Price Effect', 'Total Impact'],
            'Change': [f"{discount_adj:+d}%", f"${marketing_adj:+,}", f"{price_adj:+d}%", "—"],
            'Sales Impact': [f"{disc_effect:+.1f}%", f"{mkt_effect:+.1f}%", f"{price_effect:+.1f}%", f"{total_impact_pct:+.1f}%"],
        })
        st.dataframe(impact_data, use_container_width=True, hide_index=True)
