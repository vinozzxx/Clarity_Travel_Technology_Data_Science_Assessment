"""
Clarity Travel Technology Solutions — Data Science Assessment Dashboard
Professional Streamlit application structured around the assessment deliverables.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import re
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from rag_engine import get_rag
_ENV_GROQ_KEY = os.environ.get('GROQ_API_KEY', '')

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, RocCurveDisplay,
                             precision_score, recall_score, f1_score)
import xgboost as xgb

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Clarity TTS — DS Assessment",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0b0f1a; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0b0f1a 100%);
    border-right: 1px solid #1e2d45;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.85rem;
    color: #94a3b8;
    padding: 2px 0;
}
section[data-testid="stSidebar"] .stRadio label:hover { color: #e2e8f0; }

/* ── KPI / Metric Cards ──────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #141e30 0%, #1a2744 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 20px 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(59,130,246,0.15);
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.75rem;
    font-weight: 700;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    font-size: 0.78rem;
}

/* ── Headings ─────────────────────────────────────────────────────────────── */
h1 { color: #f1f5f9 !important; font-weight: 800 !important; letter-spacing: -0.02em; }
h2 { color: #cbd5e1 !important; font-weight: 700 !important;
     border-bottom: 1px solid #1e3a5f; padding-bottom: 10px; margin-top: 8px; }
h3 { color: #94a3b8 !important; font-weight: 600 !important; }

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
button[data-baseweb="tab"] { font-weight: 600; color: #64748b; font-size: 0.88rem; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
}

/* ── Risk Badges ─────────────────────────────────────────────────────────── */
.risk-high {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #991b1b;
    color: #fecaca; border-radius: 14px; padding: 24px;
    text-align: center; font-size: 1.6rem; font-weight: 800;
    box-shadow: 0 4px 24px rgba(239,68,68,0.25);
}
.risk-medium {
    background: linear-gradient(135deg, #431407, #78350f);
    border: 1px solid #92400e;
    color: #fde68a; border-radius: 14px; padding: 24px;
    text-align: center; font-size: 1.6rem; font-weight: 800;
    box-shadow: 0 4px 24px rgba(245,158,11,0.25);
}
.risk-low {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #166534;
    color: #bbf7d0; border-radius: 14px; padding: 24px;
    text-align: center; font-size: 1.6rem; font-weight: 800;
    box-shadow: 0 4px 24px rgba(34,197,94,0.25);
}

/* ── Info Cards ──────────────────────────────────────────────────────────── */
.info-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 14px; padding: 22px 20px; margin-bottom: 12px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.3);
    transition: border-color 0.2s ease;
}
.info-card:hover { border-color: #3b82f6; }
.info-card .card-icon { font-size: 2rem; margin-bottom: 8px; }
.info-card .card-title {
    color: #e2e8f0; font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;
}
.info-card .card-text { color: #94a3b8; font-size: 0.85rem; line-height: 1.6; }

/* ── Section Label ───────────────────────────────────────────────────────── */
.section-label {
    color: #475569; font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 16px 0 6px 0; padding-left: 4px;
}

/* ── Insight Box ─────────────────────────────────────────────────────────── */
.insight-box {
    background: linear-gradient(135deg, #0c1a2e, #132036);
    border-left: 4px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px; margin: 12px 0;
    color: #93c5fd; font-size: 0.86rem; line-height: 1.65;
}

/* ── Cat Badge ───────────────────────────────────────────────────────────── */
.cat-badge {
    display: inline-block; padding: 8px 20px;
    border-radius: 999px; font-weight: 700; font-size: 1rem;
    letter-spacing: 0.02em;
}

/* ── Severity ─────────────────────────────────────────────────────────────── */
.sev-high   { background:#450a0a; color:#fca5a5; border-radius:8px; padding:6px 14px; font-weight:700; }
.sev-medium { background:#431407; color:#fde68a; border-radius:8px; padding:6px 14px; font-weight:700; }
.sev-low    { background:#052e16; color:#86efac; border-radius:8px; padding:6px 14px; font-weight:700; }

/* ── Chat bubbles ─────────────────────────────────────────────────────────── */
.chat-q {
    background:#1e293b; border-left:3px solid #3b82f6;
    border-radius:0 10px 10px 0; padding:12px 16px; margin:10px 0;
}
.chat-a {
    background: linear-gradient(135deg,#0f2240,#0f172a);
    border-left:3px solid #22c55e;
    border-radius:0 10px 10px 0; padding:16px 18px; margin:10px 0;
}

/* ── Model badge ─────────────────────────────────────────────────────────── */
.model-selected {
    background: linear-gradient(135deg,#1e3a5f,#1a2744);
    border:1px solid #3b82f6; border-radius:10px; padding:16px 20px;
}

/* ── Footer ──────────────────────────────────────────────────────────────── */
.footer {
    text-align:center; color:#475569; font-size:0.75rem;
    border-top:1px solid #1e3a5f; padding-top:16px; margin-top:32px;
}

/* ── Tech badge ──────────────────────────────────────────────────────────── */
.tech-badge {
    display:inline-block; background:#1e293b; border:1px solid #334155;
    border-radius:8px; padding:6px 14px; margin:4px;
    color:#94a3b8; font-size:0.8rem; font-weight:600;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB DARK THEME DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
PLT_DARK = {
    'figure.facecolor': '#0f172a',
    'axes.facecolor':   '#0f172a',
    'text.color':       '#e2e8f0',
    'axes.labelcolor':  '#94a3b8',
    'xtick.color':      '#94a3b8',
    'ytick.color':      '#94a3b8',
    'axes.edgecolor':   '#1e3a5f',
    'grid.color':       '#1e293b',
    'grid.linestyle':   '--',
    'grid.alpha':       0.6,
    'axes.spines.top':  False,
    'axes.spines.right':False,
}
plt.rcParams.update(PLT_DARK)
sns.set_theme(style="dark", palette="muted")

ACCENT   = '#3b82f6'   # blue
SUCCESS  = '#22c55e'   # green
WARNING  = '#f59e0b'   # amber
DANGER   = '#ef4444'   # red
PURPLE   = '#a855f7'
CYAN     = '#06b6d4'
PALETTE  = [ACCENT, SUCCESS, WARNING, DANGER, PURPLE, CYAN,
            '#ec4899', '#f97316', '#14b8a6', '#8b5cf6']


# ══════════════════════════════════════════════════════════════════════════════
# NLP — COMPLAINT CATEGORIES
# ══════════════════════════════════════════════════════════════════════════════
CATEGORY_PATTERNS = {
    'Schedule / Delay': [
        r'\bdelay\w*\b', r'\bschedule\b', r'\bflight cancell\w*\b', r'\bdiverted\b',
        r'\bmissed\b', r'\bpostponed\b', r'\breschedul\w+\b', r'\blate\b',
    ],
    'Refund Issues': [
        r'\brefund\w*\b', r'\bmoney back\b', r'\breimburse\w*\b',
        r'\bno refund\b', r'\bnot refunded\b', r'\brefund pending\b',
    ],
    'Ticketing Issues': [
        r'\bticket\w*\b', r'\bPNR\b', r'\bbooking error\b', r'\bname change\b',
        r'\bincorrect\b', r'\bwrong\b', r'\bduplicate\b', r'\bnot issued\b',
    ],
    'Payment / Pricing': [
        r'\bprice\w*\b', r'\bovercharg\w+\b', r'\bhidden fee\b', r'\bfare\b',
        r'\bextra charge\b', r'\bcharged\b', r'\bpayment\b', r'\bbilling\b',
    ],
    'Baggage': [
        r'\bbaggage\b', r'\bluggage\b', r'\bbag\b', r'\bsuitcase\b',
        r'\blost bag\b', r'\bdamaged\b', r'\bmissing bag\b',
    ],
    'Seat Issues': [
        r'\bseat\b', r'\bseating\b', r'\bupgrade\b', r'\blegroom\b',
        r'\bassignment\b', r'\bno seat\b', r'\bseat change\b',
    ],
    'Visa / Documentation': [
        r'\bvisa\b', r'\bdocument\w*\b', r'\bpassport\b', r'\bimmigration\b',
        r'\bentry\b', r'\bdeniedd?\b', r'\bclearance\b',
    ],
    'Ancillary Services': [
        r'\bmeal\b', r'\bfood\b', r'\bwi-?fi\b', r'\bentertainment\b',
        r'\blounge\b', r'\bancillar\w*\b', r'\bspecial assistance\b',
    ],
    'Customer Service': [
        r'\brude\b', r'\bunhelpful\b', r'\bstaff\b', r'\bagent\b',
        r'\bno response\b', r'\bwait\w*\b', r'\bcustomer service\b',
    ],
}

CAT_COLOURS = {
    'Schedule / Delay':    ('#bee3f8', '#1e3a5f'),
    'Refund Issues':       ('#fecaca', '#450a0a'),
    'Ticketing Issues':    ('#fde68a', '#451a03'),
    'Payment / Pricing':   ('#bbf7d0', '#052e16'),
    'Baggage':             ('#e9d8fd', '#2e1065'),
    'Seat Issues':         ('#fce7f3', '#500724'),
    'Visa / Documentation':('#cffafe', '#083344'),
    'Ancillary Services':  ('#fed7aa', '#431407'),
    'Customer Service':    ('#f1f5f9', '#0f172a'),
    'Other':               ('#e2e8f0', '#1e293b'),
}

SEVERITY_MAP = {
    'Refund Issues':        'High',
    'Schedule / Delay':     'High',
    'Payment / Pricing':    'High',
    'Ticketing Issues':     'Medium',
    'Baggage':              'Medium',
    'Customer Service':     'Medium',
    'Seat Issues':          'Low',
    'Visa / Documentation': 'Low',
    'Ancillary Services':   'Low',
    'Other':                'Low',
}

ROUTING_MAP = {
    'Schedule / Delay':     ('🛫 Ops / Airline Liaison',      'Contact airline ops team to check IROPS status.'),
    'Refund Issues':        ('💰 Finance / Refunds Team',      'Escalate to finance for refund processing. Check fare rules.'),
    'Ticketing Issues':     ('🔧 NDC/GDS Tech Team',          'Investigate PNR via GDS terminal or NDC API logs.'),
    'Payment / Pricing':    ('📊 Revenue Management',          'Check if fare discrepancy is in airline price cache.'),
    'Baggage':              ('🧳 Airport Services',            'Raise PIR with originating airport. Track via WorldTracer.'),
    'Seat Issues':          ('🪑 Ground Services',             'Check seat availability and reassign via PSS.'),
    'Visa / Documentation': ('📋 Documentation Team',          'Verify travel documents and advise on entry requirements.'),
    'Ancillary Services':   ('🍽️ Inflight Services',          'Coordinate with airline catering/inflight services.'),
    'Customer Service':     ('👥 Agent Relations',             'Flag to agent quality team for coaching.'),
    'Other':                ('📩 General Support',             'Route to Tier 1 support for manual triage.'),
}


def classify_complaint(text: str) -> str:
    t = str(text).lower()
    for cat, pats in CATEGORY_PATTERNS.items():
        if any(re.search(p, t) for p in pats):
            return cat
    return 'Other'


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
_BASE_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    df = pd.read_csv(_BASE_DIR / 'data' / 'clarity_bookings_dataset.csv')
    df['booking_date']   = pd.to_datetime(df['booking_date'])
    df['departure_date'] = pd.to_datetime(df['departure_date'])

    df['fare_basis']            = df['fare_basis'].fillna('Unknown')
    df['ancillary_revenue_inr'] = df['ancillary_revenue_inr'].fillna(0)
    df['payment_method']        = df['payment_method'].fillna('Unknown')
    df['customer_complaint']    = df['customer_complaint'].fillna('')

    df['is_cancelled']           = df['booking_status'].isin(['Cancelled', 'Refunded']).astype(int)
    df['route']                  = df['origin'] + ' → ' + df['destination']
    df['booking_month']          = df['booking_date'].dt.month
    df['booking_month_name']     = df['booking_date'].dt.strftime('%b')
    df['booking_dayofweek']      = df['booking_date'].dt.dayofweek
    df['fare_per_passenger']     = df['total_fare_inr'] / df['pax_count']
    df['is_repeat_customer']     = (df['prior_bookings'] > 0).astype(int)
    df['has_ancillary']          = (df['ancillary_revenue_inr'] > 0).astype(int)
    route_avg                    = df.groupby('route')['total_fare_inr'].transform('mean')
    df['fare_to_avg_route_fare'] = df['total_fare_inr'] / route_avg
    df['tax_ratio']              = df['taxes_inr'] / df['total_fare_inr'].replace(0, np.nan)
    df['customer_type']          = np.where(df['prior_bookings'] > 0, 'Repeat', 'New')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Training ML models…")
def train_model(df):
    cat_feats = ['airline', 'cabin_class', 'trip_type',
                 'booking_channel', 'booking_source', 'haul_type']
    num_feats = ['lead_time_days', 'pax_count', 'base_fare_inr', 'taxes_inr',
                 'total_fare_inr', 'prior_bookings',
                 'fare_per_passenger', 'booking_month', 'booking_dayofweek',
                 'is_repeat_customer', 'has_ancillary',
                 'fare_to_avg_route_fare', 'tax_ratio']

    X = df[cat_feats + num_feats]
    y = df['is_cancelled']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                         ('scl', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='Unknown')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    pre = ColumnTransformer([('num', num_pipe, num_feats),
                             ('cat', cat_pipe, cat_feats)])

    lr = Pipeline([('pre', pre),
                   ('clf', LogisticRegression(class_weight='balanced',
                                              max_iter=1000, random_state=42))])
    lr.fit(X_train, y_train)

    xg = Pipeline([('pre', pre),
                   ('clf', xgb.XGBClassifier(n_estimators=300, max_depth=4,
                                              learning_rate=0.05, subsample=0.8,
                                              use_label_encoder=False,
                                              eval_metric='logloss', random_state=42))])
    xg.fit(X_train, y_train)

    # Metrics
    lr_proba = lr.predict_proba(X_test)[:, 1]
    xg_proba = xg.predict_proba(X_test)[:, 1]
    lr_pred  = lr.predict(X_test)
    xg_pred  = xg.predict(X_test)

    lr_auc = roc_auc_score(y_test, lr_proba)
    xg_auc = roc_auc_score(y_test, xg_proba)

    metrics = {
        'Logistic Regression': {
            'Precision': precision_score(y_test, lr_pred, zero_division=0),
            'Recall':    recall_score(y_test, lr_pred, zero_division=0),
            'F1 Score':  f1_score(y_test, lr_pred, zero_division=0),
            'ROC-AUC':   lr_auc,
        },
        'XGBoost': {
            'Precision': precision_score(y_test, xg_pred, zero_division=0),
            'Recall':    recall_score(y_test, xg_pred, zero_division=0),
            'F1 Score':  f1_score(y_test, xg_pred, zero_division=0),
            'ROC-AUC':   xg_auc,
        },
    }

    # Feature importance
    cat_enc   = xg.named_steps['pre'].transformers_[1][1].named_steps['ohe']
    cat_names = list(cat_enc.get_feature_names_out(cat_feats))
    all_names = num_feats + cat_names
    imps      = xg.named_steps['clf'].feature_importances_
    feat_df   = (pd.DataFrame({'Feature': all_names, 'Importance': imps})
                 .sort_values('Importance', ascending=False).head(15).reset_index(drop=True))

    cm_lr = confusion_matrix(y_test, lr_pred)
    cm_xg = confusion_matrix(y_test, xg_pred)

    return (lr, xg, lr_auc, xg_auc,
            X_test, y_test,
            lr_proba, xg_proba,
            lr_pred, xg_pred,
            feat_df, cat_feats, num_feats,
            metrics, cm_lr, cm_xg)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: INSIGHT BOX
# ══════════════════════════════════════════════════════════════════════════════
def insight_box(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        <span style="font-size:1.8rem;">✈️</span>
        <div>
            <div style="color:#f1f5f9;font-weight:800;font-size:1rem;line-height:1.2">Clarity TTS</div>
            <div style="color:#64748b;font-size:0.7rem;">DS Assessment Dashboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Navigation grouping
    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "🏠 Project Overview",
            "── Part 1: Revenue Analysis",
            "── Part 1: Cancellation Analysis",
            "── Part 1: Booking Trends",
            "── Part 1: Customer Behaviour",
            "── Part 2: Booking Risk Predictor",
            "── Part 2: Model Performance",
            "── Part 2: Feature Importance",
            "── Part 3: Live Complaint Classifier",
            "── Part 3: Batch Complaint Analysis",
            "── Part 3: AI Complaint Summariser",
            "🔍 AI Data Assistant",
            "ℹ️ About Project",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Python · Pandas · Scikit-learn · XGBoost · Streamlit · Groq Llama-3")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ══════════════════════════════════════════════════════════════════════════════
df = load_data()

(lr_model, xg_model, lr_auc, xg_auc,
 X_test, y_test,
 lr_proba, xg_proba,
 lr_pred, xg_pred,
 feat_df, cat_feats, num_feats,
 metrics, cm_lr, cm_xg) = train_model(df)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Project Overview":
    st.markdown("""
    <div style="margin-bottom:8px;">
        <div style="color:#64748b;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;">
            Data Science Assessment
        </div>
        <h1 style="margin:4px 0 2px 0;">Clarity Travel Technology Solutions</h1>
        <div style="color:#94a3b8;font-size:1.05rem;">Data Science Assessment Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    section_label("Dataset at a Glance")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Bookings",         f"{len(df):,}")
    k2.metric("Total Revenue",          f"₹{df['total_fare_inr'].sum()/1e7:.2f} Cr")
    k3.metric("Cancellation Rate",      f"{df['is_cancelled'].mean():.1%}")
    k4.metric("Avg Satisfaction Score", f"{df['satisfaction_score'].mean():.2f} / 5")

    st.divider()

    # ── Feature Cards ─────────────────────────────────────────────────────────
    section_label("What's Inside This Dashboard")
    fc1, fc2 = st.columns(2)

    with fc1:
        st.markdown("""
        <div class="info-card">
            <div class="card-icon">📊</div>
            <div class="card-title">Business Intelligence Dashboard</div>
            <div class="card-text">
                Explore revenue patterns, cancellation drivers, seasonal booking trends
                and customer behaviour across airlines, routes, cabin classes and booking channels.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <div class="card-icon">💬</div>
            <div class="card-title">Complaint Intelligence</div>
            <div class="card-text">
                Classify airline customer complaints into 9 operational categories using
                rule-based NLP. Generate AI-powered executive summaries using Groq Llama-3
                for rapid support team briefings.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with fc2:
        st.markdown("""
        <div class="info-card">
            <div class="card-icon">🤖</div>
            <div class="card-title">Cancellation Risk Prediction</div>
            <div class="card-text">
                Predict whether a booking is likely to be cancelled or refunded using
                Logistic Regression and XGBoost models trained on 2,000 Clarity bookings.
                ROC-AUC: <strong style="color:#3b82f6">""" + f"{xg_auc:.4f}" + """</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <div class="card-icon">🔍</div>
            <div class="card-title">AI Data Assistant</div>
            <div class="card-text">
                Ask natural-language questions about the Clarity booking dataset.
                Powered by a RAG pipeline: sentence-transformer embeddings + ChromaDB
                vector search + Groq Llama-3 generation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Model Performance Quick View ──────────────────────────────────────────
    section_label("Model Performance Summary")
    mc1, mc2 = st.columns(2)
    with mc1:
        perf_data = {
            'Model':     ['Logistic Regression', 'XGBoost'],
            'Precision': [f"{metrics['Logistic Regression']['Precision']:.4f}",
                          f"{metrics['XGBoost']['Precision']:.4f}"],
            'Recall':    [f"{metrics['Logistic Regression']['Recall']:.4f}",
                          f"{metrics['XGBoost']['Recall']:.4f}"],
            'F1 Score':  [f"{metrics['Logistic Regression']['F1 Score']:.4f}",
                          f"{metrics['XGBoost']['F1 Score']:.4f}"],
            'ROC-AUC':   [f"{lr_auc:.4f}", f"{xg_auc:.4f}"],
        }
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    with mc2:
        status_counts = df['booking_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        status_counts['Share'] = (status_counts['Count']/len(df)*100).map('{:.1f}%'.format)
        st.dataframe(status_counts, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — REVENUE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "── Part 1: Revenue Analysis":
    st.title("📊 Part 1 — Revenue Analysis")
    st.caption("Identifying which airlines, routes and cabin classes generate the highest revenue for Clarity TTS.")

    # Top KPIs
    airline_rev  = df.groupby('airline')['total_fare_inr'].sum()
    route_rev    = df.groupby('route')['total_fare_inr'].sum()
    cabin_rev    = df.groupby('cabin_class')['total_fare_inr'].sum()
    top_airline  = airline_rev.idxmax()
    top_route    = route_rev.idxmax()
    top_cabin    = cabin_rev.idxmax()

    section_label("Revenue KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Top Revenue Airline", top_airline,
              f"₹{airline_rev[top_airline]/1e6:.1f}M")
    k2.metric("Top Revenue Route",   top_route[:20]+"…" if len(top_route)>20 else top_route,
              f"₹{route_rev[top_route]/1e6:.1f}M")
    k3.metric("Top Revenue Cabin",   top_cabin,
              f"₹{cabin_rev[top_cabin]/1e6:.1f}M")
    k4.metric("Total Revenue",       f"₹{df['total_fare_inr'].sum()/1e7:.2f} Cr")

    st.divider()

    with st.expander("💡 What am I seeing?", expanded=True):
        st.markdown("""
        This section identifies which **airlines, routes and cabin classes** generate the highest revenue
        for Clarity TTS. Revenue analysis helps prioritise airline partnerships, optimise route selection,
        and understand which cabin classes drive the most value.

        - **Higher-revenue airlines** signal strong partnership value and should receive priority support.
        - **Top routes** indicate where demand is strongest and where premium fares are accepted.
        - **Cabin class breakdown** reveals how much of the revenue comes from premium vs economy travellers.
        """)

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    section_label("Revenue by Airline")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Airline Revenue Breakdown**")
        ar = airline_rev.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        colours = [ACCENT if i == ar.idxmax() else '#1e3a5f' for i in ar.index]
        bars = ax.barh(ar.index, ar.values/1e6, color=colours, edgecolor='none', height=0.6)
        ax.set_xlabel('Revenue (₹ Million)', color='#94a3b8')
        ax.set_title('Revenue by Airline', color='#e2e8f0', fontweight='bold', pad=12)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:.0f}M'))
        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Revenue by Cabin Class**")
        cabin_order = ['Economy', 'Premium Economy', 'Business', 'First']
        cr = cabin_rev.reindex(cabin_order).fillna(0)
        fig, ax = plt.subplots(figsize=(7, 5))
        clrs = [ACCENT, SUCCESS, WARNING, DANGER]
        bars = ax.bar(cr.index, cr.values/1e6, color=clrs, edgecolor='none', width=0.6)
        for bar, val in zip(bars, cr.values/1e6):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
                    f'₹{val:.1f}M', ha='center', va='bottom', color='#e2e8f0', fontsize=9)
        ax.set_ylabel('Revenue (₹ Million)', color='#94a3b8')
        ax.set_title('Revenue by Cabin Class', color='#e2e8f0', fontweight='bold', pad=12)
        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    section_label("Route & Distribution")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Top 10 Routes by Revenue**")
        rr = route_rev.sort_values(ascending=True).tail(10)
        fig, ax = plt.subplots(figsize=(7, 5))
        colours_r = [ACCENT if i == rr.index[-1] else '#1e3a5f' for i in rr.index]
        ax.barh(rr.index, rr.values/1e6, color=colours_r, edgecolor='none', height=0.6)
        ax.set_xlabel('Revenue (₹ Million)', color='#94a3b8')
        ax.set_title('Top 10 Routes by Revenue', color='#e2e8f0', fontweight='bold', pad=12)
        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown("**NDC vs GDS Revenue Share**")
        src_rev = df.groupby('booking_source')['total_fare_inr'].sum()
        fig, ax = plt.subplots(figsize=(5, 5))
        wedge_props = {'edgecolor': '#0f172a', 'linewidth': 3}
        ax.pie(src_rev.values, labels=src_rev.index,
               autopct='%1.1f%%', startangle=90,
               colors=[ACCENT, SUCCESS],
               wedgeprops=wedge_props,
               textprops={'color': '#e2e8f0'})
        ax.set_title('NDC vs GDS Revenue', color='#e2e8f0', fontweight='bold', pad=12)
        fig.patch.set_facecolor('#0f172a')
        st.pyplot(fig); plt.close()

    insight_box(
        "Business insight: Premium cabin classes (Business & First) likely drive disproportionate "
        "revenue relative to booking volume. Partnerships with airlines operating high-revenue routes "
        "should be prioritised for commercial negotiations."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CANCELLATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "── Part 1: Cancellation Analysis":
    st.title("❌ Part 1 — Cancellation Analysis")
    st.caption("Understanding where and why cancellations occur to minimise revenue leakage.")

    cancel_rate = df['is_cancelled'].mean() * 100
    ch_rates    = df.groupby('booking_channel')['is_cancelled'].mean()
    cab_rates   = df.groupby('cabin_class')['is_cancelled'].mean()

    section_label("Cancellation KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overall Cancellation Rate",    f"{cancel_rate:.2f}%")
    k2.metric("Highest Cancel Channel",       ch_rates.idxmax(),
              f"{ch_rates.max()*100:.1f}%")
    k3.metric("Highest Cancel Cabin",         cab_rates.idxmax(),
              f"{cab_rates.max()*100:.1f}%")
    k4.metric("Total Cancelled / Refunded",   f"{df['is_cancelled'].sum():,}")

    st.divider()

    with st.expander("💡 What am I seeing?", expanded=True):
        st.markdown("""
        Higher cancellation rates indicate potential **revenue leakage** — bookings that are made
        but never flown. Understanding cancellation patterns allows Clarity TTS to:

        - Flag high-risk bookings early for agent outreach
        - Apply deposit policies to high-risk channels or cabin classes
        - Negotiate cancellation fee structures with airlines
        """)

    st.divider()

    section_label("Cancellation by Booking Attributes")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Cancellation Rate by Booking Channel**")
        ch = ch_rates.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        clrs = [DANGER if v == ch.max() else '#1e3a5f' for v in ch.values]
        ax.barh(ch.index, ch.values*100, color=clrs, edgecolor='none', height=0.55)
        ax.axvline(cancel_rate, color=WARNING, linestyle='--', linewidth=1.5, label='Average')
        ax.set_xlabel('Cancellation Rate (%)', color='#94a3b8')
        ax.set_title('By Booking Channel', color='#e2e8f0', fontweight='bold', pad=12)
        ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='#94a3b8')
        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Cancellation Rate by Cabin Class**")
        cab = cab_rates.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        clrs = [DANGER if v == cab.max() else '#1e3a5f' for v in cab.values]
        ax.barh(cab.index, cab.values*100, color=clrs, edgecolor='none', height=0.55)
        ax.axvline(cancel_rate, color=WARNING, linestyle='--', linewidth=1.5, label='Average')
        ax.set_xlabel('Cancellation Rate (%)', color='#94a3b8')
        ax.set_title('By Cabin Class', color='#e2e8f0', fontweight='bold', pad=12)
        ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='#94a3b8')
        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Cancellation Rate by Trip Type**")
        tt = df.groupby('trip_type')['is_cancelled'].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(tt.index, tt.values*100, color=ACCENT, edgecolor='none', height=0.55)
        ax.axvline(cancel_rate, color=WARNING, linestyle='--', linewidth=1.5, label='Average')
        ax.set_xlabel('Cancellation Rate (%)', color='#94a3b8')
        ax.set_title('By Trip Type', color='#e2e8f0', fontweight='bold', pad=12)
        ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='#94a3b8')
        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown("**Lead Time vs Cancellation Rate**")
        df['lead_time_bin'] = pd.cut(df['lead_time_days'],
            bins=[0, 7, 30, 90, 180, 365],
            labels=['0–7d', '8–30d', '31–90d', '91–180d', '181d+'])
        lt_rates = df.groupby('lead_time_bin', observed=True)['is_cancelled'].mean() * 100
        clrs_lt  = [DANGER, WARNING, SUCCESS, ACCENT, PURPLE]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(lt_rates.index, lt_rates.values, color=clrs_lt, edgecolor='none', width=0.6)
        ax.axhline(cancel_rate, color=WARNING, linestyle='--', linewidth=1.5, label='Average')
        ax.set_ylabel('Cancellation Rate (%)', color='#94a3b8')
        ax.set_title('By Lead-Time Bucket', color='#e2e8f0', fontweight='bold', pad=12)
        ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='#94a3b8')
        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    insight_box(
        "Bookings made well in advance (high lead time) tend to have higher cancellation rates, "
        "as plans are more likely to change. Short lead-time bookings signal committed travellers "
        "and represent lower cancellation risk."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BOOKING TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "── Part 1: Booking Trends":
    st.title("📅 Part 1 — Booking Trends")
    st.caption("Seasonal demand patterns to support capacity planning and revenue management.")

    month_order = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    monthly_vol = df.groupby('booking_month_name').size().reindex(month_order).fillna(0)
    monthly_rev = df.groupby('booking_month_name')['total_fare_inr'].sum().reindex(month_order).fillna(0)

    section_label("Monthly KPIs")
    k1, k2, k3 = st.columns(3)
    k1.metric("Peak Booking Month",    monthly_vol.idxmax(),
              f"{int(monthly_vol.max()):,} bookings")
    k2.metric("Peak Revenue Month",    monthly_rev.idxmax(),
              f"₹{monthly_rev.max()/1e6:.1f}M")
    k3.metric("Avg Monthly Bookings",  f"{monthly_vol.mean():.0f}")

    st.divider()

    with st.expander("💡 What am I seeing?", expanded=True):
        st.markdown("""
        Seasonal demand patterns help Clarity TTS and its airline partners:
        - **Anticipate capacity requirements** during peak travel months
        - **Optimise pricing** strategies during high-demand periods
        - **Plan marketing campaigns** ahead of booking surges
        """)

    st.divider()

    section_label("Monthly Volume & Revenue")
    st.markdown("**Monthly Booking Volume**")
    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(len(month_order))
    ax.bar(month_order, monthly_vol.values, color=ACCENT, alpha=0.4, edgecolor='none', width=0.6)
    ax.plot(month_order, monthly_vol.values, 'o-', color=ACCENT, linewidth=2.5, markersize=6)
    ax.set_ylabel('Number of Bookings', color='#94a3b8')
    ax.set_title('Monthly Booking Volume — 2025', color='#e2e8f0', fontweight='bold', pad=12)
    fig.patch.set_facecolor('#0f172a')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("**Monthly Revenue Trend**")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(month_order, monthly_rev.values/1e6, alpha=0.25, color=SUCCESS)
    ax.plot(month_order, monthly_rev.values/1e6, 'o-', color=SUCCESS, linewidth=2.5, markersize=6)
    ax.set_ylabel('Revenue (₹ Million)', color='#94a3b8')
    ax.set_title('Monthly Revenue Trend — 2025', color='#e2e8f0', fontweight='bold', pad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:.0f}M'))
    fig.patch.set_facecolor('#0f172a')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    section_label("Channel Mix")
    st.markdown("**Booking Channel Mix by Month**")
    ch_month = df.groupby(['booking_month_name', 'booking_channel']).size().unstack(fill_value=0)
    ch_month = ch_month.reindex(month_order)
    ch_pct   = ch_month.div(ch_month.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ch_pct.plot(kind='bar', stacked=True, ax=ax,
                color=[ACCENT, SUCCESS, WARNING, DANGER],
                edgecolor='none', width=0.75)
    ax.set_xticklabels(month_order, rotation=0, fontsize=9)
    ax.set_ylabel('Share (%)', color='#94a3b8')
    ax.set_title('Booking Channel Mix by Month', color='#e2e8f0', fontweight='bold', pad=12)
    ax.legend(title='Channel', bbox_to_anchor=(1.01, 1), loc='upper left',
              facecolor='#1e293b', edgecolor='none', labelcolor='#e2e8f0',
              title_fontsize=9)
    fig.patch.set_facecolor('#0f172a')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    insight_box(
        "Consistent channel mix month-over-month suggests stable distribution partnerships. "
        "Shifts in channel share may indicate changes in agent or OTA commission structures "
        "or seasonal consumer preferences."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER BEHAVIOUR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "── Part 1: Customer Behaviour":
    st.title("👤 Part 1 — Customer Behaviour")
    st.caption("Comparing new and repeat customer value, loyalty and risk profiles.")

    summary = df.groupby('customer_type').agg(
        Bookings       = ('booking_id',       'count'),
        Avg_Fare       = ('total_fare_inr',   'mean'),
        Cancel_Rate    = ('is_cancelled',     'mean'),
        Avg_Sat        = ('satisfaction_score','mean'),
    ).round(2)
    summary['Cancel_Rate_Pct'] = (summary['Cancel_Rate'] * 100).round(2)

    section_label("New vs Repeat Customer KPIs")
    k1, k2, k3, k4 = st.columns(4)
    new_pct = summary.loc['New', 'Bookings'] / summary['Bookings'].sum() * 100
    k1.metric("New Customers",    f"{summary.loc['New','Bookings']:,}",
              f"{new_pct:.1f}% of bookings")
    k2.metric("Repeat Customers", f"{summary.loc['Repeat','Bookings']:,}",
              f"{100-new_pct:.1f}% of bookings")
    k3.metric("Repeat Avg Fare",  f"₹{summary.loc['Repeat','Avg_Fare']:,.0f}")
    k4.metric("New Avg Fare",     f"₹{summary.loc['New','Avg_Fare']:,.0f}")

    st.divider()

    with st.expander("💡 What am I seeing?", expanded=True):
        st.markdown("""
        Repeat customers are often **more valuable and loyal** — they tend to book higher fares,
        cancel less frequently, and report higher satisfaction scores. Understanding the gap between
        new and repeat customers helps Clarity TTS design **loyalty incentive programmes** and
        **targeted retention strategies**.
        """)

    st.divider()

    section_label("Comparison Table")
    display_summary = summary[['Bookings', 'Avg_Fare', 'Cancel_Rate_Pct', 'Avg_Sat']].copy()
    display_summary.columns = ['Bookings', 'Avg Fare (₹)', 'Cancel Rate (%)', 'Avg Satisfaction']
    display_summary['Avg Fare (₹)'] = display_summary['Avg Fare (₹)'].map('₹{:,.2f}'.format)
    st.dataframe(display_summary, use_container_width=True)

    st.divider()
    section_label("Visual Comparison")
    col1, col2, col3 = st.columns(3)
    palette = {'New': '#1e3a5f', 'Repeat': ACCENT}

    metrics_to_plot = [
        ('Avg_Fare',        'Avg Total Fare (₹)',      '₹{:.0f}'),
        ('Cancel_Rate_Pct', 'Cancellation Rate (%)',   '{:.2f}%'),
        ('Avg_Sat',         'Avg Satisfaction Score',  '{:.2f}'),
    ]
    for col, (metric, label, fmt) in zip([col1, col2, col3], metrics_to_plot):
        with col:
            fig, ax = plt.subplots(figsize=(4.5, 4))
            clrs = [palette.get(i, ACCENT) for i in summary.index]
            bars = ax.bar(summary.index, summary[metric], color=clrs, edgecolor='none', width=0.5)
            for bar, val in zip(bars, summary[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                        fmt.format(val), ha='center', va='bottom',
                        color='#e2e8f0', fontsize=8.5)
            ax.set_title(label, color='#e2e8f0', fontsize=10, fontweight='bold', pad=10)
            fig.patch.set_facecolor('#0f172a')
            ax.set_ylim(0, summary[metric].max() * 1.2)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    insight_box(
        "Repeat customers with prior bookings > 0 show lower cancellation rates, indicating "
        "higher commitment at the time of booking. Consider incentivising repeat bookings through "
        "loyalty schemes or preferential channel access."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — BOOKING RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "── Part 2: Booking Risk Predictor":
    st.title("🤖 Part 2 — Booking Risk Predictor")
    st.caption("Enter booking details to predict the probability of cancellation or refund.")
    st.divider()

    col_inp, col_out = st.columns([1.4, 1], gap="large")

    with col_inp:
        section_label("Booking Details")

        c1, c2 = st.columns(2)
        airline     = c1.selectbox("Airline",      sorted(df['airline'].unique()))
        cabin_class = c2.selectbox("Cabin Class",  ['Economy', 'Premium Economy', 'Business', 'First'])

        c3, c4 = st.columns(2)
        trip_type = c3.selectbox("Trip Type",  ['One-Way', 'Round-Trip', 'Multi-City'])
        haul_type = c4.selectbox("Haul Type",  ['Long-Haul', 'Medium-Haul'])

        c5, c6 = st.columns(2)
        booking_channel = c5.selectbox("Booking Channel", ['B2B_Agent', 'B2C_Website', 'API_Partner', 'Mobile_App'])
        booking_source  = c6.selectbox("Booking Source",  ['NDC', 'GDS'])

        c7, c8, c9 = st.columns(3)
        lead_time      = c7.number_input("Lead Time (Days)", 0, 365, 30)
        pax_count      = c8.number_input("Passengers",       1, 9,   2)
        prior_bookings = c9.number_input("Prior Bookings",   0, 100, 0)

        c10, c11 = st.columns(2)
        base_fare = c10.number_input("Base Fare (₹)",  1000, 2000000, 25000)
        taxes     = c11.number_input("Taxes (₹)",      0,    500000,  5000)

        total_fare        = (base_fare + taxes) * pax_count
        booking_month_sel = st.slider("Booking Month", 1, 12, 6,
                                      format="Month %d")

        st.metric("Calculated Total Fare", f"₹{total_fare:,.0f}")

    with col_out:
        section_label("Risk Assessment")

        route_avg_global = df['total_fare_inr'].mean()
        f2r = total_fare / route_avg_global if route_avg_global else 1.0

        input_row = pd.DataFrame([{
            'airline': airline, 'cabin_class': cabin_class,
            'trip_type': trip_type, 'booking_channel': booking_channel,
            'booking_source': booking_source, 'haul_type': haul_type,
            'lead_time_days': lead_time, 'pax_count': pax_count,
            'base_fare_inr': base_fare, 'taxes_inr': taxes,
            'total_fare_inr': total_fare, 'prior_bookings': prior_bookings,
            'fare_per_passenger': total_fare / pax_count if pax_count else total_fare,
            'booking_month': booking_month_sel, 'booking_dayofweek': 1,
            'is_repeat_customer': int(prior_bookings > 0),
            'has_ancillary': 0,
            'fare_to_avg_route_fare': f2r,
            'tax_ratio': taxes / total_fare if total_fare else 0,
        }])

        prob     = xg_model.predict_proba(input_row)[0][1]
        prob_pct = prob * 100

        if prob_pct >= 60:
            risk_label, risk_css = "🔴 HIGH RISK", "risk-high"
            action = "⚠️ Flag for proactive agent outreach. Consider mandatory deposit."
        elif prob_pct >= 35:
            risk_label, risk_css = "🟡 MEDIUM RISK", "risk-medium"
            action = "ℹ️ Monitor booking. Send automated confirmation reminder."
        else:
            risk_label, risk_css = "🟢 LOW RISK", "risk-low"
            action = "✅ Standard processing. No intervention required."

        st.markdown(f"""
        <div class="{risk_css}">
            {risk_label}<br>
            <span style="font-size:2.4rem">{prob_pct:.1f}%</span><br>
            <span style="font-size:0.8rem;opacity:0.8">Cancellation Probability</span>
        </div>
        """, unsafe_allow_html=True)

        # Gauge bar
        fig, ax = plt.subplots(figsize=(5, 0.7))
        ax.barh([0], [100], color='#1e293b', height=0.5)
        colour = DANGER if prob_pct >= 60 else WARNING if prob_pct >= 35 else SUCCESS
        ax.barh([0], [prob_pct], color=colour, height=0.5)
        ax.set_xlim(0, 100); ax.axis('off')
        fig.patch.set_facecolor('#0f172a')
        st.pyplot(fig); plt.close()

        st.markdown(f"**Recommended Action:** {action}")
        st.divider()

        # Why explanation
        section_label("Why This Score?")
        lead_impact   = "High" if lead_time > 90 else "Medium" if lead_time > 30 else "Low"
        fare_impact   = "High" if total_fare > 100000 else "Medium" if total_fare > 30000 else "Low"
        hist_impact   = "Low (loyal customer)" if prior_bookings > 3 else "Medium" if prior_bookings > 0 else "High (new customer)"
        channel_cancel = df[df['booking_channel'] == booking_channel]['is_cancelled'].mean() * 100

        st.markdown(f"""
        | Factor | Value | Risk Impact |
        |--------|-------|-------------|
        | ⏱️ Lead Time | {lead_time} days | {lead_impact} |
        | 💰 Total Fare | ₹{total_fare:,.0f} | {fare_impact} |
        | 📚 Customer History | {prior_bookings} prior bookings | {hist_impact} |
        | 📡 Channel Avg Cancel | {booking_channel} | {channel_cancel:.1f}% avg |
        """)

        st.divider()

        # Model comparison
        section_label("Model Comparison")
        lr_prob = lr_model.predict_proba(input_row)[0][1] * 100
        comp_df = pd.DataFrame({
            'Model': ['Logistic Regression', 'XGBoost (Selected)'],
            'Cancellation Risk': [f"{lr_prob:.1f}%", f"{prob_pct:.1f}%"]
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "── Part 2: Model Performance":
    st.title("📈 Part 2 — Model Performance")
    st.caption("Comparative evaluation of Logistic Regression and XGBoost on the cancellation prediction task.")

    section_label("Model Comparison")
    perf_rows = []
    for mname, mvals in metrics.items():
        perf_rows.append({
            'Model':     mname,
            'Precision': f"{mvals['Precision']:.4f}",
            'Recall':    f"{mvals['Recall']:.4f}",
            'F1 Score':  f"{mvals['F1 Score']:.4f}",
            'ROC-AUC':   f"{mvals['ROC-AUC']:.4f}",
        })
    st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Selected model callout
    selected_model = 'XGBoost' if xg_auc >= lr_auc else 'Logistic Regression'
    st.markdown(f"""
    <div class="model-selected">
        <div style="color:#64748b;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">
            ✅ Selected Model for Production
        </div>
        <div style="color:#f1f5f9;font-size:1.2rem;font-weight:700;margin:6px 0 4px 0;">
            {selected_model}
        </div>
        <div style="color:#94a3b8;font-size:0.85rem;">
            Highest ROC-AUC ({max(xg_auc, lr_auc):.4f}) — best at distinguishing cancelled
            from non-cancelled bookings. Strong recall ensures high-risk bookings are
            captured rather than missed.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    section_label("Confusion Matrix & ROC Curve")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**XGBoost Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_xg, annot=True, fmt='d', cmap='Blues', ax=ax,
                    linewidths=0.5,
                    xticklabels=['Not Cancelled', 'Cancelled'],
                    yticklabels=['Not Cancelled', 'Cancelled'],
                    cbar=False,
                    annot_kws={'size': 13, 'color': '#e2e8f0'})
        ax.set_xlabel('Predicted Label', color='#94a3b8')
        ax.set_ylabel('True Label', color='#94a3b8')
        ax.set_title('XGBoost Confusion Matrix', color='#e2e8f0', fontweight='bold', pad=12)
        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**ROC Curve — Both Models**")
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, xg_proba, ax=ax,
                                          name=f'XGBoost (AUC={xg_auc:.3f})',
                                          color=ACCENT)
        RocCurveDisplay.from_predictions(y_test, lr_proba, ax=ax,
                                          name=f'Log Reg (AUC={lr_auc:.3f})',
                                          color=SUCCESS)
        ax.plot([0,1],[0,1], 'k--', lw=1, color='#475569')
        ax.set_title('ROC Curve', color='#e2e8f0', fontweight='bold', pad=12)
        ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='#e2e8f0', fontsize=8)
        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    insight_box(
        f"XGBoost achieves a ROC-AUC of {xg_auc:.4f}, indicating strong discriminative power. "
        "A random classifier scores 0.5, so the model provides meaningful predictive signal "
        "well above baseline — useful for proactive booking management."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "── Part 2: Feature Importance":
    st.title("🔬 Part 2 — Feature Importance")
    st.caption("Understanding which booking attributes most influence cancellation predictions.")

    section_label("Top Predictive Features (XGBoost)")
    col1, col2 = st.columns([1.3, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        fi = feat_df.sort_values('Importance', ascending=True)
        clrs = [ACCENT if i == len(fi)-1 else
                SUCCESS if i >= len(fi)-4 else '#1e3a5f'
                for i in range(len(fi))]
        ax.barh(fi['Feature'], fi['Importance'], color=clrs, edgecolor='none', height=0.65)
        ax.set_xlabel('Feature Importance (Gain)', color='#94a3b8')
        ax.set_title('XGBoost Feature Importance (Top 15)', color='#e2e8f0', fontweight='bold', pad=12)
        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        section_label("Business Interpretation")
        interpretations = [
            ("lead_time_days",           "Lead Time",           "Days between booking and departure. Longer lead time → higher cancellation risk as plans change."),
            ("total_fare_inr",           "Total Fare",          "Higher-value bookings may face more scrutiny or financial pressure. Can cut both ways."),
            ("fare_to_avg_route_fare",   "Fare vs Route Avg",   "Booking priced above route average may indicate premium travellers who cancel less."),
            ("tax_ratio",                "Tax Ratio",           "Proportion of total fare that is taxes. High ratio may indicate complex itineraries."),
            ("base_fare_inr",            "Base Fare",           "Core ticket cost. Reflects the airline pricing tier and distance."),
        ]
        for feat, label, interp in interpretations:
            rank = feat_df[feat_df['Feature'] == feat].index
            importance_val = feat_df[feat_df['Feature'] == feat]['Importance'].values
            imp_str = f"{importance_val[0]:.4f}" if len(importance_val) > 0 else "—"
            with st.expander(f"**{label}** — Importance: {imp_str}"):
                st.markdown(f"<span style='color:#94a3b8'>{interp}</span>",
                            unsafe_allow_html=True)

    st.divider()

    section_label("Full Feature Importance Table")
    display_fi = feat_df.copy()
    display_fi['Importance'] = display_fi['Importance'].map('{:.6f}'.format)
    display_fi.index = range(1, len(display_fi)+1)
    st.dataframe(display_fi, use_container_width=True)

    insight_box(
        "Lead time is consistently the strongest predictor of cancellation risk. "
        "Fare-related features (total fare, base fare, tax ratio) collectively contribute "
        "significant predictive power, suggesting that pricing context matters as much as "
        "booking timing."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — LIVE COMPLAINT CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "── Part 3: Live Complaint Classifier":
    st.title("💬 Part 3 — Live Complaint Classifier")
    st.caption("Classify airline customer complaints into operational categories in real time.")

    st.markdown("""
    This tool automatically categorises airline customer complaints into **9 operational categories**
    using rule-based NLP and Regex pattern matching — enabling support teams to route issues instantly.
    """)

    col_desc, col_cats = st.columns([1, 1])
    with col_desc:
        st.markdown("""
        **Current Classification Method:**
        - ✅ Rule-Based NLP
        - ✅ Regex Pattern Matching
        """)
    with col_cats:
        cats_html = " ".join([
            f'<span class="tech-badge">{c}</span>'
            for c in CATEGORY_PATTERNS.keys()
        ])
        st.markdown(f"**Supported Categories:**<br>{cats_html}", unsafe_allow_html=True)

    st.divider()

    user_text = st.text_area(
        "Enter complaint text:",
        placeholder="e.g. My flight was delayed by 4 hours and I still haven't received my refund...",
        height=130,
    )

    if st.button("🔍 Classify Complaint", type="primary") and user_text.strip():
        category = classify_complaint(user_text)
        bg, fg   = CAT_COLOURS.get(category, ('#e2e8f0', '#1e293b'))
        severity = SEVERITY_MAP.get(category, 'Low')
        team, action = ROUTING_MAP.get(category, ('📩 General Support', 'Manual review required.'))

        sev_css = {'High': 'sev-high', 'Medium': 'sev-medium', 'Low': 'sev-low'}.get(severity, 'sev-low')

        st.markdown(f"""
        <div style="background:{bg};color:{fg};border-radius:14px;
                    padding:24px;margin:16px 0;">
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.1em;opacity:0.7;margin-bottom:6px">
                Detected Category
            </div>
            <div style="font-size:1.9rem;font-weight:800;margin-bottom:8px">{category}</div>
            <span class="{sev_css}">⚡ {severity} Severity</span>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Suggested Routing → {team}**\n\n{action}")
        with col_b:
            matched = []
            for cat, pats in CATEGORY_PATTERNS.items():
                for p in pats:
                    m = re.search(p, user_text.lower())
                    if m:
                        matched.append(m.group())
            if matched:
                kw_html = " ".join([f'<span class="tech-badge" style="color:#3b82f6">{kw}</span>'
                                     for kw in set(matched)])
                st.markdown(f"**Matched Keywords:**<br>{kw_html}", unsafe_allow_html=True)
            else:
                st.caption("No specific keywords matched — classified as Other.")

    st.divider()
    with st.expander("📋 Category Reference Guide"):
        ref_data = []
        for cat in CATEGORY_PATTERNS:
            sev  = SEVERITY_MAP.get(cat, 'Low')
            team = ROUTING_MAP.get(cat, ('—', '—'))[0]
            ref_data.append({'Category': cat, 'Severity': sev, 'Routing Team': team})
        ref_data.append({'Category': 'Other', 'Severity': 'Low', 'Routing Team': '📩 General Support'})
        st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — BATCH COMPLAINT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "── Part 3: Batch Complaint Analysis":
    st.title("📋 Part 3 — Batch Complaint Analysis")
    st.caption("Dataset-wide classification of all customer complaints to identify trends and hot spots.")

    cdf = df[df['customer_complaint'].str.len() > 0].copy()
    cdf['category'] = cdf['customer_complaint'].apply(classify_complaint)
    cat_counts = cdf['category'].value_counts()
    top_cat    = cat_counts.idxmax()
    top_airline_complaint = cdf.groupby('airline').size().idxmax()
    top_channel_complaint = cdf.groupby('booking_channel').size().idxmax()

    section_label("Complaint KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Complaints",       f"{len(cdf):,}")
    k2.metric("Top Category",           top_cat, f"{cat_counts[top_cat]:,} complaints")
    k3.metric("Most Affected Airline",  top_airline_complaint)
    k4.metric("Most Affected Channel",  top_channel_complaint)

    st.divider()

    section_label("Complaint Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Complaint Category Count**")
        cc = cat_counts.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        clrs_cc = [DANGER if i == cc.index[-1] else '#1e3a5f' for i in cc.index]
        ax.barh(cc.index, cc.values, color=clrs_cc, edgecolor='none', height=0.65)
        ax.set_xlabel('Number of Complaints', color='#94a3b8')
        ax.set_title('Complaints by Category', color='#e2e8f0', fontweight='bold', pad=12)
        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Complaints vs Booking Status**")
        ct_status = pd.crosstab(cdf['category'], cdf['booking_status'], normalize='columns') * 100
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(ct_status, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    linewidths=0.4, cbar=False,
                    annot_kws={'size': 8})
        ax.set_title('Category × Booking Status (%)', color='#e2e8f0', fontweight='bold', pad=12)
        ax.tick_params(axis='x', rotation=30, labelcolor='#94a3b8')
        ax.tick_params(axis='y', labelcolor='#94a3b8')
        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    section_label("Airline Breakdown")
    st.markdown("**Complaint Category vs Airline (%)**")
    ct_airline = pd.crosstab(cdf['category'], cdf['airline'], normalize='columns') * 100
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(ct_airline, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                linewidths=0.3, cbar=False,
                annot_kws={'size': 8})
    ax.set_title('Complaint Category by Airline (%)', color='#e2e8f0', fontweight='bold', pad=12)
    ax.tick_params(axis='x', rotation=30, labelcolor='#94a3b8')
    ax.tick_params(axis='y', labelcolor='#94a3b8')
    fig.patch.set_facecolor('#0f172a')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    section_label("All Categorised Complaints")
    display_cdf = cdf[['booking_id', 'airline', 'booking_channel',
                        'booking_status', 'total_fare_inr',
                        'customer_complaint', 'category']].copy()
    display_cdf['total_fare_inr'] = display_cdf['total_fare_inr'].map('₹{:,.0f}'.format)
    display_cdf.columns = ['Booking ID', 'Airline', 'Channel', 'Status',
                           'Fare', 'Complaint', 'Category']
    st.dataframe(display_cdf, use_container_width=True, hide_index=True, height=380)

    insight_box(
        "Concentrations of Schedule/Delay complaints in cancelled bookings may indicate "
        "a causal relationship — airlines with higher delay rates drive higher cancellations. "
        "This warrants deeper investigation with the airline operations teams."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 11 — AI COMPLAINT SUMMARISER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "── Part 3: AI Complaint Summariser":
    st.title("🧠 Part 3 — AI Complaint Summariser")
    st.caption("Generate an executive-level support summary from all complaints using Groq Llama-3.")

    if not _ENV_GROQ_KEY or not _ENV_GROQ_KEY.startswith("gsk_"):
        st.warning(
            "⚠️ Groq API key not configured. Set `GROQ_API_KEY` in your `.env` file.  \n"
            "Get a free key at [console.groq.com](https://console.groq.com)"
        )
        st.stop()

    cdf = df[df['customer_complaint'].str.len() > 0].copy()
    cdf['category'] = cdf['customer_complaint'].apply(classify_complaint)

    st.markdown(f"""
    <div class="info-card">
        <div class="card-icon">📊</div>
        <div class="card-title">Ready to Summarise</div>
        <div class="card-text">
            <strong style="color:#e2e8f0">{len(cdf):,}</strong> customer complaints identified
            across <strong style="color:#e2e8f0">{cdf['category'].nunique()}</strong> categories.
            Groq Llama-3 will analyse these and generate an executive summary with recommended actions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("🤖 Generate Support Summary", type="primary", use_container_width=True):
        with st.spinner("🧠 Groq Llama-3 is analysing all complaints and generating executive summary…"):
            try:
                from groq import Groq
                client = Groq(api_key=_ENV_GROQ_KEY)

                # Build context
                cat_summary = cdf['category'].value_counts().to_dict()
                airline_summary = cdf.groupby('airline')['category'].value_counts().unstack(fill_value=0).to_string()
                sample_complaints = cdf.sample(min(30, len(cdf)), random_state=42)['customer_complaint'].tolist()
                complaints_text = "\n".join([f"- {c}" for c in sample_complaints])

                prompt = f"""You are a senior customer experience analyst at Clarity Travel Technology Solutions.

Here is a summary of {len(cdf)} customer complaints from our airline booking dataset:

COMPLAINT CATEGORY DISTRIBUTION:
{cat_summary}

COMPLAINTS BY AIRLINE:
{airline_summary}

SAMPLE COMPLAINTS (30 of {len(cdf)}):
{complaints_text}

Please provide an EXECUTIVE SUPPORT SUMMARY with the following sections:
1. TOP ISSUES — The 3 most critical complaint themes and their business impact
2. URGENCY LEVEL — Overall urgency assessment (Critical/High/Medium/Low)
3. RECOMMENDED ACTIONS — 5 specific, actionable recommendations for the support team
4. PRIORITY RANKING — Which issues to address first and why
5. SUGGESTED TEAM ACTIONS — Which teams should be involved and what they should do

Format your response clearly with headers. Be specific, data-driven and business-focused."""

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system",
                         "content": "You are a senior customer experience analyst generating executive briefings."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=2048,
                )
                summary_text = response.choices[0].message.content

                st.markdown("""
                <div style="background:linear-gradient(135deg,#0f172a,#1e293b);
                            border:1px solid #1e3a5f; border-radius:14px; padding:28px;
                            margin-top:16px;">
                    <div style="color:#64748b;font-size:0.72rem;font-weight:700;
                                text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px">
                        🤖 AI Executive Summary — Groq Llama-3
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<div style='color:#e2e8f0;line-height:1.8;'>{summary_text.replace(chr(10), '<br>')}</div>",
                            unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error generating summary: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 12 — AI DATA ASSISTANT  (Metrics → Groq, no ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "\U0001f50d AI Data Assistant":
    st.title("\U0001f50d AI Data Assistant")
    st.markdown("""
    Ask natural-language questions about the Clarity booking dataset.
    The assistant analyses **pre-computed business metrics** from all 2,000 bookings
    and uses **Groq Llama-3** to generate structured, data-driven insights.

    \U0001f4a1 The assistant answers **only from the Clarity booking dataset** and does not use external information.
    """)
    st.divider()

    if not _ENV_GROQ_KEY or not _ENV_GROQ_KEY.startswith("gsk_"):
        st.warning(
            "\u26a0\ufe0f Please set your **`GROQ_API_KEY`** in the `.env` file to use this feature.  \n"
            "Get a free key at [console.groq.com](https://console.groq.com)"
        )
        st.info(
            "**How it works:** Key metrics are pre-computed from the entire 2,000-row dataset "
            "and passed as context to Groq Llama-3, which generates structured business answers "
            "with insights and recommendations \u2014 no vector database required."
        )
        st.stop()

    @st.cache_data
    def build_dataset_context(_df):
        lines = []
        sep = "=" * 60
        lines.append(sep)
        lines.append("CLARITY BOOKINGS DATASET \u2014 FULL BUSINESS METRICS CONTEXT")
        lines.append(f"Total bookings: {len(_df):,}")
        lines.append(f"Date range: {_df['booking_date'].min().date()} to {_df['booking_date'].max().date()}")
        lines.append(f"Airlines: {', '.join(sorted(_df['airline'].unique()))}")
        lines.append(sep)

        # Revenue
        lines.append("\n-- REVENUE --")
        lines.append(f"Total revenue: INR {_df['total_fare_inr'].sum():,.0f}")
        lines.append(f"Average fare per booking: INR {_df['total_fare_inr'].mean():,.0f}")
        lines.append(f"Median fare: INR {_df['total_fare_inr'].median():,.0f}")

        ar = _df.groupby('airline')['total_fare_inr'].sum().sort_values(ascending=False)
        lines.append("Revenue by airline:")
        for a, v in ar.items():
            lines.append(f"  {a}: INR {v:,.0f}")

        cabin_order = ['Economy', 'Premium Economy', 'Business', 'First']
        cr = _df.groupby('cabin_class')['total_fare_inr'].sum().reindex(cabin_order)
        lines.append("Revenue by cabin class:")
        for c, v in cr.items():
            lines.append(f"  {c}: INR {v:,.0f}")

        src_rev = _df.groupby('booking_source')['total_fare_inr'].sum()
        lines.append("Revenue by booking source (NDC vs GDS):")
        for s, v in src_rev.items():
            lines.append(f"  {s}: INR {v:,.0f}")

        ch_rev = _df.groupby('booking_channel')['total_fare_inr'].sum().sort_values(ascending=False)
        lines.append("Revenue by booking channel:")
        for ch, v in ch_rev.items():
            lines.append(f"  {ch}: INR {v:,.0f}")

        rr = _df.groupby('route')['total_fare_inr'].sum().sort_values(ascending=False).head(10)
        lines.append("Top 10 routes by revenue:")
        for route, v in rr.items():
            lines.append(f"  {route}: INR {v:,.0f}")

        # Cancellations
        lines.append("\n-- CANCELLATIONS --")
        overall_cancel = _df['is_cancelled'].mean() * 100
        lines.append(f"Overall cancellation/refund rate: {overall_cancel:.2f}%")
        lines.append(f"Total cancelled/refunded: {_df['is_cancelled'].sum():,} of {len(_df):,}")

        status_counts = _df['booking_status'].value_counts()
        lines.append("Booking status breakdown:")
        for status, cnt in status_counts.items():
            lines.append(f"  {status}: {cnt:,} ({cnt/len(_df)*100:.1f}%)")

        ch_cancel = _df.groupby('booking_channel')['is_cancelled'].mean().sort_values(ascending=False)
        lines.append("Cancellation rate by booking channel:")
        for ch, rate in ch_cancel.items():
            lines.append(f"  {ch}: {rate*100:.2f}%")

        cab_cancel = _df.groupby('cabin_class')['is_cancelled'].mean().sort_values(ascending=False)
        lines.append("Cancellation rate by cabin class:")
        for cab, rate in cab_cancel.items():
            lines.append(f"  {cab}: {rate*100:.2f}%")

        tt_cancel = _df.groupby('trip_type')['is_cancelled'].mean().sort_values(ascending=False)
        lines.append("Cancellation rate by trip type:")
        for tt, rate in tt_cancel.items():
            lines.append(f"  {tt}: {rate*100:.2f}%")

        al_cancel = _df.groupby('airline')['is_cancelled'].mean().sort_values(ascending=False)
        lines.append("Cancellation rate by airline:")
        for al, rate in al_cancel.items():
            lines.append(f"  {al}: {rate*100:.2f}%")

        # Booking trends
        lines.append("\n-- BOOKING TRENDS --")
        month_order = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        mon_vol = _df.groupby('booking_month_name').size().reindex(month_order).fillna(0)
        lines.append("Monthly booking volume:")
        for m, v in mon_vol.items():
            lines.append(f"  {m}: {int(v):,}")

        mon_rev = _df.groupby('booking_month_name')['total_fare_inr'].sum().reindex(month_order).fillna(0)
        lines.append("Monthly revenue:")
        for m, v in mon_rev.items():
            lines.append(f"  {m}: INR {v:,.0f}")

        # Routes
        lines.append("\n-- ROUTE POPULARITY --")
        route_vol = _df.groupby('route').size().sort_values(ascending=False).head(10)
        lines.append("Top 10 routes by booking volume:")
        for route, cnt in route_vol.items():
            lines.append(f"  {route}: {cnt:,} bookings")

        # Customer behaviour
        lines.append("\n-- CUSTOMER BEHAVIOUR --")
        cust = _df.groupby('customer_type').agg(
            Count       =('booking_id',       'count'),
            Avg_Fare    =('total_fare_inr',   'mean'),
            Cancel_Rate =('is_cancelled',     'mean'),
            Avg_Sat     =('satisfaction_score','mean'),
        ).round(2)
        lines.append("New vs Repeat customer comparison:")
        for ctype, row in cust.iterrows():
            lines.append(
                f"  {ctype}: {int(row['Count']):,} bookings | "
                f"Avg fare INR {row['Avg_Fare']:,.0f} | "
                f"Cancel rate {row['Cancel_Rate']*100:.1f}% | "
                f"Avg satisfaction {row['Avg_Sat']:.2f}"
            )

        # Satisfaction
        lines.append("\n-- SATISFACTION --")
        lines.append(f"Overall avg satisfaction: {_df['satisfaction_score'].mean():.2f} / 5")
        sat_al = _df.groupby('airline')['satisfaction_score'].mean().sort_values(ascending=False)
        lines.append("Average satisfaction by airline:")
        for al, sat in sat_al.items():
            lines.append(f"  {al}: {sat:.2f}")

        # Complaints
        lines.append("\n-- COMPLAINTS --")
        cdf = _df[_df['customer_complaint'].str.len() > 0].copy()
        cdf['category'] = cdf['customer_complaint'].apply(classify_complaint)
        lines.append(f"Total bookings with complaints: {len(cdf):,} ({len(cdf)/len(_df)*100:.1f}%)")

        cat_counts = cdf['category'].value_counts()
        lines.append("Complaint categories:")
        for cat, cnt in cat_counts.items():
            lines.append(f"  {cat}: {cnt:,}")

        al_complaints = cdf.groupby('airline').size().sort_values(ascending=False)
        lines.append("Complaints by airline:")
        for al, cnt in al_complaints.items():
            lines.append(f"  {al}: {cnt:,}")

        ch_complaints = cdf.groupby('booking_channel').size().sort_values(ascending=False)
        lines.append("Complaints by channel:")
        for ch, cnt in ch_complaints.items():
            lines.append(f"  {ch}: {cnt:,}")

        # Channel performance
        lines.append("\n-- CHANNEL PERFORMANCE --")
        ch_stats = _df.groupby('booking_channel').agg(
            Bookings    =('booking_id',     'count'),
            Revenue     =('total_fare_inr', 'sum'),
            Avg_Fare    =('total_fare_inr', 'mean'),
            Cancel_Rate =('is_cancelled',  'mean'),
        )
        lines.append("Booking channel performance summary:")
        for ch, row in ch_stats.iterrows():
            lines.append(
                f"  {ch}: {int(row['Bookings']):,} bookings | "
                f"Revenue INR {row['Revenue']:,.0f} | "
                f"Avg fare INR {row['Avg_Fare']:,.0f} | "
                f"Cancel rate {row['Cancel_Rate']*100:.1f}%"
            )

        lines.append("\n" + sep)
        return "\n".join(lines)

    ASSISTANT_SYSTEM_PROMPT = """You are an Airline Booking Analytics Assistant for Clarity Travel Technology Solutions.

You have access to airline booking data and business metrics generated from the dataset.

Your responsibilities:
1. Answer ONLY using the provided dataset context.
2. Do not invent information.
3. If information is unavailable in the dataset, clearly say: "This information is not available in the current dataset."
4. Provide concise business-focused answers.
5. Highlight important insights, trends, risks, and recommendations.

Always respond using EXACTLY this format:

### Answer
Provide a direct, specific answer with numbers from the data.

### Key Insight
Explain the most important takeaway from the data.

### Business Impact
Explain why this matters to Clarity Travel Technology Solutions.

### Recommendation
Suggest one clear, practical action."""

    def ask_groq_with_metrics(question, context, api_key):
        from groq import Groq
        client = Groq(api_key=api_key)
        user_message = f"Dataset Context:\n{context}\n\nUser Question:\n{question}"
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    # Suggestion buttons
    suggestions = [
        "Which airline generated the highest revenue?",
        "What is the overall cancellation rate?",
        "Which routes are most popular?",
        "Compare repeat customers and new customers.",
        "Which booking channel performs best?",
        "Which cabin class generates the highest revenue?",
        "Which month had the highest booking volume?",
        "What factors are associated with cancellations?",
        "What are the most common complaint categories?",
        "Which airline receives the most complaints?",
    ]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chosen_suggestion" not in st.session_state:
        st.session_state.chosen_suggestion = ""

    section_label("Example Questions \u2014 Click to Ask")
    scols = st.columns(2)
    for i, sug in enumerate(suggestions):
        if scols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state.chosen_suggestion = sug

    st.divider()

    with st.expander("\u2699\ufe0f How does this work?", expanded=False):
        st.markdown("""
        **Architecture: Dataset Metrics \u2192 Groq Llama-3**

        ```
        Your Question
             \u2193
        Pre-computed Dataset Metrics
        (Revenue, Cancellations, Routes,
         Customers, Complaints, Channels)
             \u2193
        Groq Llama-3
             \u2193
        Structured Business Answer
        (Answer \u2192 Key Insight \u2192 Business Impact \u2192 Recommendation)
        ```

        All metrics are computed from the **full 2,000-row dataset** \u2014 not samples.
        No vector database required.
        """)

    with st.form(key="assistant_form", clear_on_submit=True):
        user_q = st.text_input(
            "Ask a question about the Clarity bookings:",
            value=st.session_state.get("chosen_suggestion", ""),
            placeholder="e.g. Which airline generates the highest revenue?",
        )
        submitted = st.form_submit_button("\U0001f50d Ask", type="primary", use_container_width=True)

    if submitted and user_q.strip():
        st.session_state.chosen_suggestion = ""
        with st.spinner("\U0001f4ca Computing dataset metrics \u2192 \U0001f916 Generating insight with Groq Llama-3\u2026"):
            try:
                context = build_dataset_context(df)
                answer  = ask_groq_with_metrics(user_q, context, _ENV_GROQ_KEY)
                st.session_state.chat_history.append({
                    "question": user_q,
                    "answer":   answer,
                })
            except Exception as e:
                st.error(f"\u274c Error: {e}")

    for i, turn in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"""
        <div class="chat-q">
            <span style="color:#64748b;font-size:0.72rem;font-weight:700">YOUR QUESTION</span><br>
            <span style="color:#e2e8f0;font-weight:600">{turn['question']}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="chat-a">
            <span style="color:#22c55e;font-size:0.72rem;font-weight:700;letter-spacing:0.08em">
                \U0001f916 AI DATA ASSISTANT \u2014 Groq Llama-3
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(turn['answer'])

        if i < len(st.session_state.chat_history) - 1:
            st.divider()

    if st.session_state.chat_history:
        st.divider()
        if st.button("\U0001f5d1\ufe0f Clear conversation"):
            st.session_state.chat_history      = []
            st.session_state.chosen_suggestion = ""
            st.rerun()


# PAGE 13 — ABOUT PROJECT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About Project":
    st.title("ℹ️ About This Project")
    st.caption("Clarity Travel Technology Solutions — Data Science Assessment Submission")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Project Objective")
        st.markdown("""
        To analyse the Clarity TTS booking dataset and deliver three key data science deliverables:

        1. **Business Intelligence** — Uncover revenue patterns, cancellation drivers and customer behaviour
        2. **Cancellation Prediction** — Build and evaluate ML models to predict booking cancellations
        3. **Complaint Intelligence** — Classify and summarise airline customer complaints at scale
        """)

        st.subheader("📂 Dataset Overview")
        col_a, col_b = st.columns(2)
        col_a.metric("Total Records",  f"{len(df):,}")
        col_b.metric("Features",       "23")
        col_a.metric("Airlines",       df['airline'].nunique())
        col_b.metric("Routes",         df['route'].nunique())
        col_a.metric("Date Range",     "2025")
        col_b.metric("Booking Sources","NDC & GDS")

        st.subheader("⚙️ Feature Engineering")
        st.markdown("""
        Key engineered features:
        - `is_cancelled` — Binary target (Cancelled + Refunded = 1)
        - `route` — Origin → Destination concatenation
        - `fare_per_passenger` — Total fare divided by passenger count
        - `is_repeat_customer` — Binary flag for prior bookings > 0
        - `fare_to_avg_route_fare` — Fare relative to route average
        - `tax_ratio` — Tax proportion of total fare
        - `booking_month`, `booking_dayofweek` — Temporal features
        """)

    with col2:
        st.subheader("🤖 Modelling Approach")
        st.markdown(f"""
        **Task:** Binary Classification (Will this booking be cancelled or refunded?)

        **Models Trained:**
        - Logistic Regression (baseline, interpretable)
        - XGBoost (gradient boosted trees, high performance)

        **Pipeline:** Scikit-learn ColumnTransformer
        - Numeric: Median imputation + StandardScaler
        - Categorical: Constant imputation + OneHotEncoder

        **Evaluation:** Stratified 80/20 train-test split
        - XGBoost ROC-AUC: **{xg_auc:.4f}**
        - Logistic Regression ROC-AUC: **{lr_auc:.4f}**
        """)

        st.subheader("💬 NLP Approach")
        st.markdown("""
        **Complaint Classification:** Rule-based regex pattern matching
        - 9 operational categories
        - Severity scoring (High / Medium / Low)
        - Support team routing recommendations

        **AI Summarisation:** Groq Llama-3 (llama-3.1-8b-instant)
        - Executive summary generation from batch complaints
        - Structured output: Top Issues, Urgency, Actions

        **RAG Pipeline:**
        - Sentence Transformers (all-MiniLM-L6-v2) for embeddings
        - ChromaDB persistent vector store
        - Groq Llama-3 for grounded answer generation
        """)

        st.subheader("🛠️ Tools & Technologies")
        tech_stack = [
            "Python 3.11", "Pandas", "NumPy", "Scikit-Learn",
            "XGBoost", "Streamlit", "Matplotlib", "Seaborn",
            "Sentence Transformers", "ChromaDB", "Groq API",
            "Groq Llama-3", "python-dotenv",
        ]
        badges_html = " ".join([f'<span class="tech-badge">{t}</span>' for t in tech_stack])
        st.markdown(badges_html, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div class="footer">
        <strong>Clarity Travel Technology Solutions</strong> — Data Science Assessment Submission<br>
        June 2026 &nbsp;|&nbsp; Python · Scikit-Learn · XGBoost · Streamlit · Groq Llama-3
    </div>
    """, unsafe_allow_html=True)
