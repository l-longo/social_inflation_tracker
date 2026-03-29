"""
Social Inflation Tracker — Streamlit Dashboard + Chatbot
─────────────────────────────────────────────────────────
Tab 1: Daily inflation-keyword mention charts (US vs. Europe)
Tab 2: AI chatbot grounded in Reddit conversation data
"""

import html as html_module
import pathlib
import re
import random
import time as _time
import datetime as _dt
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from openai import OpenAI

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Inflation Tracker",
    page_icon=None,
    layout="wide",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="st-"], .stMarkdown, p, li, span, label {
    font-family: 'Inter', sans-serif !important;
    color: #e8eaf0;
}
code, pre, .stMetric [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── App & header ── */
.stApp { background-color: #0d0f14; }
header[data-testid="stHeader"] {
    background: #0d0f14;
    border-bottom: 1px solid #1e2130;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #10121a;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * { color: #c8cdd8 !important; }
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] strong { color: #ffffff !important; font-weight: 600; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(20,23,32,0.5);
    border: 1px solid #252840;
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] p {
    color: #8892b0 !important;
    font-size: 0.74rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 500;
}
[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 600 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e2130;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8892b0 !important;
    font-size: 0.84rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    padding: 10px 24px;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #4f8ef7 !important;
}

/* ── Dividers ── */
hr { border-color: #1e2130 !important; }

/* ── Region header bands ── */
.region-header-us {
    background: linear-gradient(90deg, #1a2540 0%, #0d0f14 100%);
    border-left: 3px solid #4f8ef7;
    padding: 10px 18px;
    border-radius: 4px;
    margin: 28px 0 16px 0;
}
.region-header-eu {
    background: linear-gradient(90deg, #1e1530 0%, #0d0f14 100%);
    border-left: 3px solid #a78bfa;
    padding: 10px 18px;
    border-radius: 4px;
    margin: 28px 0 16px 0;
}
.region-header-us h2, .region-header-eu h2 {
    margin: 0;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #ffffff !important;
}
.region-header-us p, .region-header-eu p {
    margin: 3px 0 0 0;
    font-size: 0.77rem;
    color: #8892b0 !important;
}

/* ── Headings ── */
h1 { color: #ffffff !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h2, h3, h4, h5 { color: #e0e4f0 !important; font-weight: 600 !important; }

/* ── Selectbox / dropdown ── */
[data-baseweb="select"] > div {
    background-color: #1a1d28 !important;
    border-color: #2a2f4a !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div { color: #e8eaf0 !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li,
[data-baseweb="menu"] { background: #1a1d28 !important; color: #e8eaf0 !important; }
[data-baseweb="option"] { background: #1a1d28 !important; color: #e8eaf0 !important; }
[data-baseweb="option"]:hover { background: #252840 !important; }

/* ── Radio buttons ── */
[data-testid="stRadio"] label, [data-testid="stRadio"] p { color: #c8cdd8 !important; font-size: 0.85rem; }

/* ── Multiselect ── */
[data-baseweb="tag"] { background: #252840 !important; }
[data-baseweb="tag"] span { color: #e8eaf0 !important; }

/* ── Field labels ── */
.stSelectbox > label, .stRadio > label,
.stMultiSelect > label, .stSlider > label {
    color: #c8cdd8 !important;
    font-size: 0.78rem !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Chat input box — white bg, black text, unambiguous ── */
[data-testid="stChatInput"],
[data-testid="stChatInputContainer"],
div[data-testid="stChatInput"] > div {
    background: #ffffff !important;
    border: 1px solid #c8cdd8 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"],
textarea[data-testid="stChatInputTextArea"],
div[data-testid="stChatInput"] textarea {
    color: #111111 !important;
    caret-color: #111111 !important;
    background: #ffffff !important;
    -webkit-text-fill-color: #111111 !important;
}
[data-testid="stChatInput"] textarea::placeholder,
div[data-testid="stChatInput"] textarea::placeholder {
    color: #999aaa !important;
    -webkit-text-fill-color: #999aaa !important;
}

/* ── Custom chat bubbles ── */
.chat-window {
    display: flex;
    flex-direction: column;
    gap: 14px;
    margin: 8px 0 20px 0;
}
.chat-row { display: flex; width: 100%; }
.chat-row-user      { justify-content: flex-end; }
.chat-row-assistant { justify-content: flex-start; }
.chat-bubble {
    max-width: 76%;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.88rem;
    line-height: 1.65;
}
.chat-bubble-user {
    background: #1e3a5f;
    border: 1px solid #2a4f7a;
    color: #dce8f8 !important;
}
.chat-bubble-assistant {
    background: #f5f6fa;
    border: 1px solid #dde1ec;
    color: #1a1d28 !important;
}
.chat-bubble-assistant p,
.chat-bubble-assistant li,
.chat-bubble-assistant span,
.chat-bubble-assistant strong,
.chat-bubble-assistant em { color: #1a1d28 !important; }
.chat-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 7px;
    opacity: 0.6;
}
.chat-bubble-user      .chat-label { color: #90c4e8 !important; }
.chat-bubble-assistant .chat-label { color: #556080 !important; }

/* ── Sample question buttons ── */
.stButton > button {
    background: rgba(20,23,32,0.6) !important;
    border: 1px solid #2a2f4a !important;
    border-radius: 6px !important;
    color: #c8cdd8 !important;
    font-size: 0.79rem !important;
    font-weight: 400 !important;
    padding: 8px 14px !important;
    text-align: left !important;
    white-space: normal !important;
    line-height: 1.45 !important;
    min-height: 56px !important;
    width: 100% !important;
    transition: border-color 0.15s, background 0.15s !important;
}
.stButton > button:hover {
    background: #1c2035 !important;
    border-color: #4f8ef7 !important;
    color: #ffffff !important;
}

/* ── Alerts ── */
.stAlert { background: rgba(20,23,32,0.6) !important; border-radius: 6px !important; }
.stAlert p { color: #e8eaf0 !important; }

/* ── Code blocks ── */
.stCodeBlock, code { background: #0a0c12 !important; color: #a8d8f0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a2f4a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path(__file__).parent / "data"

SUBREDDITS_US = {
    "economy":   {"color": "#4f8ef7", "label": "r/economy"},
    "economics": {"color": "#38bdf8", "label": "r/economics"},
}

SUBREDDITS_EU = {
    "europe":  {"color": "#a78bfa", "label": "r/europe"},
    "italy":   {"color": "#fb7185", "label": "r/italy"},
    "spain":   {"color": "#fbbf24", "label": "r/spain"},
    "germany": {"color": "#34d399", "label": "r/germany"},
    "france":  {"color": "#f97316", "label": "r/france"},
}

ALL_SUBREDDITS = {**SUBREDDITS_US, **SUBREDDITS_EU}

FORECAST_COUNTRIES = {
    "United States": {
        "label": "United States (CPI)",
        "source": "fred",
        "code": None,
        "measure": "US CPI",
        "reddit_subs": ["economy", "economics"],
        "color": "#4f8ef7",
    },
    "Euro Area": {
        "label": "Euro Area (HICP)",
        "source": "ecb",
        "code": "U2",
        "measure": "Euro Area HICP",
        "reddit_subs": ["europe", "economics"],
        "color": "#a78bfa",
    },
    "Italy": {
        "label": "Italy (HICP)",
        "source": "ecb",
        "code": "IT",
        "measure": "Italian HICP",
        "reddit_subs": ["europe", "italy"],
        "color": "#fb7185",
    },
    "Germany": {
        "label": "Germany (HICP)",
        "source": "ecb",
        "code": "DE",
        "measure": "German HICP",
        "reddit_subs": ["europe", "germany"],
        "color": "#34d399",
    },
    "France": {
        "label": "France (HICP)",
        "source": "ecb",
        "code": "FR",
        "measure": "French HICP",
        "reddit_subs": ["europe", "france"],
        "color": "#f97316",
    },
    "Spain": {
        "label": "Spain (HICP)",
        "source": "ecb",
        "code": "ES",
        "measure": "Spanish HICP",
        "reddit_subs": ["europe", "spain"],
        "color": "#fbbf24",
    },
}

KEYWORDS_TRACKED = [
    "inflation", "hyperinflation", "disinflation",
    "deflation", "price", "prices",
]

LLM_MODELS = {
    "gpt-oss-120b":              "GPT OSS 120B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B Instruct",
    "minimax-m2":                "MiniMax M2",
    "mistral-small-3.2-24b":     "Mistral Small 3.2 24B",
}

FORECAST_LLM_MODELS = {
    "gpt-4o": "GPT-4o",
    **LLM_MODELS,
}

LLM_BASE_URL = "https://api-gpt.jrc.ec.europa.eu/v1"
MAX_CONTEXT_THREADS = 30

SAMPLE_QUESTIONS = [
    "Are users expecting higher or lower inflation in the coming months?",
    "What is the general sentiment toward current price levels?",
    "Are users speculating on central bank decisions or monetary policy?",
    "Which economic sectors or goods are mentioned most frequently?",
    "Are there notable differences between US and European discussions?",
]


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_mentions(subreddit: str) -> pd.DataFrame:
    path = DATA_DIR / f"{subreddit}_daily_mentions.parquet"
    if not path.exists():
        return pd.DataFrame(
            columns=["date", "mentions_comments", "mentions_submissions", "mentions_total"]
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    # No hard date cutoff — the slider controls the visible range
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=300)
def load_conversations(subreddit: str) -> pd.DataFrame:
    path = DATA_DIR / f"{subreddit}_conversations.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"])
    return df


# ─── INFLATION DATA FETCHERS ──────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_us_cpi() -> pd.DataFrame:
    """Fetch US CPI YoY % from OECD SDMX REST API (monthly, all-items)."""
    # GY = growth rate vs same period of previous year (YoY %)
    url = (
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/"
        "USA.M.N.CPI.PA._T.N.GY"
        "?format=csvfilewithlabels&startPeriod=2010-01"
    )
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    # Column names vary; find time and obs-value columns robustly
    time_col  = next(c for c in df.columns if "TIME" in c.upper() or "PERIOD" in c.upper())
    value_col = next(c for c in df.columns if "OBS" in c.upper() and "VALUE" in c.upper())
    df["date"]      = pd.to_datetime(df[time_col])
    df["inflation"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["inflation"])
    return df[["date", "inflation"]].sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600)
def fetch_hicp(country_code: str) -> pd.DataFrame:
    """Fetch HICP annual rate from ECB API."""
    series_key = "M.U2.N.000000.4D0.ANR" if country_code == "U2" else f"M.{country_code}.N.000000.4D0.ANR"
    url = f"https://data-api.ecb.europa.eu/service/data/HICP/{series_key}"
    resp = requests.get(url, params={"format": "csvdata"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df["date"] = pd.to_datetime(df["TIME_PERIOD"])
    df["inflation"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["inflation"])
    return df[["date", "inflation"]].sort_values("date").reset_index(drop=True)


def fetch_inflation_data(country_key: str) -> pd.DataFrame:
    meta = FORECAST_COUNTRIES[country_key]
    if meta["source"] == "fred":
        return fetch_us_cpi()
    return fetch_hicp(meta["code"])


def _next_month_date(last_date: pd.Timestamp) -> pd.Timestamp:
    if last_date.month == 12:
        return pd.Timestamp(year=last_date.year + 1, month=1, day=1)
    return pd.Timestamp(year=last_date.year, month=last_date.month + 1, day=1)



def run_single_forecast(
    client,
    model_key: str,
    series_df: pd.DataFrame,
    country_name: str,
    measure: str,
    temperature: float,
    context_text: str | None = None,
    run_index: int = 0,
) -> tuple:
    """
    Send one chain-of-thought prompt to the LLM.
    Returns (forecast_value, next_date, reasoning_text).
    The prompt is identical across all 30 runs; temperature variation
    drives distributional spread through the chain-of-thought reasoning.
    """
    last_n = series_df.tail(36)
    series_text = "\n".join(
        f"  {row['date'].strftime('%Y-%m')}: {row['inflation']:.2f}%"
        for _, row in last_n.iterrows()
    )
    last_row = series_df.iloc[-1]
    last_date: pd.Timestamp = last_row["date"]
    last_value: float = float(last_row["inflation"])
    next_date = _next_month_date(last_date)
    next_month_str = next_date.strftime("%B %Y")

    system_msg = (
        "You are a senior macroeconomic forecaster. "
        "When asked for an inflation forecast you must:\n"
        "1. Briefly reason your choice."
        "(3–5 sentences).\n"
        "2. End your response with exactly one line in the format:\n"
        "   FORECAST: <number>\n"
        "where <number> is the year-on-year % inflation rate rounded to one decimal "
        "(e.g. FORECAST: 2.4). Do not add any text after that line."
    )

    current_month_str = _dt.date.today().strftime("%B %Y")

    user_msg = (
        f"Assume that you are in {current_month_str}. "
        f"Please give me your best forecast of year-over-year {measure} inflation "
        f"in {country_name} for the current month.\n\n"
        f"Here is the historical series to inform your forecast "
        f"(last 36 months of available data, last observation: "
        f"{last_date.strftime('%B %Y')} = {last_value:.2f}%):\n\n"
        f"{series_text}\n\n"
    )
    if context_text:
        user_msg += (
            "Recent Reddit discussions about inflation in this country/region "
            "(last 24 h):\n\n"
            f"{context_text[:3500]}\n\n"
        )
    user_msg += (
        "Reason briefly, then write your final answer as:\n"
        "FORECAST: <number>"
    )

    response = client.chat.completions.create(
        model=model_key,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=300,
        timeout=60,
    )
    raw = response.choices[0].message.content.strip()

    # Extract FORECAST: <number>; fall back to last number in text
    fc_match = re.search(r"FORECAST\s*:\s*(-?\d+\.?\d*)", raw, re.IGNORECASE)
    if fc_match:
        value = float(fc_match.group(1))
        reasoning = raw[:raw.lower().rfind("forecast")].strip()
    else:
        nums = re.findall(r"-?\d+\.?\d*", raw)
        value = float(nums[-1]) if nums else None
        reasoning = raw

    return value, next_date, reasoning


# ─── REDDIT CONTEXT BUILDER ───────────────────────────────────────────────────
def build_context_block(subs: list, only_inflation: bool = True, max_threads: int = MAX_CONTEXT_THREADS) -> str:
    parts = []
    for sub in subs:
        df = load_conversations(sub)
        if df.empty:
            continue
        if only_inflation and "has_inflation_keywords" in df.columns:
            df = df[df["has_inflation_keywords"]]
        if "created_utc" in df.columns:
            df = df.sort_values("created_utc", ascending=False)
        df = df.head(max_threads)
        for _, row in df.iterrows():
            conv = row.get("conversation", "")
            if conv:
                parts.append(f"── r/{sub} ──\n{conv}\n")
    return "\n".join(parts) if parts else "(No conversation data available yet.)"


def _is_context_length_error(exc: Exception) -> bool:
    """Return True if the exception is a context-window overflow."""
    msg = str(exc).lower()
    triggers = [
        "maximum context length",
        "context_length_exceeded",
        "context window",
        "too many tokens",
        "input too long",
        "prompt is too long",
        "reduce the length",
    ]
    if any(t in msg for t in triggers):
        return True
    # Some gateways return HTTP 400 with token-related text
    if "400" in msg and any(w in msg for w in ("token", "context", "length")):
        return True
    return False


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True for 429 / rate-limit responses."""
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def _is_transient_error(exc: Exception) -> bool:
    """Return True for timeouts and connection errors worth retrying."""
    msg = str(exc).lower()
    return any(w in msg for w in (
        "timeout", "timed out", "connection", "502", "503", "504", "service unavailable"
    ))


def get_llm_client():
    token = st.secrets.get("JRC_TOKEN", "")
    if not token:
        return None
    return OpenAI(api_key=token, base_url=LLM_BASE_URL)


# ─── CHAT RENDERER ────────────────────────────────────────────────────────────
def _md_to_html(text: str) -> str:
    """Minimal Markdown -> HTML for chat bubble rendering."""
    t = html_module.escape(text)

    # Fenced code blocks
    t = re.sub(
        r"```(?:\w+)?\n(.*?)```",
        lambda m: (
            "<pre style='background:#eef0f6;border-radius:4px;padding:8px 10px;"
            "font-family:monospace;font-size:0.82em;color:#1a1d28;"
            "overflow-x:auto;margin:6px 0;'>"
            f"{m.group(1)}</pre>"
        ),
        t, flags=re.DOTALL,
    )
    # Inline code
    t = re.sub(
        r"`([^`]+)`",
        r"<code style='background:#eef0f6;padding:1px 5px;border-radius:3px;"
        r"font-size:0.84em;font-family:monospace;'>\1</code>",
        t,
    )
    # Bold & italic
    t = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", t)
    t = re.sub(r"\*\*(.+?)\*\*",     r"<strong>\1</strong>", t)
    t = re.sub(r"\*(.+?)\*",         r"<em>\1</em>", t)

    # Process line by line for lists
    lines = t.split("\n")
    out, in_ul, in_ol = [], False, False
    ul_style = "margin:6px 0 6px 20px;padding:0;"
    ol_style = "margin:6px 0 6px 20px;padding:0;"

    for line in lines:
        b = re.match(r"^[\*\-]\s+(.*)", line)
        n = re.match(r"^\d+\.\s+(.*)", line)
        if b:
            if not in_ul:
                if in_ol: out.append("</ol>"); in_ol = False
                out.append(f"<ul style='{ul_style}'>"); in_ul = True
            out.append(f"<li>{b.group(1)}</li>")
        elif n:
            if not in_ol:
                if in_ul: out.append("</ul>"); in_ul = False
                out.append(f"<ol style='{ol_style}'>"); in_ol = True
            out.append(f"<li>{n.group(1)}</li>")
        else:
            if in_ul: out.append("</ul>"); in_ul = False
            if in_ol: out.append("</ol>"); in_ol = False
            if line.strip() == "":
                out.append("<div style='height:6px;'></div>")
            else:
                out.append(f"<p style='margin:3px 0;'>{line}</p>")

    if in_ul: out.append("</ul>")
    if in_ol: out.append("</ol>")
    return "\n".join(out)


def render_chat(messages: list) -> None:
    if not messages:
        return
    rows = ['<div class="chat-window">']
    for msg in messages:
        content_html = _md_to_html(msg["content"])
        if msg["role"] == "user":
            rows.append(
                '<div class="chat-row chat-row-user">'
                '<div class="chat-bubble chat-bubble-user">'
                '<div class="chat-label">You</div>'
                f'<div>{content_html}</div>'
                "</div></div>"
            )
        else:
            rows.append(
                '<div class="chat-row chat-row-assistant">'
                '<div class="chat-bubble chat-bubble-assistant">'
                '<div class="chat-label">Analyst</div>'
                f'<div>{content_html}</div>'
                "</div></div>"
            )
    rows.append("</div>")
    st.markdown("\n".join(rows), unsafe_allow_html=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")
    st.markdown("---")

    st.markdown("**United States**")
    selected_us = st.multiselect(
        "us_subs",
        options=list(SUBREDDITS_US.keys()),
        default=list(SUBREDDITS_US.keys()),
        format_func=lambda s: SUBREDDITS_US[s]["label"],
        label_visibility="collapsed",
    )

    st.markdown("**Europe**")
    selected_eu = st.multiselect(
        "eu_subs",
        options=list(SUBREDDITS_EU.keys()),
        default=list(SUBREDDITS_EU.keys()),
        format_func=lambda s: SUBREDDITS_EU[s]["label"],
        label_visibility="collapsed",
    )

    selected_subs = selected_us + selected_eu

    st.markdown("---")
    breakdown = st.radio("Breakdown", ["Total", "Comments vs. Submissions"], index=0)

    st.markdown("---")
    st.markdown("**Keywords tracked**")
    st.code(", ".join(KEYWORDS_TRACKED), language=None)

    st.markdown("---")
    # File status — shows which parquet files are present/missing
    st.markdown("**Data files**")
    all_known = list(SUBREDDITS_US.keys()) + list(SUBREDDITS_EU.keys())
    for sub in all_known:
        found = (DATA_DIR / f"{sub}_daily_mentions.parquet").exists()
        dot   = "&#9679;"
        color = "#34d399" if found else "#f87171"
        label = ALL_SUBREDDITS.get(sub, {}).get("label", f"r/{sub}")
        st.markdown(
            f"<span style='color:{color};font-size:0.7rem;'>{dot}</span>"
            f"<span style='font-size:0.78rem;color:#c8cdd8;margin-left:6px;'>{label}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    if st.button("Reload data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Data sourced from Reddit via PRAW · Updated periodically")


# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_dashboard, tab_chat, tab_forecast = st.tabs(["Dashboard", "Conversational Analysis", "⚡ Real-time Forecast"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("# Social Inflation Tracker")
    st.markdown(
        "Daily volume of inflation-related keyword mentions across Reddit communities. "
        "Results are segmented by geographic focus: United States and Europe."
    )

    datasets = {sub: load_mentions(sub) for sub in selected_subs}
    non_empty = [d["date"] for d in datasets.values() if not d.empty]

    if not non_empty:
        st.warning(
            "No data files found in the `data/` directory. "
            "Run `collect_data_parquet.py` and push the parquet files before launching the dashboard."
        )
        st.stop()

    all_dates = pd.concat(non_empty, ignore_index=True)
    date_min, date_max = all_dates.min().date(), all_dates.max().date()

    # Default view starts at March 2026; slider allows going back to January if needed
    _march = _dt.date(2026, 3, 1)
    default_start = max(date_min, _march)

    if date_min < date_max:
        date_range = st.slider(
            "Date range",
            min_value=date_min,
            max_value=date_max,
            value=(default_start, date_max),
            format="YYYY-MM-DD",
        )
    else:
        date_range = (date_min, date_max)

    for sub in list(datasets.keys()):
        df = datasets[sub]
        if not df.empty:
            mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
            datasets[sub] = df.loc[mask]

    y_max = 0
    for sub in selected_subs:
        df = datasets[sub]
        if not df.empty:
            y_max = max(y_max, df["mentions_total"].max())
    y_max = int(y_max * 1.15) + 1

    def render_region(region_subs, region_meta, header_class, region_name, description):
        active = [s for s in region_subs if s in datasets]
        if not active:
            return
        st.markdown(
            f'<div class="{header_class}"><h2>{region_name}</h2><p>{description}</p></div>',
            unsafe_allow_html=True,
        )
        # PUT THIS INSTEAD:
        for sub in active:
            df = datasets[sub]
            label = region_meta[sub]["label"]
            total = int(df["mentions_total"].sum()) if not df.empty else 0
            avg   = round(df["mentions_total"].mean(), 1) if not df.empty else 0.0
            peak  = int(df["mentions_total"].max()) if not df.empty else 0
            c1, c2, c3, _ = st.columns([1, 1, 1, 3])
            c1.metric(f"{label}  ·  Total",     f"{total:,}")
            c2.metric(f"{label}  ·  Daily avg", f"{avg:,}")
            c3.metric(f"{label}  ·  Peak day",  f"{peak:,}")

        st.markdown("")

        for sub in active:
            df = datasets[sub]
            meta = region_meta[sub]
            st.markdown(f"##### {meta['label']}")
            if df.empty:
                st.info(
                    f"No data for {meta['label']} in the selected range. "
                    f"Expected: `data/{sub}_daily_mentions.parquet`"
                )
                continue
            fig = go.Figure()
            if breakdown == "Total":
                fig.add_trace(go.Bar(
                    x=df["date"], y=df["mentions_total"],
                    name="Total mentions",
                    marker_color=meta["color"], marker_line_width=0, opacity=0.9,
                    hovertemplate="%{x|%b %d, %Y}<br>Mentions: <b>%{y}</b><extra></extra>",
                ))
            else:
                fig.add_trace(go.Bar(
                    x=df["date"], y=df["mentions_comments"], name="Comments",
                    marker_color=meta["color"], opacity=0.9,
                    hovertemplate="%{x|%b %d, %Y}<br>Comments: <b>%{y}</b><extra></extra>",
                ))
                fig.add_trace(go.Bar(
                    x=df["date"], y=df["mentions_submissions"], name="Submissions",
                    marker_color="#475569", opacity=0.85,
                    hovertemplate="%{x|%b %d, %Y}<br>Submissions: <b>%{y}</b><extra></extra>",
                ))
                fig.update_layout(barmode="stack")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#c8cdd8", size=12),
                xaxis=dict(
                    title="", gridcolor="rgba(255,255,255,0.04)",
                    tickformat="%b %d", tickcolor="#3a3f55", linecolor="#1e2130",
                ),
                yaxis=dict(
                    title="Keyword mentions",
                    gridcolor="rgba(255,255,255,0.06)",
                    tickcolor="#3a3f55", linecolor="#1e2130", range=[0, y_max],
                ),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11),
                ),
                margin=dict(l=50, r=20, t=36, b=40),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

    render_region(
        selected_us, SUBREDDITS_US, "region-header-us",
        "United States", "r/economy · r/economics",
    )
    render_region(
        selected_eu, SUBREDDITS_EU, "region-header-eu",
        "Europe", "r/europe · r/italy · r/spain · r/germany · r/france",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONVERSATIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("# Conversational Analysis")
    st.markdown(
        "Query an AI analyst grounded in the most recent Reddit discussions on inflation and prices. "
        "Select the community scope and model, then ask follow-up questions — "
        "the full conversation is retained across turns."
    )

    client = get_llm_client()
    if client is None:
        st.error(
            "JRC_TOKEN is not configured. "
            "Add your JWT token under Streamlit Cloud → Settings → Secrets:\n\n"
            '```toml\nJRC_TOKEN = "your-token-here"\n```'
        )
        st.stop()

    # ── Configuration row ─────────────────────────────────────────────────────
    cfg_c1, cfg_c2, cfg_c3 = st.columns([1, 1, 2])

    with cfg_c1:
        region_choice = st.radio(
            "Community scope",
            options=["United States", "Europe", "Both"],
            index=2,
        )

    with cfg_c2:
        selected_model_key = st.selectbox(
            "Language model",
            options=list(LLM_MODELS.keys()),
            format_func=lambda k: LLM_MODELS[k],
            index=1,
        )

    with cfg_c3:
        if region_choice == "United States":
            chat_subs = list(SUBREDDITS_US.keys())
        elif region_choice == "Europe":
            chat_subs = list(SUBREDDITS_EU.keys())
        else:
            chat_subs = list(SUBREDDITS_US.keys()) + list(SUBREDDITS_EU.keys())

        n_threads = sum(len(load_conversations(s)) for s in chat_subs)
        infl_threads = 0
        for s in chat_subs:
            df_c = load_conversations(s)
            if not df_c.empty and "has_inflation_keywords" in df_c.columns:
                infl_threads += int(df_c["has_inflation_keywords"].sum())
        st.markdown(
            "<div style='margin-top:28px;font-size:0.82rem;color:#8892b0;'>"
            f"Threads loaded: <strong style='color:#c8cdd8;'>{n_threads}</strong>"
            "&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"With inflation keywords: <strong style='color:#c8cdd8;'>{infl_threads}</strong>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Session state ─────────────────────────────────────────────────────────
    state_key = f"{region_choice}_{selected_model_key}"
    if st.session_state.get("active_chat_key") != state_key:
        st.session_state.active_chat_key = state_key
        st.session_state.chat_history = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # ── Sample questions ──────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown(
            "<p style='font-size:0.76rem;color:#8892b0;font-weight:500;"
            "text-transform:uppercase;letter-spacing:0.07em;margin-bottom:10px;'>"
            "Suggested questions</p>",
            unsafe_allow_html=True,
        )
        q_cols = st.columns(len(SAMPLE_QUESTIONS))
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if q_cols[i].button(q, key=f"sq_{i}"):
                st.session_state.pending_question = q
                st.rerun()
        st.markdown("")

    # ── Render conversation ───────────────────────────────────────────────────
    render_chat(st.session_state.chat_history)

    # ── Input ─────────────────────────────────────────────────────────────────
    user_input = None
    typed = st.chat_input("Ask a question about inflation discussions...")
    if typed:
        user_input = typed
    elif st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None

    # ── LLM call ──────────────────────────────────────────────────────────────
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        sub_list = ", ".join(f"r/{s}" for s in chat_subs)

        def _build_api_messages(n_threads: int) -> list:
            context = build_context_block(chat_subs, only_inflation=True, max_threads=n_threads)
            system_prompt = (
                "You are a senior economic analyst assistant embedded in the Social Inflation Tracker.\n"
                f"You have access to recent Reddit conversations from: {sub_list}.\n\n"
                "These threads discuss inflation, price levels, monetary policy, and related topics.\n"
                "Use them as your primary evidence base. Cite specific posts or users when relevant.\n"
                "If the data does not contain sufficient information, state that clearly.\n\n"
                "Be analytical and concise. Distinguish dominant views from minority ones.\n"
                "Avoid speculation beyond what the data supports.\n\n"
                "═══ RECENT CONVERSATIONS ═══\n"
                f"{context}\n"
                "═══ END OF CONVERSATIONS ═══"
            )
            msgs = [{"role": "system", "content": system_prompt}]
            msgs += st.session_state.chat_history[-20:]
            return msgs

        with st.spinner("Analysing…"):

            reply = None
            last_error: str = ""
            n_threads = MAX_CONTEXT_THREADS
            min_threads = 2
            transient_retries = 2   # how many times to retry on timeout / 5xx
            rate_limit_wait  = 8    # seconds to wait on 429

            def _call(n_t: int) -> str:
                response = client.chat.completions.create(
                    model=selected_model_key,
                    messages=_build_api_messages(n_t),
                    temperature=0.3,
                    max_tokens=1200,
                    timeout=60,
                )
                return response.choices[0].message.content

            while n_threads >= min_threads:
                try:
                    reply = _call(n_threads)
                    break

                except Exception as exc:
                    last_error = repr(exc)

                    if _is_context_length_error(exc):
                        if n_threads > min_threads:
                            n_threads = max(min_threads, n_threads // 2)
                            continue          # retry with fewer threads
                        # Already at minimum — try trimming chat history too
                        st.session_state.chat_history = st.session_state.chat_history[-6:]
                        try:
                            reply = _call(n_threads)
                        except Exception as exc2:
                            last_error = repr(exc2)
                        break

                    elif _is_rate_limit_error(exc):
                        _time.sleep(rate_limit_wait)
                        try:
                            reply = _call(n_threads)
                        except Exception as exc2:
                            last_error = repr(exc2)
                        break

                    elif _is_transient_error(exc) and transient_retries > 0:
                        transient_retries -= 1
                        _time.sleep(3)
                        continue              # retry same n_threads

                    else:
                        break                 # non-recoverable — fall through

            # ── Store last error for debugging ────────────────────────────────
            st.session_state["last_llm_error"] = last_error if not reply else ""

            if reply is None:
                if _is_context_length_error(Exception(last_error)):
                    reply = (
                        "⚠️ The model's context window is full. "
                        "Try narrowing the community scope, or click **Clear conversation** to start fresh."
                    )
                else:
                    reply = (
                        "⚠️ The model returned an error. "
                        "Please try again in a moment."
                    )

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    # ── Sidebar chat controls ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Active model**")
        st.code(selected_model_key, language=None)
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.session_state["last_llm_error"] = ""
            st.rerun()
        last_err = st.session_state.get("last_llm_error", "")
        if last_err:
            with st.expander("⚠️ Last API error (debug)", expanded=False):
                st.code(last_err, language=None)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REAL-TIME FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown("# Real-time Inflation Forecast")
    st.markdown(
        "Select a country, a language model, and a forecast type. "
        "The tracker will run **30 independent simulations** at randomly varied temperatures "
        "(0.3 – 0.8) and display the forecast distribution with confidence bands. "
        "Conditional forecasts additionally ground the LLM with recent Reddit conversations."
    )

    client_fc = get_llm_client()
    if client_fc is None:
        st.error(
            "JRC_TOKEN is not configured. "
            "Add your JWT token under Streamlit Cloud → Settings → Secrets."
        )
        st.stop()

    # ── Config row ────────────────────────────────────────────────────────────
    fc_c1, fc_c2, fc_c3 = st.columns([1, 1, 1])

    with fc_c1:
        fc_country = st.selectbox(
            "Country / region",
            options=list(FORECAST_COUNTRIES.keys()),
            format_func=lambda k: FORECAST_COUNTRIES[k]["label"],
        )

    with fc_c2:
        fc_model_key = st.selectbox(
            "Language model",
            options=list(FORECAST_LLM_MODELS.keys()),
            format_func=lambda k: FORECAST_LLM_MODELS[k],
            key="fc_model",
        )

    with fc_c3:
        fc_type = st.radio(
            "Forecast type",
            options=["Unconditional", "Conditional on Reddit"],
            index=0,
            key="fc_type_radio",
        )

    fc_meta   = FORECAST_COUNTRIES[fc_country]
    fc_color  = fc_meta["color"]
    fc_subs   = fc_meta["reddit_subs"]
    fc_measure = fc_meta["measure"]

    # Context info
    if fc_type == "Conditional on Reddit":
        n_ctx = sum(len(load_conversations(s)) for s in fc_subs)
        st.caption(
            f"Conditional context: {', '.join('r/'+s for s in fc_subs)}  ·  "
            f"{n_ctx} threads available"
        )

    st.markdown("---")

    # ── Run button ────────────────────────────────────────────────────────────
    run_col, clear_col = st.columns([2, 1])
    with run_col:
        run_btn = st.button(
            "▶  Run 30-simulation Forecast",
            type="primary",
            use_container_width=True,
        )
    with clear_col:
        if st.button("✕  Clear results", use_container_width=True):
            for k in ["fc_results", "fc_temps", "fc_reasonings", "fc_series", "fc_country_done",
                      "fc_model_done", "fc_type_done", "fc_next_date", "fc_errors"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── Run simulations ───────────────────────────────────────────────────────
    if run_btn:
        # Fetch series
        with st.spinner(f"Fetching {fc_measure} data…"):
            try:
                series_df = fetch_inflation_data(fc_country)
            except Exception as e:
                st.error(f"Failed to fetch inflation data: {e}")
                st.stop()

        # Build Reddit context if conditional
        context_text = None
        if fc_type == "Conditional on Reddit":
            context_text = build_context_block(fc_subs, only_inflation=True, max_threads=20)
            if context_text.startswith("(No conversation"):
                context_text = None

        # Run 30 simulations
        results, temps, errors, reasonings = [], [], [], []
        next_date_result = None
        progress_bar = st.progress(0, text="Starting simulations…")
        status_slot   = st.empty()

        for i in range(30):
            temp = random.uniform(0.1, 0.9)
            try:
                val, next_date_result, reasoning = run_single_forecast(
                    client_fc, fc_model_key, series_df,
                    fc_country, fc_measure, temp, context_text,
                    run_index=i,
                )
                if val is not None:
                    results.append(val)
                    temps.append(temp)
                    reasonings.append(reasoning)
                    status_slot.markdown(
                        f"<span style='font-size:0.82rem;color:#8892b0;'>"
                        f"Run {i+1}/30 &nbsp;·&nbsp; temp={temp:.2f} "
                        f"&nbsp;·&nbsp; forecast: <strong style='color:#e8eaf0;'>{val:.2f}%</strong></span>",
                        unsafe_allow_html=True,
                    )
                else:
                    errors.append(f"Run {i+1}: could not parse number from model output")
            except Exception as exc:
                errors.append(f"Run {i+1}: {repr(exc)}")
            progress_bar.progress((i + 1) / 30, text=f"Simulation {i+1} / 30")
            _time.sleep(0.2)

        progress_bar.empty()
        status_slot.empty()

        # Persist to session state
        st.session_state.fc_results    = results
        st.session_state.fc_temps      = temps
        st.session_state.fc_reasonings = reasonings
        st.session_state.fc_series     = series_df
        st.session_state.fc_country_done = fc_country
        st.session_state.fc_model_done   = fc_model_key
        st.session_state.fc_type_done  = fc_type
        st.session_state.fc_next_date  = next_date_result
        st.session_state.fc_errors     = errors
        st.rerun()

    # ── Display results ────────────────────────────────────────────────────────
    if st.session_state.get("fc_results"):
        results      = st.session_state.fc_results
        temps        = st.session_state.fc_temps
        reasonings   = st.session_state.get("fc_reasonings", [""] * len(results))
        series_df    = st.session_state.fc_series
        next_date    = st.session_state.fc_next_date
        done_country = st.session_state.fc_country_done
        done_model   = st.session_state.fc_model_done
        done_type    = st.session_state.fc_type_done
        errors_fc    = st.session_state.get("fc_errors", [])

        done_meta  = FORECAST_COUNTRIES[done_country]
        done_color = done_meta["color"]

        if not results:
            st.warning("All 30 simulations failed to parse a valid number. See errors below.")
        else:
            mean_fc              = float(np.mean(results))
            median_fc            = float(np.median(results))
            std_fc               = float(np.std(results))
            p5, p25, p75, p95   = (float(x) for x in np.percentile(results, [5, 25, 75, 95]))
            last_actual          = float(series_df.iloc[-1]["inflation"])
            last_date_str        = series_df.iloc[-1]["date"].strftime("%B %Y")
            next_month_label     = next_date.strftime("%B %Y")

            # ── Summary metrics ──────────────────────────────────────────────
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Last actual",     f"{last_actual:.2f}%",  delta=None, help=last_date_str)
            m2.metric("Forecast mean",   f"{mean_fc:.2f}%",
                      delta=f"{mean_fc - last_actual:+.2f}pp", help=next_month_label)
            m3.metric("Forecast median", f"{median_fc:.2f}%")
            m4.metric("Std dev",         f"{std_fc:.2f}pp")
            m5.metric("90 % CI",         f"[{p5:.2f}, {p95:.2f}]")

            st.markdown("")

            # ── Chart ────────────────────────────────────────────────────────
            hist_display = series_df.tail(36).copy()

            # Jitter the 30 simulation points a tiny bit on x so they're visible
            rng = np.random.default_rng(42)
            x_jitter = [
                next_date + pd.Timedelta(hours=float(rng.uniform(-60, 60)))
                for _ in results
            ]

            rgb_hex   = done_color.lstrip("#")
            r, g, b   = int(rgb_hex[0:2], 16), int(rgb_hex[2:4], 16), int(rgb_hex[4:6], 16)
            rgba_dim  = f"rgba({r},{g},{b},0.25)"
            rgba_mid  = f"rgba({r},{g},{b},0.55)"
            rgba_full = f"rgba({r},{g},{b},1.0)"

            fig_fc = go.Figure()

            # Historical line
            fig_fc.add_trace(go.Scatter(
                x=hist_display["date"],
                y=hist_display["inflation"],
                mode="lines",
                name="Historical",
                line=dict(color=done_color, width=2.5),
                hovertemplate="%{x|%b %Y}: <b>%{y:.2f}%</b><extra></extra>",
            ))

            # Dotted bridge from last actual → mean forecast
            fig_fc.add_trace(go.Scatter(
                x=[hist_display.iloc[-1]["date"], next_date],
                y=[last_actual, mean_fc],
                mode="lines",
                line=dict(color=done_color, width=1.5, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))

            # Individual simulation dots (jittered)
            fig_fc.add_trace(go.Scatter(
                x=x_jitter,
                y=results,
                mode="markers",
                name=f"30 simulations",
                marker=dict(color=done_color, size=6, opacity=0.35,
                            line=dict(color="white", width=0.5)),
                hovertemplate="Sim: <b>%{y:.2f}%</b><extra></extra>",
            ))

            # 90 % CI — thin bar
            fig_fc.add_trace(go.Scatter(
                x=[next_date], y=[mean_fc],
                mode="markers",
                marker=dict(size=1, color=rgba_dim),
                error_y=dict(
                    type="data", symmetric=False,
                    array=[p95 - mean_fc],
                    arrayminus=[mean_fc - p5],
                    color=rgba_dim,
                    thickness=3, width=14,
                ),
                name="90 % CI",
                hoverinfo="skip",
            ))

            # 50 % CI — thick bar
            fig_fc.add_trace(go.Scatter(
                x=[next_date], y=[mean_fc],
                mode="markers",
                marker=dict(size=1, color=rgba_mid),
                error_y=dict(
                    type="data", symmetric=False,
                    array=[p75 - mean_fc],
                    arrayminus=[mean_fc - p25],
                    color=rgba_mid,
                    thickness=8, width=10,
                ),
                name="50 % CI",
                hoverinfo="skip",
            ))

            # Mean star marker
            fig_fc.add_trace(go.Scatter(
                x=[next_date], y=[mean_fc],
                mode="markers",
                name=f"Mean forecast: {mean_fc:.2f}%",
                marker=dict(
                    color="#ffffff", size=16, symbol="star",
                    line=dict(color=done_color, width=2.5),
                ),
                hovertemplate=f"Forecast mean: <b>{mean_fc:.2f}%</b><extra></extra>",
            ))

            # Shade background for forecast column
            fig_fc.add_vrect(
                x0=hist_display.iloc[-1]["date"], x1=next_date + pd.Timedelta(days=15),
                fillcolor="rgba(255,255,255,0.03)",
                layer="below", line_width=0,
            )
            fig_fc.add_vline(
                x=next_date,
                line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dash"),
            )

            fig_fc.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#c8cdd8", size=12),
                title=dict(
                    text=(
                        f"{done_meta['measure']} — {done_model} "
                        f"({'Conditional' if 'Conditional' in done_type else 'Unconditional'}) "
                        f"· Forecast for {next_month_label}"
                    ),
                    font=dict(size=13, color="#e8eaf0"),
                    x=0,
                ),
                xaxis=dict(
                    title="", gridcolor="rgba(255,255,255,0.04)",
                    tickformat="%b %Y", tickcolor="#3a3f55", linecolor="#1e2130",
                ),
                yaxis=dict(
                    title="Inflation (YoY %)",
                    gridcolor="rgba(255,255,255,0.06)",
                    tickcolor="#3a3f55", linecolor="#1e2130",
                    ticksuffix="%",
                ),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11),
                ),
                margin=dict(l=60, r=20, t=60, b=50),
                height=420,
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # ── Download CSV ──────────────────────────────────────────────────
            today_str = _dt.date.today().strftime("%Y%m%d")
            safe_country = done_country.lower().replace(" ", "_")
            safe_model   = done_model_key = done_model.replace("/", "-").replace(" ", "_").lower()
            safe_type    = "conditional" if "Conditional" in done_type else "unconditional"
            csv_filename = f"forecast_{safe_country}_{safe_model}_{safe_type}_{today_str}.csv"

            fc_df = pd.DataFrame({
                "run":            range(1, len(results) + 1),
                "temperature":    [round(t, 4) for t in temps],
                "forecast_pct":   [round(v, 4) for v in results],
                "reasoning":      reasonings,
                "country":        done_country,
                "measure":        done_meta["measure"],
                "model":          done_model,
                "forecast_type":  done_type,
                "forecast_month": next_month_label,
                "run_date":       _dt.date.today().isoformat(),
            })

            dl_col, stat_col = st.columns([1, 2])
            with dl_col:
                st.download_button(
                    label="⬇  Download forecasts CSV",
                    data=fc_df.to_csv(index=False),
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True,
                )
            with stat_col:
                st.markdown(
                    f"<div style='font-size:0.80rem;color:#8892b0;padding-top:10px;'>"
                    f"<b style='color:#c8cdd8;'>{len(results)}</b> valid simulations  ·  "
                    f"<b style='color:#c8cdd8;'>{len(errors_fc)}</b> failed  ·  "
                    f"Model: <b style='color:#c8cdd8;'>{done_model}</b>  ·  "
                    f"Filename: <code style='font-size:0.76rem;'>{csv_filename}</code>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Show errors if any
        if errors_fc:
            with st.expander(f"⚠️ {len(errors_fc)} simulation error(s)", expanded=False):
                for err in errors_fc:
                    st.code(err, language=None)


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Social Inflation Tracker  ·  "
    f"Tracking {len(KEYWORDS_TRACKED)} keywords across "
    f"{len(SUBREDDITS_US)} US and {len(SUBREDDITS_EU)} European subreddits  ·  "
    "Data sourced from Reddit via PRAW"
)
