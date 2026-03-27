"""
Social Inflation Tracker — Streamlit Dashboard + Chatbot
─────────────────────────────────────────────────────────
Tab 1: Daily inflation-keyword mention charts (US vs. Europe)
Tab 2: AI chatbot grounded in Reddit conversation data
"""

import html as html_module
import pathlib
import re
import pandas as pd
import plotly.graph_objects as go
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
tab_dashboard, tab_chat = st.tabs(["Dashboard", "Conversational Analysis"])


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

    import datetime as _dt
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
            import time as _time

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


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Social Inflation Tracker  ·  "
    f"Tracking {len(KEYWORDS_TRACKED)} keywords across "
    f"{len(SUBREDDITS_US)} US and {len(SUBREDDITS_EU)} European subreddits  ·  "
    "Data sourced from Reddit via PRAW"
)
