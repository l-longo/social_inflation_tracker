"""
Social Inflation Tracker — Streamlit Dashboard + Chatbot
─────────────────────────────────────────────────────────
Tab 1: Daily inflation-keyword mention charts (US vs. Europe)
Tab 2: AI chatbot grounded in Reddit conversation data
"""

import pathlib
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

/* ── App background ── */
.stApp {
    background-color: #0d0f14;
}

/* ── Top header bar ── */
header[data-testid="stHeader"] {
    background: #0d0f14;
    border-bottom: 1px solid #1e2130;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #10121a;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * {
    color: #c8cdd8 !important;
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
    font-weight: 600;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #14172080;
    border: 1px solid #252840;
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] p {
    color: #8892b0 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 500;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] {
    color: #64b5f6 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e2130;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8892b0 !important;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    padding: 10px 24px;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #4f8ef7 !important;
}

/* ── Section dividers ── */
hr {
    border-color: #1e2130 !important;
}

/* ── Region header bands ── */
.region-header-us {
    background: linear-gradient(90deg, #1a2540 0%, #0d0f14 100%);
    border-left: 3px solid #4f8ef7;
    padding: 10px 18px;
    border-radius: 4px;
    margin: 24px 0 16px 0;
}
.region-header-eu {
    background: linear-gradient(90deg, #1e1530 0%, #0d0f14 100%);
    border-left: 3px solid #a78bfa;
    padding: 10px 18px;
    border-radius: 4px;
    margin: 24px 0 16px 0;
}
.region-header-us h2, .region-header-eu h2 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #ffffff !important;
}
.region-header-us p, .region-header-eu p {
    margin: 2px 0 0 0;
    font-size: 0.78rem;
    color: #8892b0 !important;
}

/* ── Chat sample question buttons ── */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #14172080;
    border: 1px solid #2a2f4a;
    border-radius: 6px;
    color: #c8cdd8 !important;
    font-size: 0.80rem;
    font-weight: 400;
    padding: 8px 14px;
    text-align: left;
    white-space: normal;
    line-height: 1.4;
    min-height: 60px;
    width: 100%;
    transition: border-color 0.15s, background 0.15s;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #1c2035;
    border-color: #4f8ef7;
    color: #ffffff !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #14172060 !important;
    border: 1px solid #1e2130;
    border-radius: 8px;
    margin-bottom: 8px;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #e8eaf0 !important;
}

/* ── Selectbox / radio labels ── */
.stSelectbox label, .stRadio label, .stMultiSelect label, .stSlider label {
    color: #c8cdd8 !important;
    font-size: 0.82rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Info / warning boxes ── */
.stAlert {
    background: #14172080 !important;
    border-radius: 6px;
}
.stAlert p {
    color: #e8eaf0 !important;
}

/* ── Code blocks ── */
.stCodeBlock, code {
    background: #0a0c12 !important;
    color: #a8d8f0 !important;
}

/* ── Page title ── */
h1 {
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}
h2, h3 {
    color: #e0e4f0 !important;
    font-weight: 600 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a2f4a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path(__file__).parent / "data"

# US-based communities
SUBREDDITS_US = {
    "economy":    {"color": "#4f8ef7", "label": "r/economy"},
    "economics":  {"color": "#38bdf8", "label": "r/economics"},
}

# European communities
SUBREDDITS_EU = {
    "europe":    {"color": "#a78bfa", "label": "r/europe"},
    "italy":     {"color": "#fb7185", "label": "r/italy"},
    "spain":     {"color": "#fbbf24", "label": "r/spain"},
    "germany":   {"color": "#34d399", "label": "r/germany"},
    "france":    {"color": "#f97316", "label": "r/france"},
}

ALL_SUBREDDITS = {**SUBREDDITS_US, **SUBREDDITS_EU}

KEYWORDS_TRACKED = [
    "inflation", "hyperinflation", "disinflation",
    "deflation", "price", "prices",
]

LLM_MODELS = {
    "gpt-oss-120b":               "GPT OSS 120B",
    "llama-3.3-70b-instruct-ui":  "Llama 3.3 70B Instruct",
    "minimax-m2":                 "MiniMax M2",
    "mistral-small-3.2-24b":      "Mistral Small 3.2 24B",
}

LLM_BASE_URL = "https://api-gpt.jrc.ec.europa.eu/v1"
MAX_CONTEXT_THREADS = 30

SAMPLE_QUESTIONS = [
    "Are users expecting higher or lower inflation in the coming months?",
    "What is the general sentiment of users toward current price levels?",
    "Are users speculating on central bank decisions or monetary policy?",
    "Which economic concerns are mentioned most frequently?",
    "Are users discussing any specific goods or sectors affected by inflation?",
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
    df = df[df["date"] > pd.Timestamp("2026-03-07")]
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


def build_context_block(subs: list[str], only_inflation: bool = True) -> str:
    """Assemble recent Reddit threads into a text block for the LLM."""
    parts = []
    for sub in subs:
        df = load_conversations(sub)
        if df.empty:
            continue
        if only_inflation and "has_inflation_keywords" in df.columns:
            df = df[df["has_inflation_keywords"]]
        if "created_utc" in df.columns:
            df = df.sort_values("created_utc", ascending=False)
        df = df.head(MAX_CONTEXT_THREADS)
        for _, row in df.iterrows():
            conv = row.get("conversation", "")
            if conv:
                parts.append(f"── r/{sub} ──\n{conv}\n")
    return "\n".join(parts) if parts else "(No conversation data available yet.)"


def get_llm_client() -> OpenAI | None:
    token = st.secrets.get("JRC_TOKEN", "")
    if not token:
        return None
    return OpenAI(api_key=token, base_url=LLM_BASE_URL)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")
    st.markdown("---")

    st.markdown("**United States**")
    selected_us = st.multiselect(
        "US subreddits",
        options=list(SUBREDDITS_US.keys()),
        default=list(SUBREDDITS_US.keys()),
        format_func=lambda s: SUBREDDITS_US[s]["label"],
        label_visibility="collapsed",
    )

    st.markdown("**Europe**")
    selected_eu = st.multiselect(
        "EU subreddits",
        options=list(SUBREDDITS_EU.keys()),
        default=list(SUBREDDITS_EU.keys()),
        format_func=lambda s: SUBREDDITS_EU[s]["label"],
        label_visibility="collapsed",
    )

    selected_subs = selected_us + selected_eu

    st.markdown("---")

    breakdown = st.radio(
        "Breakdown",
        ["Total", "Comments vs. Submissions"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Keywords tracked**")
    st.code(", ".join(KEYWORDS_TRACKED), language=None)

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
        "Daily volume of inflation-related keyword mentions across Reddit communities (2026). "
        "Data is segmented by geographic focus: United States and Europe."
    )

    # Load all data
    datasets = {sub: load_mentions(sub) for sub in selected_subs}
    all_dates = pd.concat(
        [d["date"] for d in datasets.values() if not d.empty], ignore_index=True
    )

    if all_dates.empty:
        st.warning("No data found in the `data/` directory. Run the data preparation scripts first.")
        st.stop()

    date_min, date_max = all_dates.min().date(), all_dates.max().date()

    if date_min < date_max:
        date_range = st.slider(
            "Date range",
            min_value=date_min,
            max_value=date_max,
            value=(date_min, date_max),
            format="YYYY-MM-DD",
        )
    else:
        date_range = (date_min, date_max)

    # Apply date filter
    for sub in list(datasets.keys()):
        df = datasets[sub]
        if not df.empty:
            mask = (
                (df["date"].dt.date >= date_range[0])
                & (df["date"].dt.date <= date_range[1])
            )
            datasets[sub] = df.loc[mask]

    # Shared Y-axis maximum across all charts for comparability
    y_max = 0
    for sub in selected_subs:
        df = datasets[sub]
        if not df.empty:
            col = "mentions_total" if breakdown == "Total" else "mentions_comments"
            y_max = max(y_max, df["mentions_total"].max())
    y_max = int(y_max * 1.15) + 1

    def render_region_section(region_subs: list[str], region_meta: dict, header_class: str, region_name: str, description: str):
        """Render KPIs + charts for one geographic group."""
        active = [s for s in region_subs if s in datasets]
        if not active:
            return

        st.markdown(
            f'<div class="{header_class}"><h2>{region_name}</h2><p>{description}</p></div>',
            unsafe_allow_html=True,
        )

        # KPIs — one group of 3 per subreddit
        kpi_cols = st.columns(len(active) * 3)
        for i, sub in enumerate(active):
            df = datasets[sub]
            label = region_meta[sub]["label"]
            total = int(df["mentions_total"].sum()) if not df.empty else 0
            avg   = round(df["mentions_total"].mean(), 1) if not df.empty else 0.0
            peak  = int(df["mentions_total"].max()) if not df.empty else 0

            kpi_cols[i * 3 + 0].metric(f"{label}  ·  Total",     f"{total:,}")
            kpi_cols[i * 3 + 1].metric(f"{label}  ·  Daily avg", f"{avg:,}")
            kpi_cols[i * 3 + 2].metric(f"{label}  ·  Peak day",  f"{peak:,}")

        st.markdown("")  # spacing

        # Charts
        for sub in active:
            df = datasets[sub]
            meta = region_meta[sub]

            st.markdown(f"##### {meta['label']}")

            if df.empty:
                st.info(f"No data available for {meta['label']} in the selected date range.")
                continue

            fig = go.Figure()

            if breakdown == "Total":
                fig.add_trace(go.Bar(
                    x=df["date"],
                    y=df["mentions_total"],
                    name="Total mentions",
                    marker_color=meta["color"],
                    marker_line_width=0,
                    opacity=0.9,
                    hovertemplate="%{x|%b %d, %Y}<br>Mentions: <b>%{y}</b><extra></extra>",
                ))
            else:
                fig.add_trace(go.Bar(
                    x=df["date"],
                    y=df["mentions_comments"],
                    name="Comments",
                    marker_color=meta["color"],
                    opacity=0.9,
                    hovertemplate="%{x|%b %d, %Y}<br>Comments: <b>%{y}</b><extra></extra>",
                ))
                fig.add_trace(go.Bar(
                    x=df["date"],
                    y=df["mentions_submissions"],
                    name="Submissions",
                    marker_color="#475569",
                    opacity=0.85,
                    hovertemplate="%{x|%b %d, %Y}<br>Submissions: <b>%{y}</b><extra></extra>",
                ))
                fig.update_layout(barmode="stack")

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#c8cdd8", size=12),
                xaxis=dict(
                    title="",
                    gridcolor="rgba(255,255,255,0.04)",
                    tickformat="%b %d",
                    tickcolor="#3a3f55",
                    linecolor="#1e2130",
                ),
                yaxis=dict(
                    title="Keyword mentions",
                    gridcolor="rgba(255,255,255,0.06)",
                    tickcolor="#3a3f55",
                    linecolor="#1e2130",
                    range=[0, y_max],
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=11),
                ),
                margin=dict(l=50, r=20, t=36, b=40),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Render US section
    render_region_section(
        region_subs=selected_us,
        region_meta=SUBREDDITS_US,
        header_class="region-header-us",
        region_name="United States",
        description="r/economy · r/economics",
    )

    # Render EU section
    render_region_section(
        region_subs=selected_eu,
        region_meta=SUBREDDITS_EU,
        header_class="region-header-eu",
        region_name="Europe",
        description="r/europe · r/italy · r/spain · r/germany · r/france",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONVERSATIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("# Conversational Analysis")
    st.markdown(
        "Query an AI analyst grounded in the most recent Reddit discussions "
        "on inflation and prices. The model cites specific threads and users when relevant."
    )

    # ── Token check ──────────────────────────────────────────────────────────
    client = get_llm_client()
    if client is None:
        st.error(
            "JRC_TOKEN is not configured. "
            "Add your JWT token under Streamlit Cloud → Settings → Secrets:\n\n"
            '```toml\nJRC_TOKEN = "your-token-here"\n```'
        )
        st.stop()

    # ── Chat configuration ────────────────────────────────────────────────────
    cfg_col1, cfg_col2 = st.columns([1, 2])

    with cfg_col1:
        region_choice = st.radio(
            "Community scope",
            options=["United States", "Europe", "Both"],
            index=2,
            horizontal=False,
        )

    with cfg_col2:
        selected_model_key = st.selectbox(
            "Language model",
            options=list(LLM_MODELS.keys()),
            format_func=lambda k: LLM_MODELS[k],
            index=1,  # default: Llama 3.3 70B
        )

    # Determine which subreddits to ingest based on region choice
    if region_choice == "United States":
        chat_subs = [s for s in selected_us if s in SUBREDDITS_US]
    elif region_choice == "Europe":
        chat_subs = [s for s in selected_eu if s in SUBREDDITS_EU]
    else:
        chat_subs = selected_us + selected_eu

    st.markdown("---")

    # ── Session state initialisation ─────────────────────────────────────────
    # Reset chat when region or model changes to avoid stale context
    state_key = f"chat_{region_choice}_{selected_model_key}"
    if "active_chat_key" not in st.session_state or st.session_state.active_chat_key != state_key:
        st.session_state.active_chat_key = state_key
        st.session_state.chat_history = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # ── Sample questions ─────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("**Suggested questions**")
        btn_cols = st.columns(len(SAMPLE_QUESTIONS))
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if btn_cols[i].button(q, key=f"sq_{i}"):
                st.session_state.pending_question = q
                st.rerun()
        st.markdown("")

    # ── Render existing conversation ─────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Determine input (typed or from sample button) ─────────────────────────
    user_input: str | None = None

    typed = st.chat_input("Type a question about inflation discussions...")
    if typed:
        user_input = typed
    elif st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None

    # ── Process input ─────────────────────────────────────────────────────────
    if user_input:
        # Display user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build system prompt (context is refreshed each call so it stays current)
        context = build_context_block(chat_subs, only_inflation=True)
        sub_list = ", ".join(f"r/{s}" for s in chat_subs)
        system_prompt = f"""You are a senior economic analyst assistant embedded in the Social Inflation Tracker.
You have access to recent Reddit conversations (last several days) from: {sub_list}.

These threads discuss inflation, price levels, monetary policy, and related economic topics.
Use them as your primary evidence base. Cite specific posts or users when relevant.
If the data does not contain sufficient information to answer a question, state that clearly.

Be analytical and concise. When assessing sentiment, distinguish dominant views from minority ones.
Avoid speculation beyond what the data supports.

═══ RECENT CONVERSATIONS ═══
{context}
═══ END OF CONVERSATIONS ═══"""

        # Compose messages: system + full conversation history
        api_messages = [{"role": "system", "content": system_prompt}]
        # Include all turns (the LLM has a large context window; truncate only as a safeguard)
        api_messages += st.session_state.chat_history[-20:]

        # Call LLM and stream reply
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                try:
                    response = client.chat.completions.create(
                        model=selected_model_key,
                        messages=api_messages,
                        temperature=0.3,
                        max_tokens=1200,
                    )
                    reply = response.choices[0].message.content
                except Exception as exc:
                    reply = f"Error communicating with the model: {exc}"
            st.markdown(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # ── Sidebar addenda for chat tab ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Active model**")
        st.code(selected_model_key, language=None)

        n_threads = sum(len(load_conversations(s)) for s in chat_subs)
        infl_threads = sum(
            int(load_conversations(s).get("has_inflation_keywords", pd.Series(dtype=bool)).sum())
            if not load_conversations(s).empty and "has_inflation_keywords" in load_conversations(s).columns
            else 0
            for s in chat_subs
        )
        st.caption(f"{n_threads} threads loaded · {infl_threads} with inflation keywords")

        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Social Inflation Tracker · "
    f"Tracking {len(KEYWORDS_TRACKED)} keywords across "
    f"{len(SUBREDDITS_US)} US and {len(SUBREDDITS_EU)} European subreddits · "
    "Data sourced from Reddit via PRAW"
)
