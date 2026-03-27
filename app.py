"""
Social Inflation Tracker — Streamlit Dashboard + Chatbot
─────────────────────────────────────────────────────────
Tab 1: Daily inflation-keyword mention charts
Tab 2: AI chatbot that answers questions about Reddit conversations
"""

import pathlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Inflation Tracker",
    page_icon="📈",
    layout="wide",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="st-"] {
    font-family: 'DM Sans', sans-serif;
}
code, .stMetric [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* header bar */
header[data-testid="stHeader"] {
    background: linear-gradient(90deg, #0f0f0f 0%, #1a1a2e 100%);
}

/* metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    color: #8892b0 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricValue"] {
    color: #ccd6f6 !important;
    font-weight: 600 !important;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid #1e1e2e;
}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path(__file__).parent / "data"

SUBREDDITS = {
    "europe":    {"color": "#e63946", "label": "r/europe"},
    "economics": {"color": "#457b9d", "label": "r/economics"},
}

KEYWORDS_TRACKED = [
    "inflation", "hyperinflation", "disinflation",
    "deflation", "price", "prices",
]

LLM_MODEL = "llama-3.3-70b-instruct"
LLM_BASE_URL = "https://api-gpt.jrc.ec.europa.eu/v1"
MAX_CONTEXT_THREADS = 30  # max threads to feed as context


# ─── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_mentions(subreddit: str) -> pd.DataFrame:
    path = DATA_DIR / f"{subreddit}_daily_mentions.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["date", "mentions_comments", "mentions_submissions", "mentions_total"])
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] > pd.Timestamp("2026-03-01")]
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


def build_context_block(selected_subs: list[str], only_inflation: bool = True) -> str:
    """Build a text block of recent Reddit conversations to feed the LLM."""
    parts = []
    for sub in selected_subs:
        df = load_conversations(sub)
        if df.empty:
            continue
        if only_inflation and "has_inflation_keywords" in df.columns:
            df = df[df["has_inflation_keywords"]]
        # Sort by recency, take top N
        if "created_utc" in df.columns:
            df = df.sort_values("created_utc", ascending=False)
        df = df.head(MAX_CONTEXT_THREADS)

        for _, row in df.iterrows():
            conv = row.get("conversation", "")
            if conv:
                parts.append(f"── r/{sub} ──\n{conv}\n")

    if not parts:
        return "(No conversation data available yet.)"
    return "\n".join(parts)


def get_llm_client():
    """Create an OpenAI-compatible client using the JRC token from Streamlit secrets."""
    token = st.secrets.get("JRC_TOKEN", "")
    if not token:
        return None
    return OpenAI(api_key=token, base_url=LLM_BASE_URL)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Filters")

    selected_subs = st.multiselect(
        "Subreddits",
        options=list(SUBREDDITS.keys()),
        default=list(SUBREDDITS.keys()),
        format_func=lambda s: SUBREDDITS[s]["label"],
    )

    breakdown = st.radio(
        "Show breakdown",
        ["Total", "Comments vs Submissions"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🔎 Keywords tracked")
    st.code(", ".join(KEYWORDS_TRACKED), language=None)

    st.markdown("---")
    st.caption("Data sourced from Reddit via PRAW · Updated periodically")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_dashboard, tab_chat = st.tabs(["📊 Dashboard", "💬 Ask the data"])


# ─── TAB 1: DASHBOARD ────────────────────────────────────────────────────────
with tab_dashboard:
    st.markdown("# 📈 Social Inflation Tracker")
    st.markdown(
        "Daily mentions of **inflation-related keywords** across Reddit communities (2026)."
    )

    # Load data
    datasets = {sub: load_mentions(sub) for sub in selected_subs}
    all_dates = pd.concat([d["date"] for d in datasets.values() if not d.empty], ignore_index=True)

    if all_dates.empty:
        st.warning("No data found in `data/` folder. Run `prepare_data.py` first.")
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

    # Filter
    for sub in datasets:
        df = datasets[sub]
        if not df.empty:
            mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
            datasets[sub] = df.loc[mask]

    # KPIs
    cols = st.columns(len(selected_subs) * 3)
    col_idx = 0
    for sub in selected_subs:
        df = datasets[sub]
        label = SUBREDDITS[sub]["label"]
        total = int(df["mentions_total"].sum()) if not df.empty else 0
        avg = round(df["mentions_total"].mean(), 1) if not df.empty else 0
        peak = int(df["mentions_total"].max()) if not df.empty else 0

        cols[col_idx].metric(f"{label} — Total", f"{total:,}")
        cols[col_idx + 1].metric(f"{label} — Daily avg", f"{avg}")
        cols[col_idx + 2].metric(f"{label} — Peak day", f"{peak:,}")
        col_idx += 3

    # Charts
    st.markdown("---")

    y_max = 0
    for sub in selected_subs:
        df = datasets[sub]
        if not df.empty:
            if breakdown == "Total":
                y_max = max(y_max, df["mentions_total"].max())
            else:
                y_max = max(y_max, (df["mentions_comments"] + df["mentions_submissions"]).max())
    y_max = int(y_max * 1.1) + 1

    for sub in selected_subs:
        df = datasets[sub]
        meta = SUBREDDITS[sub]

        st.markdown(f"### {meta['label']}")

        if df.empty:
            st.info(f"No data available yet for {meta['label']}.")
            continue

        fig = go.Figure()

        if breakdown == "Total":
            fig.add_trace(go.Bar(
                x=df["date"],
                y=df["mentions_total"],
                name="Total mentions",
                marker_color=meta["color"],
                marker_line_width=0,
                opacity=0.85,
                hovertemplate="%{x|%b %d, %Y}<br>Mentions: %{y}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Bar(
                x=df["date"],
                y=df["mentions_comments"],
                name="Comments",
                marker_color=meta["color"],
                opacity=0.9,
                hovertemplate="%{x|%b %d, %Y}<br>Comments: %{y}<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                x=df["date"],
                y=df["mentions_submissions"],
                name="Submissions",
                marker_color="#f4a261",
                opacity=0.9,
                hovertemplate="%{x|%b %d, %Y}<br>Submissions: %{y}<extra></extra>",
            ))
            fig.update_layout(barmode="stack")

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#ccd6f6"),
            xaxis=dict(
                title="",
                gridcolor="rgba(255,255,255,0.04)",
                tickformat="%b %d",
            ),
            yaxis=dict(
                title="Mentions",
                gridcolor="rgba(255,255,255,0.06)",
                range=[0, y_max],
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=40, r=20, t=30, b=40),
            height=380,
        )

        st.plotly_chart(fig, use_container_width=True)


# ─── TAB 2: CHATBOT ──────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("# 💬 Ask the data")
    st.markdown(
        "Chat with an AI about recent Reddit conversations on inflation. "
        "The model has access to the latest threads from the selected subreddits."
    )

    # Check if token is configured
    client = get_llm_client()
    if client is None:
        st.error(
            "**JRC_TOKEN not configured.** "
            "Add your JWT token in Streamlit Cloud → Settings → Secrets:\n\n"
            '```toml\nJRC_TOKEN = "your-token-here"\n```'
        )
        st.stop()

    # Build context from conversation data
    context = build_context_block(selected_subs, only_inflation=True)

    SYSTEM_PROMPT = f"""You are an analyst assistant for the Social Inflation Tracker dashboard.
You have access to recent Reddit conversations (last 6 days) from the following
subreddits: {', '.join('r/' + s for s in selected_subs)}.

The conversations below discuss inflation, prices, deflation, and related
economic topics. Use them to answer the user's questions. Cite specific posts
or users when relevant. If the data doesn't contain enough information to
answer, say so honestly.

Be concise and analytical. When asked about sentiment, summarize the dominant
views and note dissenting opinions.

═══ RECENT CONVERSATIONS ═══
{context}
═══ END OF CONVERSATIONS ═══"""

    # Conversation history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about inflation discussions on Reddit..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build messages for API
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Keep last 10 turns to avoid exceeding context window
        messages += st.session_state.chat_history[-10:]

        # Call LLM
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=1000,
                    )
                    reply = response.choices[0].message.content
                except Exception as e:
                    reply = f"⚠️ LLM error: {e}"

            st.markdown(reply)

        # Save assistant reply
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # Sidebar info for chat tab
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 Chat model")
        st.code(LLM_MODEL, language=None)
        n_threads = sum(
            len(load_conversations(s)) for s in selected_subs
        )
        st.caption(f"{n_threads} conversation threads loaded")
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with Streamlit & Plotly · "
    f"Tracking {len(KEYWORDS_TRACKED)} keywords across {len(SUBREDDITS)} subreddits"
)
