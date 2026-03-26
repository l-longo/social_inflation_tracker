"""
Social Inflation Tracker — Streamlit Dashboard
────────────────────────────────────────────────
Reads pre-processed daily mention parquets from the data/ folder
and displays interactive histograms for r/europe and r/economics.
"""

import pathlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

# ─── DATA LOADING ────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path(__file__).parent / "data"

SUBREDDITS = {
    "europe":    {"color": "#e63946", "label": "r/europe"},
    "economics": {"color": "#457b9d", "label": "r/economics"},
}

KEYWORDS_TRACKED = [
    "inflation", "hyperinflation", "disinflation",
    "deflation", "price", "prices",
]


@st.cache_data(ttl=300)
def load_data(subreddit: str) -> pd.DataFrame:
    path = DATA_DIR / f"{subreddit}_daily_mentions.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["date", "mentions_comments", "mentions_submissions", "mentions_total"])
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


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


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("# 📈 Social Inflation Tracker")
st.markdown(
    "Daily mentions of **inflation-related keywords** across Reddit communities (2026)."
)

# ─── LOAD ALL DATA ────────────────────────────────────────────────────────────
datasets = {sub: load_data(sub) for sub in selected_subs}

# Date range filter (only if we have data)
all_dates = pd.concat([d["date"] for d in datasets.values() if not d.empty], ignore_index=True)

if all_dates.empty:
    st.warning("No data found in `data/` folder. Run `prepare_and_push.py` first to generate the parquets.")
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

# Filter datasets to range
for sub in datasets:
    df = datasets[sub]
    if not df.empty:
        mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
        datasets[sub] = df.loc[mask]


# ─── KPI ROW ──────────────────────────────────────────────────────────────────
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


# ─── CHARTS ───────────────────────────────────────────────────────────────────
st.markdown("---")

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
            dtick="D1" if len(df) <= 60 else None,
            tickformat="%b %d",
        ),
        yaxis=dict(
            title="Mentions",
            gridcolor="rgba(255,255,255,0.06)",
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


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with Streamlit & Plotly · "
    f"Tracking {len(KEYWORDS_TRACKED)} keywords across {len(SUBREDDITS)} subreddits"
)
