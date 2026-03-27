"""
prepare_data.py  —  Step 1 (run at JRC)
───────────────────────────────────────
Reads raw Reddit parquet files (2026 only), computes daily
inflation-keyword counts, and saves processed parquets locally.

Run push_to_github.py afterwards from outside the JRC proxy.
"""

import os
import re
import glob
import pandas as pd

# ─── CONFIG ───────────────────────────────────────────────────────────────────
RAW_DATA_PATH = "X:/Reddit/monthly_series/data_parquet/"
OUTPUT_PATH   = "X:/Reddit/dashboard_data/"  # processed files go here

SUBREDDITS = ["economy", "economics", 
              "europe", "italy", "spain", "germany", "france"]

KEYWORDS = [
    # English
    "inflation", "hyperinflation", "disinflation",
    "deflation", "price", "prices",

    # Italian
    "inflazione", "iperinflazione", "disinflazione",
    "deflazione", "prezzo", "prezzi",

    # German
    "inflation", "hyperinflation", "disinflation",
    "deflation", "preis", "preise",

    # French
    "inflation", "hyperinflation", "désinflation",
    "déflation", "prix",

    # Spanish
    "inflación", "hiperinflación", "desinflación",
    "deflación", "precio", "precios",
]


KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in KEYWORDS) + r")\b",
    re.IGNORECASE,
)


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def load_parquets_2026(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    files = [f for f in files if "2026" in os.path.basename(f)]
    if not files:
        print(f"  ⚠  No 2026 files for: {pattern}")
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    print(f"  Loaded {len(files)} file(s)")
    return pd.concat(dfs, ignore_index=True)


def count_keywords(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(KEYWORD_PATTERN.findall(text))


def build_daily(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "kw_count"])
    df = df.copy()
    df = df[df["created_utc"].dt.year >= 2026]  # filter by actual date, not filename
    # Use normalize() to keep proper datetime64 dtype (midnight timestamps).
    # dt.date produces Python datetime.date objects stored as object dtype in
    # parquet, which breaks date comparisons in the dashboard.
    df["date"] = df["created_utc"].dt.normalize()
    df["kw_count"] = df[text_col].apply(count_keywords)
    return df.groupby("date")["kw_count"].sum().reset_index()


# ─── PROCESS ──────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_PATH, exist_ok=True)

for subreddit in SUBREDDITS:
    print(f"\n📂  r/{subreddit}")
    sub_dir = os.path.join(RAW_DATA_PATH, subreddit)

    # Comments
    comments = load_parquets_2026(os.path.join(sub_dir, f"{subreddit}_comments_*.parquet"))
    if not comments.empty:
        comments.drop_duplicates(subset="id", inplace=True)
    daily_c = build_daily(comments, "body")
    daily_c.columns = ["date", "mentions_comments"]

    # Submissions
    submissions = load_parquets_2026(os.path.join(sub_dir, f"{subreddit}_submissions_*.parquet"))
    if not submissions.empty:
        submissions.drop_duplicates(subset="id", inplace=True)
    daily_s = build_daily(submissions, "title")
    daily_s.columns = ["date", "mentions_submissions"]

    # Merge
    daily = pd.merge(daily_c, daily_s, on="date", how="outer").fillna(0)
    daily["mentions_comments"] = daily["mentions_comments"].astype(int)
    daily["mentions_submissions"] = daily["mentions_submissions"].astype(int)
    daily["mentions_total"] = daily["mentions_comments"] + daily["mentions_submissions"]
    daily = daily.sort_values("date").reset_index(drop=True)

    out_path = os.path.join(OUTPUT_PATH, f"{subreddit}_daily_mentions.parquet")
    daily.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"  ✅ Saved {out_path}  ({len(daily)} days, {daily['mentions_total'].sum()} mentions)")

print("\n🎉  Done! Now run push_to_github.py from outside the proxy.")
