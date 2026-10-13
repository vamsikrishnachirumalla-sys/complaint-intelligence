# Sprint 6 (part 1) - pack the small files the dashboard needs
#
# the live dashboard can't pull 1GB+ from the CFPB every time it wakes up,
# so this writes a trimmed bundle into app_data/ that ships with the repo.
# run it after sprints 1-5
#
#   python src/sprint6_app_data.py

import shutil

import pandas as pd

from config import APP_DIR, ENRICHED_FILE, MONTHLY_FILE, RANDOM_STATE, REPORTS_DIR

APP_SAMPLE = 15_000  # enough for search to feel real, small enough for the free streamlit tier
MAX_CHARS = 1500     # long stories get trimmed, the start is where people say what happened


if __name__ == "__main__":
    df = pd.read_parquet(ENRICHED_FILE)
    df = df.sample(min(APP_SAMPLE, len(df)), random_state=RANDOM_STATE)
    df["narrative"] = df["narrative"].str.slice(0, MAX_CHARS)
    keep = ["complaint_id", "date", "product", "issue", "company", "state",
            "narrative", "topic", "sentiment"]
    df[keep].to_parquet(APP_DIR / "complaints.parquet", index=False)

    shutil.copy(MONTHLY_FILE, APP_DIR / "monthly_counts.csv")
    for name in ["rising_issues.csv", "search_metrics.json", "classifier_metrics.json",
                 "distilbert_metrics.json"]:
        src = REPORTS_DIR / name
        if src.exists():
            shutil.copy(src, APP_DIR / name)
        else:
            print(f"  {name} not found yet, the dashboard will just skip that piece")

    total = sum(f.stat().st_size for f in APP_DIR.iterdir()) / 1e6
    print(f"app_data ready: {len(df):,} complaints, {total:.1f} MB total")
