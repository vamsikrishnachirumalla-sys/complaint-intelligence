# Sprint 6 - the dashboard
#
# five tabs:
#   trends      -> what's growing (social listening view)
#   topics      -> what people talk about and how angry they are
#   search      -> find similar past complaints (BM25, built from scratch)
#   auto-route  -> paste a complaint, the numpy network picks the team
#   models      -> how good each piece actually is
#
#   streamlit run app.py

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "src"))
from search import BM25, TfidfSearch, top_k

APP = ROOT / "app_data"
FIG = ROOT / "reports" / "figures"

st.set_page_config(page_title="Complaint Intelligence", layout="wide")


@st.cache_data
def load():
    df = pd.read_parquet(APP / "complaints.parquet")
    monthly = pd.read_csv(APP / "monthly_counts.csv")
    rising = pd.read_csv(APP / "rising_issues.csv") if (APP / "rising_issues.csv").exists() else None
    return df, monthly, rising


@st.cache_resource
def engines(_docs):
    # built once per app start, takes a couple of seconds for 15k complaints
    # leading underscore tells streamlit not to hash 15k complaints on every call
    return {"BM25": BM25().fit(_docs), "TF-IDF": TfidfSearch().fit(_docs)}


@st.cache_resource
def router():
    return joblib.load(APP / "router.joblib")


def route(text, r):
    # same math as the numpy network in sprint 4, just the forward pass
    x = r["tfidf"].transform([text]) @ r["svd_components"].T
    x = (np.asarray(x) - r["mu"]) / r["sd"]
    h = np.maximum(x @ r["W1"] + r["b1"], 0)
    logits = h @ r["W2"] + r["b2"]
    p = np.exp(logits - logits.max())
    p = (p / p.sum()).ravel()
    return pd.Series(p, index=r["classes"]).sort_values(ascending=False)


def read_json(name):
    path = APP / name
    return json.loads(path.read_text()) if path.exists() else None


if not (APP / "complaints.parquet").exists():
    st.error("No app data yet. Run sprints 1-5, then `python src/sprint6_app_data.py`.")
    st.stop()

df, monthly, rising = load()
docs = df["narrative"].tolist()

st.title("Customer Complaint Intelligence")
st.caption("Built on public complaints from the CFPB Consumer Complaint Database. "
           "Names and account details are redacted by the CFPB before publishing.")

tab_trend, tab_topic, tab_search, tab_route, tab_model = st.tabs(
    ["Trends", "Topics and sentiment", "Search past complaints", "Auto-route a complaint", "How good is it?"])


with tab_trend:
    total = int(monthly["complaints"].sum())
    by_prod = monthly.groupby("product")["complaints"].sum().sort_values(ascending=False)

    k1, k2, k3 = st.columns(3)
    k1.metric("Complaints in the window", f"{total:,}")
    k2.metric(f"Share that are {by_prod.index[0].lower()}", f"{by_prod.iloc[0] / total:.0%}")
    if rising is not None and len(rising):
        top = rising.iloc[0]
        k3.metric("Fastest-rising issue", f"{top['growth']:+.0%}", f"{top['product']}: {top['issue']}",
                  delta_color="off")

    st.markdown("**Complaints per month**")
    pick = st.multiselect("Products", list(by_prod.index), default=list(by_prod.index[:5]))
    trend = (monthly[monthly["product"].isin(pick)]
             .groupby(["month", "product"])["complaints"].sum().unstack(fill_value=0))
    st.line_chart(trend)

    if rising is not None:
        st.markdown("**Issues growing fastest** (last 3 months vs the 6 before)")
        show = rising.head(15).rename(columns={"product": "Product", "issue": "Issue",
                                                "before_avg": "Before (per month)",
                                                "recent_avg": "Recent (per month)", "growth": "Change"})
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config={"Change": st.column_config.NumberColumn(format="percent"),
                                    "Before (per month)": st.column_config.NumberColumn(format="%.0f"),
                                    "Recent (per month)": st.column_config.NumberColumn(format="%.0f")})


with tab_topic:
    product = st.selectbox("Product", ["All products"] + sorted(df["product"].unique()))
    view = df if product == "All products" else df[df["product"] == product]

    left, right = st.columns(2)
    with left:
        st.markdown("**What the complaints are about**")
        st.bar_chart(view["topic"].value_counts(normalize=True).mul(100).sort_values(), horizontal=True,
                     x_label="Share of complaints (%)")
    with right:
        st.markdown("**Average sentiment by product** (lower = angrier)")
        st.bar_chart(df.groupby("product")["sentiment"].mean().sort_values(), horizontal=True,
                     x_label="VADER compound score")

    st.markdown("**Angriest complaints in this view**")
    for _, r in view.nsmallest(3, "sentiment").iterrows():
        with st.expander(f"{r['product']} · {r['issue']} · sentiment {r['sentiment']:.2f}"):
            st.write(r["narrative"])


with tab_search:
    st.markdown("Describe a problem in your own words and find the closest past complaints.")
    c1, c2 = st.columns([3, 1])
    query = c1.text_input("Search", "someone used my debit card and the bank won't refund me")
    method = c2.radio("Method", ["BM25", "TF-IDF"], horizontal=True)

    if query.strip():
        scores = engines(docs)[method].scores(query)
        hits = top_k(scores, k=10)
        hits = [i for i in hits if scores[i] > 0]
        if not hits:
            st.info("No matches. Try different words.")
        for rank, i in enumerate(hits, 1):
            r = df.iloc[i]
            with st.expander(f"{rank}. {r['product']} · {r['issue']} · {r['company']} · "
                             f"{pd.to_datetime(r['date']):%b %Y}"):
                st.write(r["narrative"])


with tab_route:
    st.markdown("Paste a complaint and the neural network (written from scratch in NumPy) "
                "picks which product team should handle it.")
    text = st.text_area("Complaint", "I paid off my car loan in full but the lender still reports me as "
                        "late and won't release the title.", height=120)
    if text.strip():
        probs = route(text, router())
        best = probs.index[0]
        st.metric("Send to", best, f"{probs.iloc[0]:.0%} confident", delta_color="off")
        if probs.iloc[0] < 0.5:
            st.warning("Low confidence. In a real queue this would go to a person to check.")
        st.bar_chart(probs.head(5).mul(100).sort_values(), horizontal=True, x_label="Probability (%)")

        st.markdown("**Similar past complaints**")
        for i in top_k(engines(docs)["BM25"].scores(text), k=3):
            r = df.iloc[i]
            st.caption(f"{r['product']} · {r['issue']}")
            st.write(r["narrative"][:300] + "...")


with tab_model:
    clf, bert, srch = (read_json(n) for n in
                       ["classifier_metrics.json", "distilbert_metrics.json", "search_metrics.json"])

    st.subheader("Routing complaints to the right team")
    rows = []
    if clf:
        rows.append({"Model": "Logistic regression (baseline)", **clf["logistic_regression"]})
        rows.append({"Model": "Neural net, NumPy from scratch", **clf["numpy_neural_net"]})
    if bert:
        rows.append({"Model": "DistilBERT, fine-tuned", **bert})
    if rows:
        t = pd.DataFrame(rows)[["Model", "accuracy", "macro_f1"]]
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={"accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
                                    "macro_f1": st.column_config.NumberColumn("Macro F1", format="percent")})
    if not bert:
        st.caption("DistilBERT results show up here after sprint 5 runs on a GPU.")

    st.subheader("Finding similar complaints")
    if srch:
        s = pd.DataFrame(srch).T.reset_index().rename(columns={"index": "Method"})
        st.dataframe(s, hide_index=True, width="stretch",
                     column_config={"precision_at_10": st.column_config.NumberColumn("Precision@10", format="percent"),
                                    "product_precision_at_10": st.column_config.NumberColumn("Same-product P@10", format="percent"),
                                    "mrr": st.column_config.NumberColumn("MRR", format="%.3f"),
                                    "ms_per_query": st.column_config.NumberColumn("ms per query", format="%.1f")})
        st.caption("A result counts as relevant when it has the same product and issue as the "
                   "complaint the query came from.")

    imgs = sorted(FIG.glob("*.png"))
    if imgs:
        st.subheader("Charts from the analysis")
        cols = st.columns(2)
        for i, img in enumerate(imgs):
            cols[i % 2].image(str(img), width="stretch")
