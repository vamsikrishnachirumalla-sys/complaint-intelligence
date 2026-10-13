# Sprint 2 - what are people actually complaining about?
#
# NLP side:     tf-idf, topic modeling, and the words that set each product apart
# social side:  sentiment, volume trends, and which issues are spiking right now
#               (same playbook a brand team uses to watch twitter/reddit chatter)
#
#   python src/sprint2_text_analytics.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.feature_selection import chi2
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import ENRICHED_FILE, FIGURES_DIR, MONTHLY_FILE, RANDOM_STATE, REPORTS_DIR, SAMPLE_FILE

N_TOPICS = 10

# words that show up in nearly every complaint and say nothing about the problem.
# found these by looking at the first topic model run, they were in every topic
EXTRA_STOP = {"told", "said", "called", "call", "calls", "asked", "contacted", "received",
              "company", "help", "please", "time", "times", "day", "days", "just", "did",
              "didn", "don", "doesn", "won", "ve", "ll", "know", "want", "get", "got",
              "going", "make", "able", "need", "thank", "like", "also", "still", "would"}
STOP = list(ENGLISH_STOP_WORDS | EXTRA_STOP)

BLUE, RED, GREY = "#1f6f8b", "#d1495b", "#9aa5b1"


def style(ax, title, xlabel=None):
    # title goes on the figure, pinned to the left edge. on the axis it shifts
    # right whenever the bar labels are long, and sometimes runs off the page
    ax.figure.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.spines[["top", "right"]].set_visible(False)


def topics(df):
    # min_df=5 drops typos and one-off names, max_df=0.5 drops words in half of all complaints
    vec = TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=5, max_df=0.5,
                          max_features=20_000, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = vec.fit_transform(df["narrative"])
    words = np.array(vec.get_feature_names_out())

    # NMF over LDA here: on tf-idf it gives cleaner, more readable topics,
    # and it's way faster on a few hundred thousand complaints
    n = min(N_TOPICS, X.shape[1] - 1)
    nmf = NMF(n_components=n, random_state=RANDOM_STATE, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)

    labels = []
    for row in nmf.components_:
        # skip terms that only repeat words already in the label,
        # otherwise you get "late payment, late, payment" which reads badly
        picked, seen = [], set()
        for term in words[row.argsort()[::-1][:30]]:
            if set(term.split()) <= seen:
                continue
            picked.append(term)
            seen |= set(term.split())
            if len(picked) == 4:
                break
        labels.append(", ".join(picked))

    df = df.assign(topic_id=W.argmax(axis=1))
    df["topic"] = df["topic_id"].map(dict(enumerate(labels)))
    return df, X, words, labels


def distinctive_words(X, words, products):
    # chi-square picks the words most tied to one product vs all the others.
    # this is the "what makes mortgage complaints sound like mortgage complaints" view
    out = {}
    for p in sorted(products.unique()):
        scores, _ = chi2(X, (products == p).values)
        scores = np.nan_to_num(scores)
        out[p] = ", ".join(words[scores.argsort()[::-1][:8]])
    return out


# TODO: try lemmatizing before tf-idf, "charge" / "charged" / "charges" split into 3 words right now
def add_sentiment(df):
    # VADER was built for social media text, short and emotional, which is
    # pretty much what these complaints are. compound runs -1 (angry) to +1
    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = df["narrative"].map(lambda t: sia.polarity_scores(t[:2000])["compound"])
    return df


def spikes(monthly):
    # compare the last 3 months against the 6 before them, per product + issue.
    # min volume so a jump from 2 to 6 complaints doesn't top the list
    months = sorted(monthly["month"].unique())
    recent, before = months[-3:], months[-9:-3]
    g = monthly.groupby(["product", "issue", "month"])["complaints"].sum().unstack(fill_value=0)

    out = pd.DataFrame({
        "recent_avg": g.reindex(columns=recent, fill_value=0).mean(axis=1),
        "before_avg": g.reindex(columns=before, fill_value=0).mean(axis=1),
    })
    out = out[out["before_avg"] >= 20]
    out["growth"] = out["recent_avg"] / out["before_avg"] - 1
    return out.sort_values("growth", ascending=False).reset_index(), recent, before


def chart_trend(monthly):
    totals = monthly.groupby(["month", "product"])["complaints"].sum().unstack(fill_value=0)
    top = totals.sum().sort_values(ascending=False).index[:5]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, p in enumerate(top):
        ax.plot(totals.index, totals[p], label=p, linewidth=2 if i == 0 else 1.5)
    # log scale, otherwise credit reporting (~88% of everything) flattens
    # the other products into a line at the bottom and the jan 2025 spike disappears
    ax.set_yscale("log")
    ax.set_xticks(totals.index[::3])
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("Complaints per month (log scale)")
    style(ax, "Monthly complaint volume, top 5 products")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_monthly_volume.png", dpi=150)
    plt.close(fig)


def chart_topics(df):
    share = df["topic"].value_counts(normalize=True).sort_values()
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(share) + 1.2))
    ax.barh(share.index, share * 100, color=BLUE)
    for i, v in enumerate(share):
        ax.text(v * 100 + 0.3, i, f"{v:.1%}", va="center", fontsize=8)
    style(ax, "What complaints are about (topic model)", "Share of complaints (%)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_topics.png", dpi=150)
    plt.close(fig)


def chart_sentiment(df):
    by = df.groupby("product")["sentiment"].mean().sort_values()
    overall = df["sentiment"].mean()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(by) + 1.2))
    ax.barh(by.index, by, color=[RED if v < overall else BLUE for v in by])
    ax.axvline(overall, color=GREY, linestyle="--", linewidth=1, label=f"overall {overall:.2f}")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), frameon=False, fontsize=9)
    style(ax, "Average sentiment by product (lower = angrier)", "VADER compound score")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_sentiment_by_product.png", dpi=150)
    plt.close(fig)


def chart_spikes(sp):
    top = sp.head(8).iloc[::-1]
    # cut long issue names at a word, with "..." so it's obvious they were trimmed
    short = lambda t, n=42: t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "..."
    names = [f"{p}: {short(i)}" for p, i in zip(top["product"], top["issue"])]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(top) + 1.2))
    ax.barh(names, top["growth"] * 100, color=[RED if g > 0 else BLUE for g in top["growth"]])
    for i, g in enumerate(top["growth"]):
        ax.text(g * 100 + 1, i, f"{g:+.0%}", va="center", fontsize=8)
    style(ax, "Fastest-growing issues (last 3 months vs the 6 before)", "Change in monthly volume (%)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_rising_issues.png", dpi=150)
    plt.close(fig)


def write_findings(df, monthly, labels, words_by_product, sp, recent, before):
    total = monthly["complaints"].sum()
    by_prod = monthly.groupby("product")["complaints"].sum().sort_values(ascending=False)
    angriest = df.groupby("product")["sentiment"].mean().sort_values()
    top_topic = df["topic"].value_counts(normalize=True)

    out = [
        "# Findings Summary (Sprint 2)",
        "",
        f"**{total:,} complaints** in the window. "
        f"**{by_prod.index[0]}** alone is {by_prod.iloc[0] / total:.0%} of them.",
        "",
        "## What stands out",
        "",
        f"1. **The biggest theme is \"{top_topic.index[0]}\"**, covering {top_topic.iloc[0]:.0%} of written complaints.",
        f"2. **{angriest.index[0]} complaints are the angriest** (average sentiment {angriest.iloc[0]:.2f}, "
        f"vs {df['sentiment'].mean():.2f} overall).",
    ]
    if len(sp):
        r = sp.iloc[0]
        out.append(f"3. **Fastest-growing issue: {r['product']}, \"{r['issue']}\"**, up {r['growth']:.0%} "
                   f"({r['before_avg']:.0f} to {r['recent_avg']:.0f} complaints a month).")

    out += ["", "## Topics found", "", "| # | Top words | Share |", "|---|---|---|"]
    for i, lab in enumerate(labels):
        out.append(f"| {i + 1} | {lab} | {(df['topic'] == lab).mean():.1%} |")

    out += ["", "## Words that set each product apart", "", "| Product | Most distinctive words |", "|---|---|"]
    for p, w in words_by_product.items():
        out.append(f"| {p} | {w} |")

    out += ["", f"## Rising issues ({', '.join(recent)} vs {before[0]} to {before[-1]})", "",
            "| Product | Issue | Before (avg/month) | Recent (avg/month) | Change |", "|---|---|---|---|---|"]
    for _, r in sp.head(10).iterrows():
        out.append(f"| {r['product']} | {r['issue']} | {r['before_avg']:.0f} | {r['recent_avg']:.0f} | {r['growth']:+.0%} |")

    out += ["", "## Caveats", "",
            "Sentiment scores come from a general-purpose lexicon, so they track tone, not whether the "
            "complaint is valid. Complaint volume reflects who files with the CFPB, not every unhappy customer."]
    (REPORTS_DIR / "findings_summary.md").write_text("\n".join(out))


if __name__ == "__main__":
    df = pd.read_parquet(SAMPLE_FILE)
    monthly = pd.read_csv(MONTHLY_FILE)

    df, X, words, labels = topics(df)
    words_by_product = distinctive_words(X, words, df["product"])
    df = add_sentiment(df)
    sp, recent, before = spikes(monthly)

    df.to_parquet(ENRICHED_FILE, index=False)
    sp.to_csv(REPORTS_DIR / "rising_issues.csv", index=False)

    chart_trend(monthly)
    chart_topics(df)
    chart_sentiment(df)
    chart_spikes(sp)
    write_findings(df, monthly, labels, words_by_product, sp, recent, before)

    print("topics:")
    for lab in labels:
        print("  -", lab)
    print("\naverage sentiment by product:")
    print(df.groupby("product")["sentiment"].mean().sort_values().round(3).to_string())
    print("\ntop rising issues:")
    print(sp.head(5)[["product", "issue", "growth"]].to_string(index=False))
