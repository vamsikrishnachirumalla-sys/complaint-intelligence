# Sprint 3 - a search engine for complaints
#
# use case: an agent gets a new complaint and wants to see how similar past
# cases were handled. so the question is which search method finds the most
# useful matches. comparing three:
#   tf-idf cosine  (the classic vector space model)
#   BM25           (what most real search engines still start from)
#   embeddings     (meaning-based, needs sentence-transformers)
#
#   python src/sprint3_search.py

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ENRICHED_FILE, FIGURES_DIR, RANDOM_STATE, REPORTS_DIR
from search import BM25, EmbeddingSearch, TfidfSearch, top_k

N_QUERIES = 300
QUERY_WORDS = 25  # people type short descriptions, not whole stories
K = 10


def make_queries(df):
    rng = np.random.default_rng(RANDOM_STATE)
    picks = rng.choice(len(df), size=min(N_QUERIES, len(df)), replace=False)
    texts = [" ".join(df["narrative"].iloc[i].split()[:QUERY_WORDS]) for i in picks]
    return picks, texts


def evaluate(engine, df, picks, texts):
    # no human relevance labels, so "relevant" = same product AND same issue.
    # strict, but it's the same rule for every method so the comparison is fair
    product = df["product"].to_numpy()
    issue = df["issue"].to_numpy()

    p_at_k, product_p, rr, latency = [], [], [], []
    for qi, text in zip(picks, texts):
        t0 = time.perf_counter()
        hits = top_k(engine.scores(text), k=K, skip=qi)
        latency.append(time.perf_counter() - t0)

        relevant = (product[hits] == product[qi]) & (issue[hits] == issue[qi])
        p_at_k.append(relevant.mean())
        product_p.append((product[hits] == product[qi]).mean())
        first = np.flatnonzero(relevant)
        rr.append(1 / (first[0] + 1) if len(first) else 0)  # reciprocal rank

    return {
        "precision_at_10": round(float(np.mean(p_at_k)), 3),
        "product_precision_at_10": round(float(np.mean(product_p)), 3),
        "mrr": round(float(np.mean(rr)), 3),
        "ms_per_query": round(float(np.mean(latency)) * 1000, 1),
    }


def example(engine, df, query, k=3):
    hits = top_k(engine.scores(query), k=k)
    return [(df["product"].iloc[i], df["issue"].iloc[i], df["narrative"].iloc[i][:160]) for i in hits]


def chart(results):
    names = list(results)
    p10 = [results[n]["precision_at_10"] * 100 for n in names]
    mrr = [results[n]["mrr"] * 100 for n in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, p10, 0.4, label="Precision@10", color="#1f6f8b")
    ax.bar(x + 0.2, mrr, 0.4, label="MRR", color="#edae49")
    for i in range(len(names)):
        ax.text(x[i] - 0.2, p10[i] + 1, f"{p10[i]:.0f}", ha="center", fontsize=9)
        ax.text(x[i] + 0.2, mrr[i] + 1, f"{mrr[i]:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    ax.set_title("Which search method finds the right past complaints?", loc="left",
                 fontsize=12, fontweight="bold", pad=24)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_search_comparison.png", dpi=150)
    plt.close(fig)


def write_report(results, examples, n_docs, query):
    out = [
        "# Search Evaluation (Sprint 3)",
        "",
        f"Searched {n_docs:,} complaints with {N_QUERIES} test queries. Each query is the first "
        f"{QUERY_WORDS} words of a held-out complaint.",
        "",
        "**Relevance rule:** a result counts as relevant if it has the same product and issue as the "
        "complaint the query came from. There are no human relevance labels for this data, so this is a "
        "consistent but strict stand-in.",
        "",
        "| Method | Precision@10 | Same-product precision@10 | MRR | ms per query |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        out.append(f"| {name} | {r['precision_at_10']:.1%} | {r['product_precision_at_10']:.1%} | "
                   f"{r['mrr']:.3f} | {r['ms_per_query']} |")

    out += ["", f'## Example: top 3 results for "{query}"', ""]
    for name, hits in examples.items():
        out.append(f"**{name}**")
        out.append("")
        for product, issue, text in hits:
            out.append(f"- *{product} / {issue}*: {text}...")
        out.append("")
    (REPORTS_DIR / "search_evaluation.md").write_text("\n".join(out))


if __name__ == "__main__":
    df = pd.read_parquet(ENRICHED_FILE)
    docs = df["narrative"].tolist()
    picks, texts = make_queries(df)

    engines = {"TF-IDF": TfidfSearch().fit(docs), "BM25": BM25().fit(docs)}
    try:
        print("building embeddings (fast on a GPU, slow on CPU)...")
        engines["Embeddings"] = EmbeddingSearch().fit(docs)
    except Exception as e:
        # usually means sentence-transformers isn't installed or the model
        # couldn't download. the other two methods still get compared
        print(f"skipping embeddings ({type(e).__name__}: {e}). "
              "fix with: pip install sentence-transformers")

    results = {name: evaluate(e, df, picks, texts) for name, e in engines.items()}

    query = "someone used my card without permission and the bank refused to give my money back"
    examples = {name: example(e, df, query) for name, e in engines.items()}

    (REPORTS_DIR / "search_metrics.json").write_text(json.dumps(results, indent=2))
    chart(results)
    write_report(results, examples, len(df), query)

    print(pd.DataFrame(results).T.to_string())
