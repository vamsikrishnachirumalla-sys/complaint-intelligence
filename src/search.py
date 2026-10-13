# the search engines, shared by sprint 3 (evaluation) and the dashboard (live search)
#
# wrote BM25 by hand on top of a sparse matrix instead of using rank_bm25.
# the library loops over documents in python, this way one query is a single
# matrix multiply, which is what makes live search in the dashboard feel instant

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

TOKENS = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"


class BM25:
    # k1 controls how fast repeated words stop adding score,
    # b controls how much long complaints get penalized. 1.5 / 0.75 are the textbook defaults
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, docs):
        self.vec = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS), token_pattern=TOKENS, min_df=2)
        tf = self.vec.fit_transform(docs).tocsr().astype(np.float32)

        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))

        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * doc_len / doc_len.mean())

        # turn every raw count into its final BM25 weight up front, so scoring a
        # query later is just "add up the weights of the query's words"
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        tf.data = idf[tf.indices] * tf.data * (self.k1 + 1) / (tf.data + norm[rows])
        self.weights = tf.T.tocsr()  # words x docs
        return self

    def scores(self, query):
        q = self.vec.transform([query])
        q.data[:] = 1  # a word typed twice in the query shouldn't count double
        return np.asarray((q @ self.weights).todense()).ravel()


class TfidfSearch:
    def fit(self, docs):
        self.vec = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), token_pattern=TOKENS,
                                   min_df=2, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(docs)
        return self

    def scores(self, query):
        return np.asarray((self.matrix @ self.vec.transform([query]).T).todense()).ravel()


class EmbeddingSearch:
    # meaning-based search: "card was stolen" can match "someone used my card"
    # even with zero shared words. needs sentence-transformers (pip install it in colab)
    def __init__(self, model="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def fit(self, docs):
        emb = self.model.encode(list(docs), batch_size=128, show_progress_bar=True,
                                convert_to_numpy=True)
        self.matrix = normalize(emb)
        return self

    def scores(self, query):
        q = normalize(self.model.encode([query], convert_to_numpy=True))
        return self.matrix @ q.ravel()


def top_k(scores, k=10, skip=None):
    if skip is not None:
        scores = scores.copy()
        scores[skip] = -np.inf  # don't let a query find itself
    idx = np.argpartition(-scores, k)[:k]
    return idx[np.argsort(-scores[idx])]
