# Sprint 4 - route complaints automatically with a neural network built from scratch
#
# goal: read a complaint, predict which product team should get it.
# the network is plain numpy, forward pass, backprop and the Adam optimizer
# all written by hand, no pytorch. the point is to show I know what's going
# on inside one, not just how to call model.fit()
#
# a logistic regression on the same data is the baseline. if the network
# can't beat that, it isn't earning its complexity
#
#   python src/sprint4_neural_net.py

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from config import APP_DIR, ENRICHED_FILE, FIGURES_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR

SPLIT_FILE = PROCESSED_DIR / "split.csv"


def make_split(df):
    # 70 / 15 / 15, stratified so every product shows up in each piece.
    # saved to disk so the DistilBERT sprint tests on the exact same complaints
    train, rest = train_test_split(df, test_size=0.3, stratify=df["product"], random_state=RANDOM_STATE)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["product"], random_state=RANDOM_STATE)
    split = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    split[["complaint_id", "split"]].to_csv(SPLIT_FILE, index=False)
    return train, val, test


# TODO: try a second hidden layer and see if it beats one
class MLP:
    # input -> 128 relu units (with dropout) -> softmax over products

    def __init__(self, n_in, n_hidden, n_out, seed=RANDOM_STATE):
        rng = np.random.default_rng(seed)
        # He init for the relu layer, keeps the signal from dying or exploding early on
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, np.float32)
        self.W2 = rng.normal(0, np.sqrt(1 / n_hidden), (n_hidden, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, np.float32)
        self.rng = rng

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X, train=False, drop=0.3):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h = np.maximum(self.z1, 0)
        if train:
            # inverted dropout: zero out 30% of units, scale the rest up
            # so nothing needs to change at prediction time
            self.mask = (self.rng.random(self.h.shape) > drop) / (1 - drop)
            self.h = self.h * self.mask
        logits = self.h @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)  # stops exp() from overflowing
        e = np.exp(logits)
        self.p = e / e.sum(axis=1, keepdims=True)
        return self.p

    def backward(self, y, l2):
        n = len(y)
        # softmax + cross entropy has a nice gradient: predicted minus actual
        d_logits = self.p.copy()
        d_logits[np.arange(n), y] -= 1
        d_logits /= n

        dW2 = self.h.T @ d_logits + l2 * self.W2
        db2 = d_logits.sum(axis=0)
        dh = d_logits @ self.W2.T
        if hasattr(self, "mask"):
            dh = dh * self.mask
        dz1 = dh * (self.z1 > 0)  # relu only passes gradient where it was active
        dW1 = self.X.T @ dz1 + l2 * self.W1
        db1 = dz1.sum(axis=0)
        return [dW1, db1, dW2, db2]


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)  # bias correction, matters a lot in the first steps
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def cross_entropy(p, y):
    return float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean())


def train_mlp(Xtr, ytr, Xval, yval, n_classes, epochs=40, batch=256, l2=1e-4, patience=4):
    net = MLP(Xtr.shape[1], 128, n_classes)
    opt = Adam(net.params())
    rng = np.random.default_rng(RANDOM_STATE)
    history, best, best_params, bad = [], np.inf, None, 0

    for epoch in range(epochs):
        order = rng.permutation(len(ytr))
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            net.forward(Xtr[idx], train=True)
            opt.step(net.backward(ytr[idx], l2))

        if hasattr(net, "mask"):
            del net.mask  # dropout mask belongs to training only
        tr_loss = cross_entropy(net.forward(Xtr), ytr)
        val_loss = cross_entropy(net.forward(Xval), yval)
        history.append((epoch + 1, tr_loss, val_loss))
        print(f"  epoch {epoch + 1:2d}  train loss {tr_loss:.4f}  val loss {val_loss:.4f}")

        # early stopping: keep the weights from the best validation epoch,
        # quit once it hasn't improved for a few epochs in a row
        if val_loss < best - 1e-4:
            best, bad = val_loss, 0
            best_params = [p.copy() for p in net.params()]
        else:
            bad += 1
            if bad >= patience:
                break

    for p, saved in zip(net.params(), best_params):
        p[:] = saved
    return net, history


def chart_history(history):
    ep, tr, val = zip(*history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, tr, label="train", color="#1f6f8b", linewidth=2)
    ax.plot(ep, val, label="validation", color="#d1495b", linewidth=2)
    best = int(np.argmin(val))
    ax.scatter([ep[best]], [val[best]], color="#d1495b", zorder=3)
    # label sits up and to the left of the dot, so it stays on the chart
    # even when the best epoch is the very last one
    ax.annotate("weights kept from here", (ep[best], val[best]), xytext=(-140, 30),
                textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(frameon=False)
    ax.set_title("Neural network training (from scratch in NumPy)", loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_nn_training.png", dpi=150)
    plt.close(fig)


def chart_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "#333")
    ax.set_xlabel("Predicted product")
    ax.set_ylabel("Actual product")
    ax.set_title("Where the neural network gets confused", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_nn_confusion.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_parquet(ENRICHED_FILE)
    train, val, test = make_split(df)

    classes = sorted(df["product"].unique())
    to_id = {c: i for i, c in enumerate(classes)}
    ytr, yval, yte = (s["product"].map(to_id).to_numpy() for s in (train, val, test))

    # 10k words is plenty for routing and keeps the saved model small enough
    # to ship with the dashboard
    tfidf = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), ngram_range=(1, 2), min_df=3,
                            max_features=10_000, sublinear_tf=True)
    Ttr = tfidf.fit_transform(train["narrative"])
    Tval, Tte = tfidf.transform(val["narrative"]), tfidf.transform(test["narrative"])

    print("baseline: logistic regression on tf-idf")
    logreg = LogisticRegression(max_iter=2000, C=4.0).fit(Ttr, ytr)
    lr_pred = logreg.predict(Tte)

    # squeeze 10k sparse features into 200 dense ones (LSA) before the network.
    # a 10k-wide first layer would be slow in numpy and overfit fast
    k = min(200, Ttr.shape[1] - 1)
    svd = TruncatedSVD(n_components=k, random_state=RANDOM_STATE)
    Str = svd.fit_transform(Ttr)
    mu, sd = Str.mean(axis=0), Str.std(axis=0) + 1e-6
    prep = lambda T: ((svd.transform(T) - mu) / sd).astype(np.float32)
    Xtr, Xval, Xte = ((Str - mu) / sd).astype(np.float32), prep(Tval), prep(Tte)

    print("training the neural network")
    net, history = train_mlp(Xtr, ytr, Xval, yval, len(classes))
    nn_pred = net.forward(Xte).argmax(axis=1)

    metrics = {
        "test_complaints": int(len(yte)),
        "classes": classes,
        "logistic_regression": {"accuracy": round(accuracy_score(yte, lr_pred), 4),
                                "macro_f1": round(f1_score(yte, lr_pred, average="macro"), 4)},
        "numpy_neural_net": {"accuracy": round(accuracy_score(yte, nn_pred), 4),
                             "macro_f1": round(f1_score(yte, nn_pred, average="macro"), 4),
                             "epochs_trained": len(history)},
    }
    (REPORTS_DIR / "classifier_metrics.json").write_text(json.dumps(metrics, indent=2))

    report = classification_report(yte, nn_pred, target_names=classes, digits=3, zero_division=0)
    (REPORTS_DIR / "nn_classification_report.txt").write_text(report)

    chart_history(history)
    chart_confusion(yte, nn_pred, classes)

    # everything the dashboard needs to route a new complaint live.
    # float32 halves the file size and the predictions don't change
    joblib.dump({"tfidf": tfidf, "svd_components": svd.components_.astype(np.float32),
                 "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
                 "W1": net.W1, "b1": net.b1, "W2": net.W2, "b2": net.b2,
                 "classes": classes}, APP_DIR / "router.joblib", compress=3)

    print(json.dumps({k: v for k, v in metrics.items() if k != "classes"}, indent=2))
    print("\n" + report)
