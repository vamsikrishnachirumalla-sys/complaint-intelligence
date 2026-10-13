# Sprint 5 - fine-tune DistilBERT for the same routing task
#
# the numpy network in sprint 4 sees complaints as a bag of words. a transformer
# reads them in order, so it can tell "I never applied for this loan" from
# "I applied for this loan and never heard back". question is whether that
# extra understanding is worth the extra cost
#
# run this in colab with a GPU (Runtime > Change runtime type > T4 GPU).
# on a CPU it works but takes hours
#
#   python src/sprint5_deep_learning.py
#   python src/sprint5_deep_learning.py --max-train 5000   # quick test run

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import ENRICHED_FILE, FIGURES_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR

MODEL = "distilbert-base-uncased"
MAX_LEN = 256  # most complaints fit. 512 would roughly double training time  # most complaints fit, the long ones get cut. 512 doubles the time for little gain


def encode(tok, texts, labels, batch, shuffle):
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    data = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels))
    return DataLoader(data, batch_size=batch, shuffle=shuffle)


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for ids, mask, _ in loader:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def chart_all_models(results):
    names = list(results)
    acc = [results[n]["accuracy"] * 100 for n in names]
    f1 = [results[n]["macro_f1"] * 100 for n in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, acc, 0.4, label="Accuracy", color="#1f6f8b")
    ax.bar(x + 0.2, f1, 0.4, label="Macro F1", color="#edae49")
    for i in range(len(names)):
        ax.text(x[i] - 0.2, acc[i] + 1, f"{acc[i]:.1f}", ha="center", fontsize=9)
        ax.text(x[i] + 0.2, f1[i] + 1, f"{f1[i]:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Test score (%)")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    ax.set_title("Routing complaints: three models, same test set", loc="left",
                 fontsize=12, fontweight="bold", pad=24)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_model_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-train", type=int, default=20_000,
                        help="cap on training complaints, keeps a T4 run around 15-20 minutes")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}" + ("" if device.type == "cuda" else " (this will be slow, switch colab to a GPU)"))

    # same train/val/test complaints as sprint 4, so the comparison is fair
    # ids come out of the archive as text but read_csv guesses numbers,
    # so force both sides to text before matching them up
    df = pd.read_parquet(ENRICHED_FILE)
    df["complaint_id"] = df["complaint_id"].astype(str)
    split = pd.read_csv(PROCESSED_DIR / "split.csv", dtype={"complaint_id": str})
    df = df.merge(split, on="complaint_id")
    classes = sorted(df["product"].unique())
    df["label"] = df["product"].map({c: i for i, c in enumerate(classes)})

    train = df[df["split"] == "train"]
    if len(train) > args.max_train:
        train = train.sample(args.max_train, random_state=RANDOM_STATE)
    test = df[df["split"] == "test"]

    tok = AutoTokenizer.from_pretrained(MODEL)
    train_dl = encode(tok, train["narrative"], train["label"].to_numpy(), args.batch, shuffle=True)
    test_dl = encode(tok, test["narrative"], test["label"].to_numpy(), args.batch * 2, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_dl) * args.epochs
    # warm up for the first 10% of steps then decay to zero. jumping straight to
    # full learning rate tends to wreck the pretrained weights early on
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")  # mixed precision, ~2x faster on a T4

    start, losses = time.time(), []
    for epoch in range(args.epochs):
        model.train()
        for step, (ids, mask, y) in enumerate(train_dl):
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
            scaler.scale(out.loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # keeps one bad batch from blowing things up
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(out.loss.item())
            if step % 100 == 0:
                print(f"  epoch {epoch + 1}  step {step}/{len(train_dl)}  loss {np.mean(losses[-100:]):.4f}")

    minutes = (time.time() - start) / 60
    y_true = test["label"].to_numpy()
    y_pred = predict(model, test_dl, device)

    result = {"accuracy": round(accuracy_score(y_true, y_pred), 4),
              "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
              "train_complaints": int(len(train)), "epochs": args.epochs,
              "train_minutes": round(minutes, 1), "device": device.type}
    (REPORTS_DIR / "distilbert_metrics.json").write_text(json.dumps(result, indent=2))
    (REPORTS_DIR / "distilbert_classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=classes, digits=3, zero_division=0))

    # line all three models up on the same chart
    earlier = json.loads((REPORTS_DIR / "classifier_metrics.json").read_text())
    chart_all_models({
        "Logistic\nregression": earlier["logistic_regression"],
        "Neural net\n(NumPy)": earlier["numpy_neural_net"],
        "DistilBERT": result,
    })

    print(json.dumps(result, indent=2))
