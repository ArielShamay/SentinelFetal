import csv
from collections import defaultdict
from statistics import mean
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def load(path: Path):
    data = defaultdict(lambda: {"label": None, "xgb": [], "rf": [], "sgd": []})
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"]
            lbl = int(row["true_label"])
            rec = data[pid]
            if rec["label"] is None:
                rec["label"] = lbl
            rec["xgb"].append(float(row["xgb_prob"]))
            rec["rf"].append(float(row["rf_prob"]))
            rec["sgd"].append(float(row["sgd_prob"]))
    labels = []
    models = {"XGBoost": [], "Random Forest": [], "SGD": []}
    for pid, rec in data.items():
        labels.append(rec["label"])
        models["XGBoost"].append(max(rec["xgb"]))
        models["Random Forest"].append(max(rec["rf"]))
        models["SGD"].append(max(rec["sgd"]))
    return labels, models


def summarize(tag: str, path: Path):
    labels, models = load(path)
    print(f"=== {tag} ===")
    print("| Model | Recall | Precision | ROC-AUC | MeanNeg | MeanPos |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, probs in models.items():
        preds = [1 if p >= 0.5 else 0 for p in probs]
        recall = recall_score(labels, preds, pos_label=1, zero_division=0)
        prec = precision_score(labels, preds, pos_label=1, zero_division=0)
        auc = roc_auc_score(labels, probs)
        neg_probs = [p for p, lbl in zip(probs, labels) if lbl == 0]
        pos_probs = [p for p, lbl in zip(probs, labels) if lbl == 1]
        mean_neg = mean(neg_probs)
        mean_pos = mean(pos_probs)
        print(f"| {name} | {recall:.3f} | {prec:.3f} | {auc:.3f} | {mean_neg:.3f} | {mean_pos:.3f} |")


def main():
    summarize("V4", Path("models/ensemble_v4/validation_preds.csv"))
    summarize("V5", Path("models/ensemble_v5/validation_preds_v5.csv"))


if __name__ == "__main__":
    main()
