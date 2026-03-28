import argparse
import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
)

LABEL_NAMES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="pred.csv",
                        help="path to prediction CSV (columns: id, label)")
    args = parser.parse_args()

    ans  = pd.read_csv("data/valid.csv",  usecols=["id", "label"])
    pred = pd.read_csv(args.pred,         usecols=["id", "label"])

    df = pd.merge(ans, pred, how="left", on="id")
    missing = df["label_y"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} validation examples have no prediction – filled with -1",
              file=sys.stderr)
    df.fillna(-1, inplace=True)

    y_true = df["label_x"].astype(int)
    y_pred = df["label_y"].astype(int)

    acc        = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                  zero_division=0)

    print("=" * 45)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro-P   : {p:.4f}")
    print(f"  Macro-R   : {r:.4f}")
    print(f"  Macro-F1  : {macro_f1:.4f}  ← primary metric")
    print("=" * 45)
    print()
    print("Per-class F1:")
    print(classification_report(y_true, y_pred,
                                 target_names=LABEL_NAMES,
                                 zero_division=0))
