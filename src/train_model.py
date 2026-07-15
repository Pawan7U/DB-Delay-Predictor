"""
train_model.py
--------------
Trains two models on the DB delay dataset:

1. Classifier -> predicts whether a train will be delayed (> 5 min)
2. Regressor  -> predicts the actual delay in minutes

Both use a RandomForest inside a preprocessing pipeline (one-hot encoding
for categoricals). Saves trained models to models/ and evaluation plots
to outputs/figures/.

Usage:
    python src/train_model.py --data data/db_delays.csv
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, r2_score,
)

FEATURES_CAT = ["train_type", "origin", "destination", "weather", "weekday"]
FEATURES_BOOL_NUM = ["month", "hour", "construction_work", "is_rush_hour", "is_weekend"]
ALL_FEATURES = FEATURES_CAT + FEATURES_BOOL_NUM


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
        ],
        remainder="passthrough",
    )


def train_classifier(X_train, X_test, y_train, y_test, out_dir):
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=3,
            random_state=42, n_jobs=-1, class_weight="balanced",
        )),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print("\n=== Classifier: 'Will this train be delayed?' ===")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["On time", "Delayed"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Delay Classifier - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "classifier_confusion_matrix.png"), dpi=150)
    plt.close()

    return pipe, metrics


def train_regressor(X_train, X_test, y_train, y_test, out_dir):
    # Delay minutes are heavily right-skewed (many small delays, a long tail
    # of big ones). Training on log1p(delay) and inverting with expm1 at
    # prediction time gives the tree model a much easier target to learn.
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("reg", RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )),
    ])
    y_train_log = np.log1p(y_train)
    pipe.fit(X_train, y_train_log)

    y_pred = np.expm1(pipe.predict(X_test))
    y_pred = np.clip(y_pred, 0, None)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Regressor: 'How many minutes delayed?' ===")
    print(f"MAE : {mae:.2f} minutes")
    print(f"R^2 : {r2:.3f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.25, s=12)
    lims = [0, max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual delay (min)")
    plt.ylabel("Predicted delay (min)")
    plt.title("Regressor: Actual vs Predicted Delay")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "regressor_actual_vs_predicted.png"), dpi=150)
    plt.close()

    return pipe, {"mae": mae, "r2": r2}


def plot_feature_importance(pipe, model_step, out_path, title):
    prep = pipe.named_steps["prep"]
    cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(FEATURES_CAT))
    feature_names = cat_names + FEATURES_BOOL_NUM
    importances = pipe.named_steps[model_step].feature_importances_

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="darkorange")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train delay prediction models")
    parser.add_argument("--data", type=str, default="data/db_delays.csv")
    parser.add_argument("--models_out", type=str, default="models")
    parser.add_argument("--figures_out", type=str, default="outputs/figures")
    args = parser.parse_args()

    os.makedirs(args.models_out, exist_ok=True)
    os.makedirs(args.figures_out, exist_ok=True)

    df = pd.read_csv(args.data)
    X = df[ALL_FEATURES].copy()
    X["construction_work"] = X["construction_work"].astype(int)
    X["is_rush_hour"] = X["is_rush_hour"].astype(int)
    X["is_weekend"] = X["is_weekend"].astype(int)

    y_clf = df["is_delayed"].astype(int)
    y_reg = df["delay_minutes"]

    X_train, X_test, ytr_clf, yte_clf, ytr_reg, yte_reg = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )

    clf_pipe, clf_metrics = train_classifier(X_train, X_test, ytr_clf, yte_clf, args.figures_out)
    reg_pipe, reg_metrics = train_regressor(X_train, X_test, ytr_reg, yte_reg, args.figures_out)

    plot_feature_importance(
        clf_pipe, "clf",
        os.path.join(args.figures_out, "classifier_feature_importance.png"),
        "Top Features - Delay Classifier",
    )
    plot_feature_importance(
        reg_pipe, "reg",
        os.path.join(args.figures_out, "regressor_feature_importance.png"),
        "Top Features - Delay Regressor",
    )

    joblib.dump(clf_pipe, os.path.join(args.models_out, "delay_classifier.joblib"))
    joblib.dump(reg_pipe, os.path.join(args.models_out, "delay_regressor.joblib"))

    print(f"\nSaved models to {args.models_out}/")
    print(f"Saved figures to {args.figures_out}/")


if __name__ == "__main__":
    main()
