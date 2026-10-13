"""
Train a spring GOOD/BAD classifier on geometric + keypoint features.

Input  : D:/spring-defects/runs/features/features.csv
Output : D:/spring-defects/runs/features/classifier.pkl
         D:/spring-defects/runs/features/classifier_report.txt

Usage:
    D:\\spring-defects\\.venv\\Scripts\\python.exe D:\\spring-defects\\scripts\\train_classifier.py
"""

import sys
import pathlib
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

ROOT       = pathlib.Path("D:/spring-defects")
CSV_PATH   = ROOT / "runs/features/features.csv"
OUTPUT_DIR = ROOT / "runs/features"
PKL_PATH   = OUTPUT_DIR / "classifier.pkl"
RPT_PATH   = OUTPUT_DIR / "classifier_report.txt"

# ── Feature groups ────────────────────────────────────────────────────────────
GEO_COLS = [
    "total_length_ratio",        # ← was total_length_px
    "tilt_angle_deg",
    "shaft_deviation_ratio",     # ← keep ratio, drop shaft_deviation_px
    "top_hook_offset_ratio",     # ← keep ratio, drop top_hook_offset_px
    "top_hook_angle_deg",
    "bottom_hook_offset_ratio",  # ← keep ratio, drop bottom_hook_offset_px
    "bottom_hook_angle_deg",
    "top_coil_gap_ratio",
    "bottom_coil_gap_ratio",
]
KP_COLS = [f"kp{i}_{c}" for i in range(7) for c in ("x", "y")]
ALL_COLS = GEO_COLS + KP_COLS   # 26 total

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Rows: {len(df)}  |  GOOD: {(df.label==0).sum()}  |  BAD: {(df.label==1).sum()}\n")

X = df[ALL_COLS].values
y = df["label"].values

# ── Classifiers to try ────────────────────────────────────────────────────────
candidates = {
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                       class_weight="balanced", random_state=42)),
    ]),
    "SVM (Linear)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="linear", C=1,
                       class_weight="balanced", random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
                       n_estimators=200, max_depth=None,
                       class_weight="balanced", random_state=42, n_jobs=-1)),
    ]),
}

# ── 5-fold stratified cross-validation ───────────────────────────────────────
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall"]

results = {}
lines   = []

header = f"\n{'=' * 60}\n  CROSS-VALIDATION RESULTS (5-fold)\n{'=' * 60}"
print(header)
lines.append(header)

for name, pipe in candidates.items():
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    results[name] = {
        "accuracy" : scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall"   : scores["test_recall"].mean(),
        "pipeline" : pipe,
    }
    block = (
        f"\n  {name}\n"
        f"    Accuracy  : {scores['test_accuracy'].mean():.3f} "
        f"(±{scores['test_accuracy'].std():.3f})\n"
        f"    Precision : {scores['test_precision'].mean():.3f} "
        f"(±{scores['test_precision'].std():.3f})\n"
        f"    Recall    : {scores['test_recall'].mean():.3f} "
        f"(±{scores['test_recall'].std():.3f})"
    )
    print(block)
    lines.append(block)

# ── Pick best model by recall (missing a bad spring is the costly error) ──────
best_name = max(results, key=lambda k: results[k]["recall"])
best_pipe = results[best_name]["pipeline"]

chosen = f"\n{'=' * 60}\n  BEST MODEL (highest recall): {best_name}\n{'=' * 60}"
print(chosen)
lines.append(chosen)

# ── Retrain best model on full dataset ───────────────────────────────────────
best_pipe.fit(X, y)
y_pred = best_pipe.predict(X)

cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

full_report = (
    f"\n  Trained on full 500 images:\n"
    f"    TP (BAD  correctly detected) : {tp}\n"
    f"    FP (GOOD wrongly flagged)    : {fp}\n"
    f"    TN (GOOD correctly passed)   : {tn}\n"
    f"    FN (BAD  missed)             : {fn}\n"
    f"    Accuracy  : {accuracy_score(y, y_pred)*100:.1f}%\n"
    f"    Precision : {precision_score(y, y_pred)*100:.1f}%\n"
    f"    Recall    : {recall_score(y, y_pred)*100:.1f}%\n"
    f"\n{classification_report(y, y_pred, target_names=['GOOD','BAD'])}"
)
print(full_report)
lines.append(full_report)

# ── Feature importance (Random Forest only) ───────────────────────────────────
if "Random Forest" in best_name:
    rf   = best_pipe.named_steps["clf"]
    imps = rf.feature_importances_
    top5 = np.argsort(imps)[::-1][:5]
    fi_block = "\n  Top 5 feature importances:\n"
    for rank, idx in enumerate(top5, 1):
        fi_block += f"    {rank}. {ALL_COLS[idx]:<30} {imps[idx]:.4f}\n"
    print(fi_block)
    lines.append(fi_block)

# ── Save model ────────────────────────────────────────────────────────────────
with open(PKL_PATH, "wb") as f:
    pickle.dump({
        "pipeline"   : best_pipe,
        "model_name" : best_name,
        "feature_cols": ALL_COLS,
    }, f)

# ── Save report ───────────────────────────────────────────────────────────────
with open(RPT_PATH, "w") as f:
    f.write("\n".join(lines))

saved = f"\n  Model saved : {PKL_PATH}\n  Report saved: {RPT_PATH}"
print(saved)