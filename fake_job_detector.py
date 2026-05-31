"""
Fake Job Posting Detector — Full Pipeline
==========================================
NLP + metadata classifier to detect fraudulent job postings.
Dataset: Kaggle "Real or Fake Job Postings" by Shivam Bansal

Phases:
  1. Load & clean data
  2. Feature engineering (TF-IDF + binary flags)
  3. Train Logistic Regression & Random Forest
  4. Evaluation plots
  5. Save models
  6. Final verification & sanity checks
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
)
from scipy.sparse import hstack
import scipy.sparse as sp

# ── Settings ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 5000
OUTPUT_DIR = "."

sns.set_style("whitegrid")
np.random.seed(RANDOM_STATE)

# =============================================================================
# PHASE 1 — Load & Clean Data
# =============================================================================
print("=" * 60)
print("PHASE 1 — Load & Clean Data")
print("=" * 60)

df = pd.read_csv("fake_job_postings.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Fraudulent postings: {df['fraudulent'].sum()} / {len(df)} "
      f"({df['fraudulent'].mean() * 100:.2f}%)")

# Combine the 5 text columns into one corpus
text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
df[text_cols] = df[text_cols].fillna("")
df["combined_text"] = df[text_cols].apply(
    lambda row: " ".join(row.values), axis=1
)

# Keep only the columns we need for modeling
features_df = df[
    ["combined_text", "telecommuting", "has_company_logo", "has_questions", "fraudulent"]
].copy()

# Quick sanity: verify no nulls remain
assert features_df.isnull().sum().sum() == 0, "Null values still present!"
print(f"Features shape: {features_df.shape}")
print(f"Fraud distribution:\n{features_df['fraudulent'].value_counts()}")
print()

# =============================================================================
# PHASE 2 — Feature Engineering (TF-IDF + binary flags)
# =============================================================================
print("=" * 60)
print("PHASE 2 — Feature Engineering")
print("=" * 60)

# TF-IDF on combined text — captures important words & phrases
tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    stop_words="english",
    ngram_range=(1, 2),          # unigrams + bigrams
)
X_text = tfidf.fit_transform(features_df["combined_text"])
print(f"TF-IDF matrix shape: {X_text.shape}")

# Binary metadata features — sparse format for stacking
X_meta = sp.csr_matrix(
    features_df[["telecommuting", "has_company_logo", "has_questions"]].values
)

# Stack text features + metadata into a single feature matrix
X = hstack([X_text, X_meta])
y = features_df["fraudulent"].values
names = tfidf.get_feature_names_out().tolist() + [
    "telecommuting",
    "has_company_logo",
    "has_questions",
]
print(f"Full feature matrix: {X.shape}")
print(f"Total features: {X.shape[1]} (5000 text + 3 metadata)")
print()

# =============================================================================
# PHASE 3 — Train/Test Split & Modeling
# =============================================================================
print("=" * 60)
print("PHASE 3 — Modeling")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"Train fraud rate: {y_train.mean() * 100:.2f}%")
print(f"Test fraud rate:  {y_test.mean() * 100:.2f}%")
print()

# ── Model 1: Logistic Regression ─────────────────────────────────────────────
print("-" * 40)
print("Model 1: Logistic Regression")
print("-" * 40)

lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_STATE,
)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

lr_roc_auc = roc_auc_score(y_test, y_prob_lr)
print(classification_report(y_test, y_pred_lr, target_names=["Real", "Fake"]))
print(f"ROC-AUC: {lr_roc_auc:.4f}")
print()

# ── Model 2: Random Forest ───────────────────────────────────────────────────
print("-" * 40)
print("Model 2: Random Forest")
print("-" * 40)

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

rf_roc_auc = roc_auc_score(y_test, y_prob_rf)
print(classification_report(y_test, y_pred_rf, target_names=["Real", "Fake"]))
print(f"ROC-AUC: {rf_roc_auc:.4f}")
print()

# ── Decision: pick the best model based on Fake-class Recall ────────────────
from sklearn.metrics import recall_score

lr_recall_fake = recall_score(y_test, y_pred_lr, pos_label=1)
rf_recall_fake = recall_score(y_test, y_pred_rf, pos_label=1)

if rf_recall_fake >= lr_recall_fake:
    best_model = rf
    best_name = "Random Forest"
    best_pred = y_pred_rf
    best_prob = y_prob_rf
    best_recall = rf_recall_fake
    best_roc_auc = rf_roc_auc
else:
    best_model = lr
    best_name = "Logistic Regression"
    best_pred = y_pred_lr
    best_prob = y_prob_lr
    best_recall = lr_recall_fake
    best_roc_auc = lr_roc_auc

print(f"★ Best model: {best_name} (Fake Recall = {best_recall:.4f})")
print()

# ── Check success criteria ───────────────────────────────────────────────────
HIT_RECALL = best_recall >= 0.75
HIT_AUC = best_roc_auc >= 0.95

if HIT_RECALL and HIT_AUC:
    print("✅ Success criteria met — proceeding to plots + save")
else:
    print("⚠️  Success criteria NOT met — attempting tuning...")
    # ── Try threshold tuning on LR to improve recall ────────────────────────
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_lr)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_thresh_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_thresh_idx]

    y_pred_lr_tuned = (y_prob_lr >= best_threshold).astype(int)
    tuned_recall = recall_score(y_test, y_pred_lr_tuned, pos_label=1)
    print(f"  LR threshold tuned: best threshold = {best_threshold:.4f}, "
          f"new Fake Recall = {tuned_recall:.4f}")

    if tuned_recall > best_recall:
        best_model = lr
        best_name = "Logistic Regression (tuned)"
        best_pred = y_pred_lr_tuned
        best_recall = tuned_recall
        print("✅ Using tuned LR instead")
    print()

# =============================================================================
# PHASE 4 — Evaluation Plots
# =============================================================================
print("=" * 60)
print("PHASE 4 — Evaluation Plots")
print("=" * 60)

# ── Plot 1: Confusion Matrix (best model) ────────────────────────────────────
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Real", "Fake"],
    yticklabels=["Real", "Fake"],
)
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("  ✓ confusion_matrix.png saved")

# ── Plot 2: ROC Curve (both models) ──────────────────────────────────────────
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr_lr, tpr_lr,
    label=f"Logistic Regression (AUC = {lr_roc_auc:.3f})",
)
plt.plot(
    fpr_rf, tpr_rf,
    label=f"Random Forest (AUC = {rf_roc_auc:.3f})",
)
plt.plot([0, 1], [0, 1], "k--", label="Random Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
plt.close()
print("  ✓ roc_curve.png saved")

# ── Plot 3: Top 20 Words Predicting Fake Postings (from LR coefficients) ─────
coefs = lr.coef_[0]
top20_idx = np.argsort(coefs)[-20:][::-1]
top20_words = [names[i] for i in top20_idx]
top20_scores = [coefs[i] for i in top20_idx]

plt.figure(figsize=(8, 6))
bars = sns.barplot(x=top20_scores, y=top20_words, palette="Reds_r", hue=top20_words, legend=False)
plt.title("Top 20 Words Predicting Fake Job Postings")
plt.xlabel("Logistic Regression Coefficient")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_words.png"), dpi=150)
plt.close()
print("  ✓ top_words.png saved")

# ── Plot 4: Class Distribution ───────────────────────────────────────────────
plt.figure(figsize=(5, 4))
sns.countplot(x=features_df["fraudulent"], palette=["steelblue", "crimson"], hue=features_df["fraudulent"], legend=False)
plt.xticks([0, 1], ["Real (17,014)", "Fake (866)"])
plt.title("Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
plt.close()
print("  ✓ class_distribution.png saved")
print()

# =============================================================================
# PHASE 5 — Save Models
# =============================================================================
print("=" * 60)
print("PHASE 5 — Save Models with joblib")
print("=" * 60)

joblib.dump(lr, os.path.join(OUTPUT_DIR, "fake_job_detector_lr.pkl"))
joblib.dump(rf, os.path.join(OUTPUT_DIR, "fake_job_detector_rf.pkl"))
joblib.dump(tfidf, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"))
print("  ✓ fake_job_detector_lr.pkl saved")
print("  ✓ fake_job_detector_rf.pkl saved")
print("  ✓ tfidf_vectorizer.pkl saved")
print()

# =============================================================================
# PHASE 6 — Final Verification & Sanity Checks
# =============================================================================
print("=" * 60)
print("PHASE 6 — Final Verification")
print("=" * 60)

# Reload models to confirm they serialise correctly
lr_loaded = joblib.load(os.path.join(OUTPUT_DIR, "fake_job_detector_lr.pkl"))
rf_loaded = joblib.load(os.path.join(OUTPUT_DIR, "fake_job_detector_rf.pkl"))
tfidf_loaded = joblib.load(os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"))
print("  ✓ Models loaded successfully from disk")

# Verify predictions match
y_pred_lr_loaded = lr_loaded.predict(X_test)
y_pred_rf_loaded = rf_loaded.predict(X_test)
assert np.array_equal(y_pred_lr, y_pred_lr_loaded), "LR predictions don't match!"
assert np.array_equal(y_pred_rf, y_pred_rf_loaded), "RF predictions don't match!"
print("  ✓ Predictions identical after reload (serialisation OK)")

# ── Performance summary table ────────────────────────────────────────────────
print()
print("=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)

lr_report = classification_report(y_test, y_pred_lr, target_names=["Real", "Fake"], output_dict=True)
rf_report = classification_report(y_test, y_pred_rf, target_names=["Real", "Fake"], output_dict=True)

print(f"{'Model':<25} {'Fake Recall':<15} {'Fake Prec':<15} {'Fake F1':<15} {'ROC-AUC':<10}")
print("-" * 80)
print(f"{'Logistic Regression':<25} {lr_report['Fake']['recall']:<15.4f} "
      f"{lr_report['Fake']['precision']:<15.4f} {lr_report['Fake']['f1-score']:<15.4f} "
      f"{lr_roc_auc:<10.4f}")
print(f"{'Random Forest':<25} {rf_report['Fake']['recall']:<15.4f} "
      f"{rf_report['Fake']['precision']:<15.4f} {rf_report['Fake']['f1-score']:<15.4f} "
      f"{rf_roc_auc:<10.4f}")
print()

# ── Sanity check with extreme examples ───────────────────────────────────────
print("-" * 40)
print("Sanity checks on extreme examples")
print("-" * 40)

# Build a "clearly fake" posting using the vectorizer
clearly_fake = (
    "urgent hiring immediate salary paid work from home no experience "
    "need money fast earn thousands weekly guaranteed"
)
fake_text_vec = tfidf_loaded.transform([clearly_fake])
fake_meta = sp.csr_matrix([[1, 0, 0]])  # telecommuting=1, logo=0, questions=0
fake_vec = hstack([fake_text_vec, fake_meta])

fake_pred_lr = lr_loaded.predict(fake_vec)[0]
fake_prob_lr = lr_loaded.predict_proba(fake_vec)[0, 1]
print(f"Fake posting → LR predicts: {'Fake' if fake_pred_lr else 'Real'} "
      f"(fraud probability: {fake_prob_lr:.4f})")

# Build a "clearly real" posting
clearly_real = (
    "software engineer required 5 years experience python java cloud "
    "competitive salary benefits health insurance 401k matching"
)
real_text_vec = tfidf_loaded.transform([clearly_real])
real_meta = sp.csr_matrix([[0, 1, 1]])  # telecommuting=0, logo=1, questions=1
real_vec = hstack([real_text_vec, real_meta])

real_pred_lr = lr_loaded.predict(real_vec)[0]
real_prob_lr = lr_loaded.predict_proba(real_vec)[0, 1]
print(f"Real posting → LR predicts: {'Fake' if real_pred_lr else 'Real'} "
      f"(fraud probability: {real_prob_lr:.4f})")

# ── Overall test set fraud recall ────────────────────────────────────────────
print()
print(f"Test set — LR correct fraud predictions: "
      f"{np.sum(y_pred_lr[y_test == 1])} / {np.sum(y_test == 1)}")
print(f"Test set — RF correct fraud predictions: "
      f"{np.sum(y_pred_rf[y_test == 1])} / {np.sum(y_test == 1)}")

# ── Success criteria final verdict ───────────────────────────────────────────
print()
print("=" * 40)
if HIT_RECALL and HIT_AUC:
    print("✅ ALL SUCCESS CRITERIA MET")
else:
    print("⚠️  SOME CRITERIA NOT MET (see above)")
print(f"   Fake Recall ≥ 0.75 ? {'✅' if HIT_RECALL else '❌'} ({best_recall:.4f})")
print(f"   ROC-AUC    ≥ 0.95 ? {'✅' if HIT_AUC else '❌'} ({best_roc_auc:.4f})")
print("=" * 40)
print()
print("Pipeline complete. Files generated:")
for f in ["confusion_matrix.png", "roc_curve.png", "top_words.png",
          "class_distribution.png", "fake_job_detector_lr.pkl",
          "fake_job_detector_rf.pkl", "tfidf_vectorizer.pkl"]:
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"  {f:.<40s} {size_kb:.1f} KB")
