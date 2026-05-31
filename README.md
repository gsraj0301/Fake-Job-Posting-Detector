# Fake Job Posting Detector

## Problem
Fraudulent job postings harm job seekers — especially students and freshers.
This project builds an ML classifier to detect fake postings using NLP and metadata.

## Dataset
- Source: [Kaggle — "Real or Fake Job Postings"](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) by Shivam Bansal
- 17,880 postings | 866 fake (4.84%) | 17,014 real
- Heavy class imbalance handled via `class_weight='balanced'`
- **Not included in repo** — download from Kaggle and place `fake_job_postings.csv` in the project root

## Approach
1. Combined text fields (title, company_profile, description, requirements, benefits) into one corpus
2. TF-IDF vectorization (5000 features, unigrams + bigrams)
3. Stacked with 3 binary metadata features (telecommuting, has_company_logo, has_questions)
4. Trained Logistic Regression and Random Forest classifiers
5. Evaluated on Recall and ROC-AUC (not raw accuracy — misleading on imbalanced data)

## Results
| Model               | Fake Recall | Fake Prec | Fake F1 | ROC-AUC |
|---------------------|-------------|-----------|---------|---------|
| Logistic Regression | **0.90**    | 0.54      | 0.68    | 0.989   |
| Random Forest       | 0.58        | **0.99**  | 0.73    | **0.991** |

**LR wins on the most important metric — Fake Recall (0.90).**
It catches 90% of fraudulent postings, while RF catches only 58%.
RF is more precise (99% of its "Fake" flags are correct) but misses too many.

For the use case — protecting job seekers — missing a fake posting is worse than flagging a real one, so LR is the deployed model.

## Key Insight
Top words predicting fake postings revealed patterns around:
- **Urgency:** "immediate", "urgent", "now"
- **Vague language:** "various", "general", "other"
- **Missing company details:** "has_company_logo" and "has_questions" are strong negative predictors (real companies have logos and ask questions)

See `top_words.png` for the full chart.

## Files
| File | Description |
|------|-------------|
| `fake_job_detector.py` | Full pipeline script |
| `fake_job_detector_lr.pkl` | Saved Logistic Regression model |
| `fake_job_detector_rf.pkl` | Saved Random Forest model |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC curve comparison |
| `top_words.png` | Top 20 predictive words |
| `class_distribution.png` | Real vs Fake distribution |

## Usage
```python
import joblib
from scipy.sparse import hstack
import scipy.sparse as sp

# Load model + vectorizer
lr = joblib.load("fake_job_detector_lr.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Prepare input
text = "urgent hiring work from home no experience needed earn thousands"
meta = sp.csr_matrix([[1, 0, 0]])  # telecommuting, logo, questions

# Predict
X_text = tfidf.transform([text])
X = hstack([X_text, meta])
prob = lr.predict_proba(X)[0, 1]
print(f"Fraud probability: {prob:.2%}")
```
