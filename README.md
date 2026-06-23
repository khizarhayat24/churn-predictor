# 🔵 Churn Intelligence — Customer Churn Prediction System

> **Teyzix Core Internship | June Batch | Task ML-2**  
> Predicting customer churn using Machine Learning — end-to-end from dataset creation to interactive dashboard.

---

## 📌 Project Overview

Customer churn prediction is one of the most commercially valuable applications of machine learning. This system was built from scratch — including a self-created dataset, a full preprocessing pipeline, three trained models, and an 8-page interactive Streamlit dashboard — within a 9-day deadline.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📁 Project Structure

```
churnpredictor/
├── app.py                      # Streamlit dashboard (8 pages, ~1500 lines)
├── engine.py                   # ChurnModelEngine — training, evaluation, segmentation
├── customer_churn_data.csv     # Self-created dataset (1,500 records)
├── best_model.pkl              # Serialized best model
├── preprocessor.pkl            # Serialized fitted preprocessing pipeline
├── metadata.json               # Run metadata — metrics, best params, features
├── config.toml                 # Streamlit theme config
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 📊 Dataset

Self-created synthetic dataset — **1,500 records, 13 columns**.

| Column | Type | Description |
|--------|------|-------------|
| Customer ID | Identifier | Unique customer reference |
| Age | Numeric | 18–70 years |
| Gender | Categorical | Male / Female |
| City | Categorical | 5 major US cities |
| Subscription Type | Categorical | Basic / Standard / Premium |
| Monthly Spending | Numeric | $10–$200 per month |
| Tenure | Numeric | Months as active customer |
| Number of Purchases | Numeric | Total transactions |
| Customer Support Requests | Numeric | Support tickets filed |
| Login Frequency | Numeric | Logins per month |
| Last Activity Date | Date | Last recorded session |
| Satisfaction Score | Numeric | Rating 1–10 |
| **Churn Status** | **Target** | **Yes / No** |

Churn labels were generated using a logistic function combining Satisfaction Score, Support Requests, Login Frequency, and Tenure — reflecting real-world churn dynamics.

---

## 🤖 Models Trained

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.653 | 0.512 | 0.536 | 0.524 | 0.660 |
| Random Forest | 0.671 | 0.528 | 0.500 | 0.514 | 0.710 |
| **XGBoost ★** | **0.658** | **0.519** | **0.527** | **0.523** | **0.720** |

★ XGBoost selected as best model by F1 Score + ROC-AUC.  
Further improved via **RandomizedSearchCV** hyperparameter optimization.

---

## 📱 Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | KPI strip, feature signals, model comparison table, AI chat assistant |
| **EDA** | 11 charts — demographics, distributions, correlation heatmap, city analysis |
| **Feature Signals** | Top churn drivers from the best model (XAI) |
| **Model Comparison** | Bar chart, evaluation table, confusion matrices for all 3 models |
| **Segmentation** | KMeans 4-cluster analysis, radar chart, segment churn profiles |
| **Risk Segments** | Test-set customers ranked by churn probability, filter + CSV export |
| **Predict Customer** | Single customer input → live churn probability + contributing factors |
| **Reports & Export** | Download model artifacts + markdown summary report |

---

## 🎯 Bonus Features Implemented

- ✅ **Explainable AI (XAI)** — Feature importance on Overview & Feature Signals pages
- ✅ **Hyperparameter Optimization** — RandomizedSearchCV on best model
- ✅ **Customer Segmentation** — KMeans clustering with radar chart
- ✅ **Churn Risk Dashboard** — Risk Segments page with probability filter
- ✅ **Multiple Model Comparison** — All metrics + confusion matrices

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Dashboard | Streamlit >= 1.40 |
| ML Models | scikit-learn, XGBoost |
| Visualization | Plotly |
| Data | Pandas, NumPy |
| Clustering | KMeans (sklearn) |
| Hyperparameter Opt | RandomizedSearchCV |
| Serialization | pickle, json |

---

## 📋 Evaluation Criteria

| Criteria | Weight | Status |
|----------|--------|--------|
| Dataset Quality & Realism | 15% | ✅ Done |
| Data Preparation & Feature Engineering | 20% | ✅ Done |
| Model Performance & Evaluation | 30% | ✅ Done |
| Problem-Solving Approach | 20% | ✅ Done |
| Documentation & Presentation | 15% | ✅ Done |

---

## 👤 Author

**Khizar Hayat**  
Teyzix Core Internship — June Batch 2026  
Task ID: ML-2 | Domain: Machine Learning

---

*Churn Intelligence Engine V1.0*
