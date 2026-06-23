import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class ChurnModelEngine:
    """
    Core churn-modeling engine.

    1. Load raw customer data.
    2. Clean structural anomalies (duplicates, identifier/date columns).
    3. Build a reusable preprocessing pipeline (impute + scale numeric,
       impute + one-hot encode categorical).
    4. Perform a stratified train/test split.
    5. Train Logistic Regression, Random Forest and XGBoost.
    6. Hyperparameter optimization via RandomizedSearchCV on best model.
    7. Evaluate every model on Accuracy / Precision / Recall / F1 / ROC-AUC / Confusion Matrix.
    8. Select the best model by F1 score (robust to churn class imbalance).
    9. Customer segmentation via KMeans clustering.
    10. Export the best model, the fitted preprocessor and run metadata.
    """

    def __init__(self):
        self.numerical_cols   = ['Age', 'Monthly Spending', 'Tenure', 'Number of Purchases',
                                  'Customer Support Requests', 'Login Frequency', 'Satisfaction Score']
        self.categorical_cols = ['Gender', 'City', 'Subscription Type']
        self.target_col       = 'Churn Status'
        self.preprocessor     = None
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'XGBoost':             XGBClassifier(eval_metric='logloss', random_state=42),
        }
        self.best_model             = None
        self.best_model_name        = ""
        self.best_params            = {}          # ← hyperparameter optimization result
        self.metrics_report         = {}
        self.engineered_features    = []

        # Segmentation
        self.segment_model          = None
        self.segment_df             = None        # full dataset with cluster labels
        self.n_segments             = 4

        # Dashboard integration state
        self.n_records              = 0
        self.overall_churn_rate     = 0.0
        self.y_test                 = None
        self.test_results           = None
        self._best_preds            = None
        self._best_probs            = None

    # -------------------------------------------------------------------------
    def build_preprocessor(self):
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler())
        ])
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, self.numerical_cols),
            ('cat', cat_transformer, self.categorical_cols)
        ])

    # -------------------------------------------------------------------------
    def _hyperparameter_optimize(self, model_name, model, X_train_proc, y_train):
        """
        Runs RandomizedSearchCV on the best-performing model to squeeze out
        better performance. Returns the re-fitted optimized estimator.
        """
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth':    [6, 8, 10, 12, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf':  [1, 2, 4],
            },
            'XGBoost': {
                'n_estimators':  [100, 200, 300],
                'max_depth':     [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample':     [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0],
            },
            'Logistic Regression': {
                'C':      [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'saga'],
                'penalty':['l2'],
            },
        }
        grid = param_grids.get(model_name)
        if grid is None:
            return model, {}

        search = RandomizedSearchCV(
            model, grid,
            n_iter=5, cv=2, scoring='f1',
            random_state=42, n_jobs=-1, verbose=0
        )
        search.fit(X_train_proc, y_train)
        return search.best_estimator_, search.best_params_

    # -------------------------------------------------------------------------
    def _run_segmentation(self, X_raw, original_df):
        """
        KMeans segmentation on scaled numeric features.
        Attaches cluster labels + churn rate per cluster to self.segment_df.
        """
        num_data = X_raw[self.numerical_cols].copy()
        # Fill any residual nulls with median before clustering
        for col in self.numerical_cols:
            num_data[col].fillna(num_data[col].median(), inplace=True)

        scaler    = StandardScaler()
        scaled    = scaler.fit_transform(num_data)
        self.segment_model = KMeans(n_clusters=self.n_segments, random_state=42, n_init=10)
        labels    = self.segment_model.fit_predict(scaled)

        seg_df = original_df[self.numerical_cols + [self.target_col]].copy()
        seg_df['Segment'] = labels

        # Per-segment summary
        summary = seg_df.groupby('Segment').agg(
            Count          = (self.target_col, 'count'),
            Churn_Rate_Pct = (self.target_col, lambda x: round(x.mean() * 100, 1)),
            Avg_Age        = ('Age',             'mean'),
            Avg_Spending   = ('Monthly Spending', 'mean'),
            Avg_Tenure     = ('Tenure',           'mean'),
            Avg_Satisfaction = ('Satisfaction Score', 'mean'),
            Avg_Support_Req  = ('Customer Support Requests', 'mean'),
            Avg_Login_Freq   = ('Login Frequency', 'mean'),
        ).reset_index()
        summary['Avg_Age']          = summary['Avg_Age'].round(1)
        summary['Avg_Spending']     = summary['Avg_Spending'].round(1)
        summary['Avg_Tenure']       = summary['Avg_Tenure'].round(1)
        summary['Avg_Satisfaction'] = summary['Avg_Satisfaction'].round(1)
        summary['Avg_Support_Req']  = summary['Avg_Support_Req'].round(1)
        summary['Avg_Login_Freq']   = summary['Avg_Login_Freq'].round(1)
        self.segment_df             = summary

    # -------------------------------------------------------------------------
    def train_and_evaluate(self, data):
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.read_csv(data)

        df.drop_duplicates(inplace=True)

        if 'Customer ID' in df.columns:
            customer_ids = df['Customer ID']
            df.drop(columns=['Customer ID'], inplace=True)
        else:
            customer_ids = pd.Series(df.index.astype(str), index=df.index, name='Customer ID')

        if 'Last Activity Date' in df.columns:
            df.drop(columns=['Last Activity Date'], inplace=True)

        X = df[self.numerical_cols + self.categorical_cols]
        df[self.target_col] = df[self.target_col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
        y = df[self.target_col].astype(int)

        self.n_records          = len(df)
        self.overall_churn_rate = round(float(y.mean()) * 100, 2)

        # ── Customer Segmentation (before split, uses full dataset) ──────────
        self._run_segmentation(X, df)

        # ── Train / Test Split ───────────────────────────────────────────────
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X, y, customer_ids.loc[X.index], test_size=0.2, random_state=42, stratify=y
        )

        self.build_preprocessor()
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc  = self.preprocessor.transform(X_test)
        self.y_test  = y_test

        best_f1 = -1.0

        for name, model in self.models.items():
            model.fit(X_train_proc, y_train)
            preds = model.predict(X_test_proc)
            probs = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else preds

            cm = confusion_matrix(y_test, preds)
            metrics = {
                'Accuracy':         round(accuracy_score(y_test, preds), 4),
                'Precision':        round(precision_score(y_test, preds), 4),
                'Recall':           round(recall_score(y_test, preds), 4),
                'F1 Score':         round(f1_score(y_test, preds), 4),
                'ROC-AUC':          round(roc_auc_score(y_test, probs), 4),
                'Confusion Matrix': cm.tolist()
            }
            self.metrics_report[name] = metrics

            if metrics['F1 Score'] > best_f1:
                best_f1              = metrics['F1 Score']
                self.best_model      = model
                self.best_model_name = name
                self._best_preds     = preds
                self._best_probs     = probs

        # ── Hyperparameter Optimization on best model ────────────────────────
        optimized_model, best_params = self._hyperparameter_optimize(
            self.best_model_name, self.best_model, X_train_proc, y_train
        )
        # Re-evaluate optimized model; update only if it improves F1
        opt_preds = optimized_model.predict(X_test_proc)
        opt_probs = optimized_model.predict_proba(X_test_proc)[:, 1]
        opt_f1    = round(f1_score(y_test, opt_preds), 4)

        if opt_f1 >= best_f1:
            self.best_model  = optimized_model
            self._best_preds = opt_preds
            self._best_probs = opt_probs
            self.best_params = best_params
            # Update metrics_report entry with tuned scores
            opt_cm = confusion_matrix(y_test, opt_preds)
            self.metrics_report[self.best_model_name] = {
                'Accuracy':         round(accuracy_score(y_test, opt_preds), 4),
                'Precision':        round(precision_score(y_test, opt_preds), 4),
                'Recall':           round(recall_score(y_test, opt_preds), 4),
                'F1 Score':         opt_f1,
                'ROC-AUC':          round(roc_auc_score(y_test, opt_probs), 4),
                'Confusion Matrix': opt_cm.tolist()
            }

        # ── Feature names after encoding ─────────────────────────────────────
        cat_features = (self.preprocessor
                        .named_transformers_['cat']
                        .named_steps['onehot']
                        .get_feature_names_out(self.categorical_cols)
                        .tolist())
        self.engineered_features = self.numerical_cols + cat_features

        # ── Test-set risk table ───────────────────────────────────────────────
        self.test_results = pd.DataFrame({
            'Customer ID':      id_test.values,
            'Actual Churn':     y_test.values,
            'Predicted Churn':  self._best_preds,
            'Churn Probability': np.round(self._best_probs, 4)
        }).sort_values('Churn Probability', ascending=False).reset_index(drop=True)

    # -------------------------------------------------------------------------
    def get_feature_importance(self, top_n=10):
        if self.best_model is None:
            return []
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_[0])
        else:
            return []
        pairs = sorted(zip(self.engineered_features, importances), key=lambda p: p[1], reverse=True)
        return pairs[:top_n]

    # -------------------------------------------------------------------------
    def export_artifacts(self, output_dir='.'):
        import os
        os.makedirs(output_dir, exist_ok=True)

        model_path   = os.path.join(output_dir, 'best_model.pkl')
        preproc_path = os.path.join(output_dir, 'preprocessor.pkl')
        meta_path    = os.path.join(output_dir, 'metadata.json')

        with open(model_path,   'wb') as f: pickle.dump(self.best_model,    f)
        with open(preproc_path, 'wb') as f: pickle.dump(self.preprocessor,  f)

        metadata = {
            'best_model_name':    self.best_model_name,
            'best_params':        {k: str(v) for k, v in self.best_params.items()},
            'features':           self.engineered_features,
            'metrics':            self.metrics_report,
            'records_processed':  self.n_records,
            'overall_churn_rate': self.overall_churn_rate,
            'n_segments':         self.n_segments,
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print("Pipeline export complete. Engine optimized.")
        return model_path, preproc_path, meta_path


if __name__ == "__main__":
    engine = ChurnModelEngine()
    engine.train_and_evaluate('customer_churn_data.csv')
    engine.export_artifacts()
