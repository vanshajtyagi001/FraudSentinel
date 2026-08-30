"""
Trains a LightGBM classifier with proper cross-validation and evaluates
business impact through a cost-weighted threshold sensitivity matrix.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)

# ---------------------------------------------------------
# Configuration & Business Logic Constants
# ---------------------------------------------------------
# Cost of a False Positive (e.g., blocking a good customer, support overhead, friction)
FP_COST = 5
# Cost of a False Negative (e.g., chargeback fee, lost merchandise, network fines)
FN_COST = 100

logger = logging.getLogger(__name__)


def load_data() -> tuple:
    """Loads preprocessed datasets """
    logger.info("Loading preprocessed parquet data...")
    X_train = pd.read_parquet("data/processed/X_train.parquet")
    X_test = pd.read_parquet("data/processed/X_test.parquet")
    y_train = pd.read_parquet("data/processed/y_train.parquet")["isFraud"]
    y_test = pd.read_parquet("data/processed/y_test.parquet")["isFraud"]
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    """
    Trains a LightGBM model using cross-validated hyperparameter tuning.
    
    Model Justification (LightGBM vs XGBoost):
    LightGBM builds trees leaf-wise rather than depth-wise, making it substantially 
    faster for datasets >500k rows. Crucially, it natively handles missing values by 
    learning the optimal split direction for NaNs during training, and avoids the need 
    for SMOTE (which interpolates fake data and ruins probability calibration for SHAP).
    """
    logger.info("Calculating scale_pos_weight for class imbalance...")
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    imbalance_ratio = neg_count / pos_count
    logger.info(f"Imbalance ratio calculated at {imbalance_ratio:.2f}")

    # Base estimator
    lgbm = LGBMClassifier(
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        n_jobs=-1
    )

    # Lightweight grid for Buildathon time constraints
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63],
        'max_depth': [5, 8]
    }

    logger.info("Starting 3-fold RandomizedSearchCV for hyperparameter tuning...")
    # Stratified K-Fold ensures each fold maintains the 3.5% fraud rate
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_grid,
        n_iter=5,  # Bound iterations to keep training time manageable
        scoring='average_precision',  # Tunes specifically for PR-AUC
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    logger.info(f"Best parameters found: {search.best_params_}")
    
    return search.best_estimator_


def evaluate_and_cost_model(model: LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    """
    Evaluates the model on the held-out test set and calculates cost-weighted metrics.
    
    Metric Justification (PR-AUC over ROC-AUC):
    In a dataset where 96.5% of transactions are legitimate, ROC-AUC is dangerously 
    misleading. The False Positive Rate (FPR) denominator is massive, meaning millions 
    of false positives barely move the ROC curve. PR-AUC evaluates precision strictly 
    against the positive class, measuring exactly what matters: when we flag fraud, 
    how often are we right?
    """
    logger.info("Generating predictions on held-out test set...")
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    # 0.5 Default Baseline
    y_pred_default = (y_probs >= 0.5).astype(int)
    cm_default = confusion_matrix(y_test, y_pred_default)
    
    report = []
    report.append("==========================================")
    report.append("      FRAUDSENTINEL EVALUATION REPORT     ")
    report.append("==========================================\n")
    report.append(f"PR-AUC Score: {pr_auc:.4f}\n")
    
    report.append("--- Baseline Metrics (Threshold = 0.5) ---")
    report.append(f"Precision: {precision_score(y_test, y_pred_default):.4f}")
    report.append(f"Recall:    {recall_score(y_test, y_pred_default):.4f}")
    report.append(f"F1 Score:  {f1_score(y_test, y_pred_default):.4f}\n")
    
    report.append("--- Confusion Matrix (Threshold = 0.5) ---")
    report.append(f"True Negatives:  {cm_default[0][0]}")
    report.append(f"False Positives: {cm_default[0][1]} (Friction)")
    report.append(f"False Negatives: {cm_default[1][0]} (Missed Fraud)")
    report.append(f"True Positives:  {cm_default[1][1]} (Caught Fraud)\n")
    
    report.append("--- Threshold Sensitivity & Cost Matrix ---")
    report.append(f"Assuming FP Cost = ${FP_COST}, FN Cost = ${FN_COST}\n")
    report.append(f"{'Thresh':<8} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'FP':<5} | {'FN':<5} | {'Total Cost':<12}")
    report.append("-" * 70)
    
    thresholds_to_test = np.arange(0.10, 0.95, 0.05)
    min_cost = float('inf')
    best_thresh = None
    
    for t in thresholds_to_test:
        y_pred_t = (y_probs >= t).astype(int)
        prec_t = precision_score(y_test, y_pred_t, zero_division=0)
        rec_t = recall_score(y_test, y_pred_t)
        f1_t = f1_score(y_test, y_pred_t)
        
        cm_t = confusion_matrix(y_test, y_pred_t)
        fp_t = cm_t[0][1]
        fn_t = cm_t[1][0]
        
        cost_t = (fp_t * FP_COST) + (fn_t * FN_COST)
        
        if cost_t < min_cost:
            min_cost = cost_t
            best_thresh = t
            
        report.append(f"{t:<8.2f} | {prec_t:<7.4f} | {rec_t:<7.4f} | {f1_t:<7.4f} | {fp_t:<5} | {fn_t:<5} | ${cost_t:<11,.2f}")

    report.append("-" * 70)
    report.append(f"\nOPTIMAL THRESHOLD: {best_thresh} (Min Cost: ${min_cost:,.2f})")
    
    return "\n".join(report)


if __name__ == "__main__":
    # Pipeline entry point logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    
    model = train_model(X_train, y_train)
    
    # Save the trained model for downstream inference and SHAP explanations
    logger.info("Saving trained LightGBM model to artifacts/fraud_model.joblib...")
    joblib.dump(model, "artifacts/fraud_model.joblib")
    
    report_str = evaluate_and_cost_model(model, X_test, y_test)
    
    # Output to console
    print("\n" + report_str)
    
    # Output to persistent file
    output_path = "outputs/evaluation_results.txt"
    with open(output_path, "w") as f:
        f.write(report_str)
    logger.info(f"Evaluation report successfully written to {output_path}")