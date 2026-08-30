"""
FraudSentinel: Section 3 & 4 - Defense-Only Decision Layer & SHAP Audit Trail

STRICT DEFENSE-ONLY ARCHITECTURE:
This module NEVER blocks, cancels, or reverses a transaction automatically. 
It operates strictly as a defense-and-routing mechanism. Its sole mandate is to 
classify risk, append an immutable, explainable audit log, and route the transaction.
"""

import os
import json
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib

# Import the new explainability functions
from explain import init_explainer, generate_explanation_text

logger = logging.getLogger(__name__)

FLAG_THRESHOLD = 0.55 
ESCALATE_THRESHOLD = 0.35 

def load_pipeline_artifacts() -> tuple:
    logger.info("Loading model and full test batch...")
    model = joblib.load("artifacts/fraud_model.joblib")
    X_batch = pd.read_parquet("data/processed/X_test.parquet")
    y_true = pd.read_parquet("data/processed/y_test.parquet")["isFraud"]
    return model, X_batch, y_true

def triage_transaction(score: float) -> str:
    if score >= FLAG_THRESHOLD:
        return "FLAG"
    elif score >= ESCALATE_THRESHOLD:
        return "ESCALATE"
    else:
        return "PASS"

def process_batch_and_log(
    model,
    X_batch: pd.DataFrame,
    scores: np.ndarray, 
    output_path: str = "outputs/audit_trail.jsonl"
) -> list:
    """
    Routes a batch of transactions and writes a structured, immutable JSONL audit trail.
    """
    logger.info(f"Processing triage and writing audit ledger to {output_path}...")
    
    # 1. Identify which rows actually need explanations to save compute
    needs_explanation_mask = scores >= ESCALATE_THRESHOLD
    X_needs_explain = X_batch[needs_explanation_mask]
    
    logger.info(f"Computing SHAP explanations for {len(X_needs_explain)} flagged/escalated records...")
    explainer = init_explainer(model)
    
    # Vectorized SHAP computation for the subset
    shap_vals_subset = explainer.shap_values(X_needs_explain)
    if isinstance(shap_vals_subset, list):
        shap_vals_subset = shap_vals_subset[1]
        
    # Map the explanations back using the DataFrame index
    explanation_map = {}
    feature_names = X_batch.columns.tolist()
    
    for i, idx in enumerate(X_needs_explain.index):
        row_shap = shap_vals_subset[i]
        row_vals = X_needs_explain.iloc[i]
        explanation_map[idx] = generate_explanation_text(row_shap, feature_names, row_vals)

    audit_records = []
    
    with open(output_path, "w") as f:
        for idx, (original_idx, row_vals) in enumerate(X_batch.iterrows()):
            score = float(scores[idx])
            decision = triage_transaction(score)
            current_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
            # Fetch the generated explanation if it exists, otherwise leave blank
            explanation = explanation_map.get(original_idx, "")
            
            record = {
                "transaction_id": str(original_idx),
                "fraud_probability_score": round(score, 4),
                "decision": decision,
                "threshold_used": FLAG_THRESHOLD,
                "timestamp": current_time,
                "explanation": explanation
            }
            
            audit_records.append(record)
            f.write(json.dumps(record) + "\n")
            
    return audit_records

def generate_batch_summary(audit_records: list, y_true: pd.Series):
    actual_fraud = y_true.values.astype(bool)
    total_actual_fraud = actual_fraud.sum()
    
    counts = {"FLAG": 0, "ESCALATE": 0, "PASS": 0}
    fraud_caught = {"FLAG": 0, "ESCALATE": 0, "PASS": 0}
    
    for idx, record in enumerate(audit_records):
        decision = record["decision"]
        counts[decision] += 1
        if actual_fraud[idx]:
            fraud_caught[decision] += 1

    report = [
        "\n==========================================",
        "      DECISION ENGINE BATCH SUMMARY       ",
        "==========================================\n",
        f"Total Transactions Processed: {len(audit_records)}",
        f"Total Actual Fraud in Batch:  {total_actual_fraud}\n",
        "--- Routing Distribution & Fraud Capture ---"
    ]
    
    for bucket in ["FLAG", "ESCALATE", "PASS"]:
        bucket_total = counts[bucket]
        captured_fraud = fraud_caught[bucket]
        capture_rate = (captured_fraud / total_actual_fraud * 100) if total_actual_fraud > 0 else 0
        report.append(f"[{bucket}]")
        report.append(f"  Volume Routed: {bucket_total} transactions")
        report.append(f"  Contained {captured_fraud} actual fraud cases ({capture_rate:.1f}% of all fraud)")
    
    summary_text = "\n".join(report)
    logger.info(summary_text)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.makedirs("outputs", exist_ok=True)
    
    model, X_batch, y_true = load_pipeline_artifacts()
    
    logger.info("Executing model scoring on batch...")
    scores = model.predict_proba(X_batch)[:, 1]
    
    audit_records = process_batch_and_log(model, X_batch, scores, "outputs/audit_trail.jsonl")
    generate_batch_summary(audit_records, y_true)