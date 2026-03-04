#!/usr/bin/env python3
"""
Inference Script for C3 vs C4 Classification

Loads trained models and generates predictions + probabilities on the test set.
Computes per-model metrics (accuracy, precision, recall, F1, AUC).

Adapted from Classification_CO2S.py step 4 (metrics portion).

Usage:
    python inference.py --experiment all_models_co2s --split test
    python inference.py --experiment all_models_co2s --split train
"""

import sys
import argparse
import numpy as np
import torch
import pickle
import json
import joblib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from models.model import FNN
from utils.YParams import YParams

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_inference_logging(experiment_name):
    log_dir = Path("experiments") / experiment_name / "inference" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"inference_{timestamp}.log"

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)


def run_inference(experiment_name, split="test", config_path="config.yaml", **kwargs):
    """Run inference for all trained models."""
    logger.info(f"Running inference for {experiment_name}, split={split}")

    params = YParams(config_path, experiment_name)
    data_name = getattr(params, "data_name", experiment_name)

    # Load preprocessed data
    data_path = Path("experiments") / data_name / f"{data_name}_preprocessed.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    if split == "train":
        X = data["train_X"]
        y = data["train_y"].ravel()
    else:
        X = data["test_X"]
        y = data["test_y"].ravel()

    sample_len = data["sample_len"]
    X_flat = X.reshape(X.shape[0], -1)

    exp_dir = Path("experiments") / experiment_name
    checkpoints_dir = exp_dir / "checkpoints"
    inference_dir = exp_dir / "inference" / "best_model"
    inference_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    models_to_run = params.models

    # ---- Neural Network ----
    fnn_path = checkpoints_dir / "fnn_model.pth"
    if "Neural Network" in models_to_run and fnn_path.exists():
        logger.info("Running FNN inference...")
        model = FNN(
            input_shape=(sample_len, 2),
            layer1=getattr(params, "fnn_layer1", 128),
            layer2=getattr(params, "fnn_layer2", 256),
            layer3=getattr(params, "fnn_layer3", 128),
            layer4=getattr(params, "fnn_layer4", 64),
        ).to(DEVICE)
        model.load_state_dict(torch.load(fnn_path, map_location=DEVICE, weights_only=True))
        model.eval()

        with torch.no_grad():
            X_t = torch.from_numpy(X).to(DEVICE)
            prob = torch.sigmoid(model(X_t)).cpu().numpy().ravel()
            pred = (prob > 0.5).astype(int)

        results["Neural Network"] = _compute_metrics(y, pred, prob)
        logger.info(f"Neural Network: {results['Neural Network']['accuracy']*100:.2f}%")

    # ---- Sklearn models ----
    for name in models_to_run:
        if name == "Neural Network":
            continue
        safe_name = name.lower().replace(" ", "_")
        model_path = checkpoints_dir / f"{safe_name}_model.joblib"
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue

        logger.info(f"Running {name} inference...")
        model = joblib.load(model_path)
        pred = model.predict(X_flat)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_flat)[:, 1]
        else:
            prob = pred.astype(float)

        results[name] = _compute_metrics(y, pred, prob)
        logger.info(f"{name}: {results[name]['accuracy']*100:.2f}%")

    # Save predictions
    output_data = {
        "results": results,
        "labels": y.tolist(),
        "split": split,
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
    }

    output_file = inference_dir / f"{experiment_name}_{split}_predictions.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(output_data, f)

    logger.info(f"Inference results saved to {output_file}")
    logger.info("Inference completed!")


def _compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics for a single model."""
    return {
        "pred": y_pred.tolist(),
        "prob": y_prob.tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run C3/C4 inference")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args()
    setup_inference_logging(args.experiment)

    try:
        run_inference(args.experiment, args.split, args.config)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
