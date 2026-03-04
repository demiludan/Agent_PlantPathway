#!/usr/bin/env python3
"""
Training Script for C3 vs C4 Classification (all 8 models)

Trains Neural Network (FNN) + 7 sklearn models on preprocessed A/Ci curve data.
Saves model checkpoints, training history, and timing info.

Adapted from Classification_CO2S.py step 4.

Usage:
    python train.py --experiment all_models_co2s
    python train.py --experiment all_models_co2s --config config.yaml
"""

import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import json
import joblib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score

from models.model import FNN, get_sklearn_models
from utils.YParams import YParams

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_training_logging(experiment_name):
    log_dir = Path("experiments") / experiment_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

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


def load_preprocessed(experiment_name, data_name, output_dir="experiments"):
    """Load preprocessed data from pickle."""
    data_path = Path(output_dir) / data_name / f"{data_name}_preprocessed.pkl"
    if not data_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {data_path}")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Loaded preprocessed data from {data_path}")
    return data


def train_neural_network(train_X, train_y, test_X, test_y, sample_len, params):
    """Train the FNN model."""
    device = DEVICE
    epochs = params.epochs
    batch_size = params.batch_size
    lr = params.learning_rate

    train_X_t = torch.from_numpy(train_X).to(device)
    train_y_t = torch.from_numpy(train_y).to(device)
    test_X_t = torch.from_numpy(test_X).to(device)
    test_y_t = torch.from_numpy(test_y).to(device)

    model = FNN(
        input_shape=(sample_len, 2),
        layer1=getattr(params, "fnn_layer1", 128),
        layer2=getattr(params, "fnn_layer2", 256),
        layer3=getattr(params, "fnn_layer3", 128),
        layer4=getattr(params, "fnn_layer4", 64),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    start = time.time()
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(train_X_t.size(0))
        total_loss, total_acc, count = 0, 0, 0

        for i in range(0, train_X_t.size(0), batch_size):
            idx = indices[i : i + batch_size]
            bx, by = train_X_t[idx], train_y_t[idx]
            optimizer.zero_grad()
            out = model(bx)
            loss = nn.BCEWithLogitsLoss()(out, by)
            loss.backward()
            optimizer.step()
            pred = (torch.sigmoid(out) > 0.5).long()
            total_loss += loss.item() * len(idx)
            total_acc += (pred == by).float().sum().item()
            count += len(idx)

        history["train_loss"].append(total_loss / count)
        history["train_acc"].append(total_acc / count)

        model.eval()
        with torch.no_grad():
            out = model(test_X_t)
            loss = nn.BCEWithLogitsLoss()(out, test_y_t)
            pred = (torch.sigmoid(out) > 0.5).long()
            history["test_loss"].append(loss.item())
            history["test_acc"].append((pred == test_y_t).float().mean().item())

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"train_loss={history['train_loss'][-1]:.4f} "
                f"test_acc={history['test_acc'][-1]*100:.2f}%"
            )

    train_time = time.time() - start
    return model, history, train_time


def train_all_models(experiment_name, config_path="config.yaml"):
    """Train all configured models."""
    logger.info(f"Starting training for: {experiment_name}")

    params = YParams(config_path, experiment_name)
    seed = params.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_name = getattr(params, "data_name", experiment_name)
    data = load_preprocessed(experiment_name, data_name)

    train_X = data["train_X"]
    test_X = data["test_X"]
    train_y = data["train_y"]
    test_y = data["test_y"]
    sample_len = data["sample_len"]

    # Output directories
    exp_dir = Path("experiments") / experiment_name
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    models_to_train = params.models
    results = {}

    # ---- Neural Network ----
    if "Neural Network" in models_to_train:
        logger.info("Training Neural Network...")
        nn_model, history, nn_time = train_neural_network(
            train_X, train_y, test_X, test_y, sample_len, params,
        )

        # Save NN checkpoint
        torch.save(nn_model.state_dict(), checkpoints_dir / "fnn_model.pth")

        # Get test predictions
        nn_model.eval()
        with torch.no_grad():
            test_t = torch.from_numpy(test_X).to(DEVICE)
            nn_prob = torch.sigmoid(nn_model(test_t)).cpu().numpy().ravel()
            nn_pred = (nn_prob > 0.5).astype(int)

        test_y_flat = test_y.ravel()
        results["Neural Network"] = {
            "pred": nn_pred.tolist(),
            "prob": nn_prob.tolist(),
            "time": nn_time,
            "accuracy": float(accuracy_score(test_y_flat, nn_pred)),
        }
        logger.info(
            f"Neural Network: {results['Neural Network']['accuracy']*100:.2f}% "
            f"({nn_time:.2f}s)"
        )

        # Save training history
        with open(exp_dir / "nn_training_history.pkl", "wb") as f:
            pickle.dump(history, f)

    # ---- Sklearn models ----
    train_flat = train_X.reshape(train_X.shape[0], -1)
    test_flat = test_X.reshape(test_X.shape[0], -1)
    train_y_flat = train_y.ravel()
    test_y_flat = test_y.ravel()

    sklearn_models = get_sklearn_models(params)
    for name, model in sklearn_models.items():
        if name not in models_to_train:
            continue
        logger.info(f"Training {name}...")
        start = time.time()
        model.fit(train_flat, train_y_flat)
        t = time.time() - start

        pred = model.predict(test_flat)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(test_flat)[:, 1]
        else:
            prob = pred.astype(float)

        acc = float(accuracy_score(test_y_flat, pred))
        results[name] = {
            "pred": pred.tolist(),
            "prob": prob.tolist(),
            "time": t,
            "accuracy": acc,
        }
        logger.info(f"{name}: {acc*100:.2f}% ({t:.2f}s)")

        # Save sklearn model
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, checkpoints_dir / f"{safe_name}_model.joblib")

    # Save combined results
    results_file = exp_dir / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"All models trained. Results saved to {results_file}")
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train C3/C4 classification models")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args()
    setup_training_logging(args.experiment)

    try:
        train_all_models(args.experiment, args.config)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
