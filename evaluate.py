#!/usr/bin/env python3
"""
Evaluation Script for C3 vs C4 Classification

Generates all figures and reports from inference results:
- NN training curves
- Accuracy comparison bar chart
- Confusion matrices (all 8 models)
- Cross-validation results
- Performance heatmap
- Summary report

Adapted from Classification_CO2S.py steps 5-7.

Usage:
    python evaluate.py --experiment all_models_co2s --split test
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

from models.model import FNN, get_sklearn_models
from utils.YParams import YParams

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLORS = {
    "Neural Network": "#3498db",
    "Logistic Regression": "#e74c3c",
    "Random Forest": "#2ecc71",
    "SVM": "#9b59b6",
    "Gradient Boosting": "#1abc9c",
    "AdaBoost": "#e67e22",
    "Naive Bayes": "#34495e",
    "Decision Tree": "#7f8c8d",
}


def setup_evaluation_logging(experiment_name):
    log_dir = Path("experiments") / experiment_name / "evaluate" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluate_{timestamp}.log"

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


def cross_validate(data_points_norm, labels, sample_len, params):
    """Run cross-validation for all models including NN."""
    logger.info("Running cross-validation...")
    n_folds = params.n_cv_folds
    seed = params.random_seed

    X_flat = data_points_norm.reshape(data_points_norm.shape[0], -1)
    y = labels.ravel()
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_results = {}

    # NN manual CV
    if "Neural Network" in params.models:
        nn_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_flat, y)):
            X_train_fold = data_points_norm[train_idx]
            X_val_fold = data_points_norm[val_idx]
            y_train_fold = labels[train_idx]
            y_val_fold = labels[val_idx]

            X_train_t = torch.from_numpy(X_train_fold).to(DEVICE)
            X_val_t = torch.from_numpy(X_val_fold).to(DEVICE)
            y_train_t = torch.from_numpy(y_train_fold).to(DEVICE)
            y_val_t = torch.from_numpy(y_val_fold).to(DEVICE)

            model = FNN(
                input_shape=(sample_len, 2),
                layer1=getattr(params, "fnn_layer1", 128),
                layer2=getattr(params, "fnn_layer2", 256),
                layer3=getattr(params, "fnn_layer3", 128),
                layer4=getattr(params, "fnn_layer4", 64),
            ).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

            for epoch in range(50):
                model.train()
                indices = torch.randperm(X_train_t.size(0))
                for i in range(0, X_train_t.size(0), params.batch_size):
                    idx = indices[i : i + params.batch_size]
                    optimizer.zero_grad()
                    out = model(X_train_t[idx])
                    loss = nn.BCEWithLogitsLoss()(out, y_train_t[idx])
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                out = model(X_val_t)
                pred = (torch.sigmoid(out) > 0.5).long()
                acc = (pred == y_val_t).float().mean().item()
                nn_scores.append(acc)

        nn_scores = np.array(nn_scores)
        cv_results["Neural Network"] = {
            "mean": float(nn_scores.mean()),
            "std": float(nn_scores.std()),
            "scores": nn_scores.tolist(),
        }
        logger.info(
            f"Neural Network CV: {nn_scores.mean()*100:.2f}% +/- {nn_scores.std()*100:.2f}%"
        )

    # Sklearn CV
    sklearn_models = get_sklearn_models(params)
    for name, model in sklearn_models.items():
        if name not in params.models:
            continue
        scores = cross_val_score(model, X_flat, y, cv=cv, scoring="accuracy")
        cv_results[name] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "scores": scores.tolist(),
        }
        logger.info(f"{name} CV: {scores.mean()*100:.2f}% +/- {scores.std()*100:.2f}%")

    return cv_results


def create_figures(results, nn_history, cv_results, output_dir):
    """Create all evaluation figures."""
    logger.info("Creating figures...")

    sorted_models = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    names = [m[0] for m in sorted_models]

    # Fig 2: NN Training curves
    if nn_history:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(nn_history["train_loss"], "b-", label="Train", linewidth=2)
        axes[0].plot(nn_history["test_loss"], "r-", label="Test", linewidth=2)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("Neural Network: Loss", fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot([a * 100 for a in nn_history["train_acc"]], "b-", label="Train", linewidth=2)
        axes[1].plot([a * 100 for a in nn_history["test_acc"]], "r-", label="Test", linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy (%)", fontsize=12)
        axes[1].set_title("Neural Network: Accuracy", fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "02_nn_training.png", dpi=150)
        plt.close()
        logger.info("Saved: 02_nn_training.png")

    # Fig 3: Accuracy comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    accs = [results[n]["accuracy"] * 100 for n in names]
    colors = [COLORS.get(n, "#95a5a6") for n in names]
    ax.barh(range(len(names)), accs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=14)
    ax.set_xlim([0, 105])
    ax.grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(accs):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "03_accuracy_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved: 03_accuracy_comparison.png")

    # Fig 4: Confusion matrices (4x2 grid)
    n_models = len(names)
    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).ravel()

    for idx, name in enumerate(names):
        ax = axes[idx]
        cm = np.array(results[name]["cm"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["C3", "C4"], fontsize=10)
        ax.set_yticklabels(["C3", "C4"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        acc = results[name]["accuracy"] * 100
        ax.set_title(f"{name}\n({acc:.1f}%)", fontsize=11, fontweight="bold")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=color, fontsize=14, fontweight="bold")

    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Confusion Matrices for All Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "04_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: 04_confusion_matrices.png")

    # Fig 5: Cross-validation
    if cv_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        cv_names = sorted(cv_results.keys(), key=lambda x: cv_results[x]["mean"], reverse=True)
        means = [cv_results[n]["mean"] * 100 for n in cv_names]
        stds = [cv_results[n]["std"] * 100 for n in cv_names]
        colors_cv = [COLORS.get(n, "#95a5a6") for n in cv_names]
        ax.barh(range(len(cv_names)), means, xerr=stds, color=colors_cv, capsize=5, alpha=0.8)
        ax.set_yticks(range(len(cv_names)))
        ax.set_yticklabels(cv_names, fontsize=11)
        ax.set_xlabel("CV Accuracy (%)", fontsize=12)
        ax.set_title(f"Cross-Validation Results", fontsize=14)
        ax.set_xlim([0, 110])
        ax.grid(True, alpha=0.3, axis="x")
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + s + 1, i, f"{m:.1f}+/-{s:.1f}%", va="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_dir / "05_cross_validation.png", dpi=150)
        plt.close()
        logger.info("Saved: 05_cross_validation.png")

    # Fig 6: Performance heatmap
    metrics_names = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    hm_data = np.array([[results[n][m] for m in metrics_names] for n in names])

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(hm_data, cmap="RdYlGn", vmin=0.7, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metric_labels)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_title("Model Performance Heatmap", fontsize=14, fontweight="bold")
    for i in range(len(names)):
        for j in range(len(metrics_names)):
            val = hm_data[i, j]
            color = "white" if val < 0.85 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
    plt.colorbar(im, shrink=0.8, label="Score")
    plt.tight_layout()
    plt.savefig(output_dir / "06_heatmap.png", dpi=150)
    plt.close()
    logger.info("Saved: 06_heatmap.png")


def save_report(results, cv_results, output_dir, experiment_name):
    """Save CSV results and text report."""
    import pandas as pd

    sorted_models = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    # CSV
    rows = []
    for name, res in results.items():
        cv = cv_results.get(name, {})
        rows.append({
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1": res["f1"],
            "AUC": res["auc"],
            "CV_Mean": cv.get("mean"),
            "CV_Std": cv.get("std"),
        })
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    df.to_csv(output_dir / "results.csv", index=False)
    logger.info("Saved: results.csv")

    # Metrics JSON (for result_loader)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metrics_json = {
        "models": {name: {k: v for k, v in res.items() if k not in ("pred", "prob")}
                   for name, res in results.items()},
        "cv_results": cv_results,
    }
    with open(metrics_dir / f"{experiment_name}_test_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    # Text report
    best = sorted_models[0]
    cv_best = max(cv_results.items(), key=lambda x: x[1]["mean"]) if cv_results else (None, {})

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    lines = [
        "=" * 70,
        "CLASSIFICATION REPORT: AnetCO2 vs CO2S",
        "=" * 70,
        "",
        "MODEL RANKING (by Test Accuracy):",
        "-" * 50,
    ]
    for i, (name, res) in enumerate(sorted_models, 1):
        lines.append(f"{i}. {name:25s} {res['accuracy']*100:.2f}%")

    lines.append(f"\nBEST MODEL: {best[0]}")
    lines.append(f"   Accuracy: {best[1]['accuracy']*100:.2f}%")
    lines.append(f"   F1-Score: {best[1]['f1']*100:.2f}%")
    lines.append(f"   AUC-ROC:  {best[1]['auc']:.4f}")

    if cv_best[0]:
        lines.append(f"\nMOST RELIABLE (CV): {cv_best[0]}")
        lines.append(f"   CV Accuracy: {cv_best[1]['mean']*100:.2f}% +/- {cv_best[1]['std']*100:.2f}%")

    summary_text = "\n".join(lines)
    summary_file = reports_dir / f"{experiment_name}_test_summary.txt"
    summary_file.write_text(summary_text)
    logger.info("Saved: summary report")

    return sorted_models


def evaluate_model(experiment_name, split="test", config_path="config.yaml", **kwargs):
    """Main evaluation function."""
    logger.info(f"Evaluating {experiment_name}, split={split}")

    params = YParams(config_path, experiment_name)
    data_name = getattr(params, "data_name", experiment_name)

    # Load inference results
    inference_dir = Path("experiments") / experiment_name / "inference" / "best_model"
    pred_file = inference_dir / f"{experiment_name}_{split}_predictions.pkl"
    with open(pred_file, "rb") as f:
        inference_data = pickle.load(f)
    results = inference_data["results"]

    # Load preprocessed data for CV
    data_path = Path("experiments") / data_name / f"{data_name}_preprocessed.pkl"
    with open(data_path, "rb") as f:
        preproc = pickle.load(f)

    # Normalize all data for CV
    data_points = preproc["data_points"]
    labels = preproc["labels"]
    sample_len = preproc["sample_len"]
    mean = preproc["mean"]
    std_val = preproc["std"]
    data_norm = (data_points - mean) / std_val

    # Cross-validation
    cv_results = cross_validate(data_norm, labels, sample_len, params)

    # Load NN training history
    nn_history = None
    history_path = Path("experiments") / experiment_name / "nn_training_history.pkl"
    if history_path.exists():
        with open(history_path, "rb") as f:
            nn_history = pickle.load(f)

    # Create output directory
    eval_dir = Path("experiments") / experiment_name / "evaluate" / "best_model"
    figures_dir = eval_dir / "figures" / split
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    create_figures(results, nn_history, cv_results, figures_dir)

    # Save report
    save_report(results, cv_results, eval_dir, experiment_name)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate C3/C4 classification")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--site_codes", default=None)
    parser.add_argument("--lead_time_step", default=None)

    args = parser.parse_args()
    setup_evaluation_logging(args.experiment)

    try:
        evaluate_model(args.experiment, args.split, args.config)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
