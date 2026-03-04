#!/usr/bin/env python3
"""
Data Preprocessing for C3 vs C4 Photosynthetic Pathway Classification

Loads parquet measurement data, extracts A/Ci curves, resamples to uniform
length, splits into train/test, normalizes, and saves preprocessed pickle.

Adapted from Classification_CO2S.py steps 1-3.

Usage:
    python data_preprocessing.py --experiment co2s_c3c4
    python data_preprocessing.py --experiment co2s_c3c4 --config config.yaml
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import logging
import torch
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils.YParams import YParams

logger = logging.getLogger(__name__)


def setup_preprocessing_logging(experiment_name):
    log_dir = Path("experiments") / experiment_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"preprocessing_{timestamp}.log"

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


def load_data(params, output_dir):
    """Load parquet measurements and create A/Ci overview figure."""
    base_path = Path(params.base_path)
    measurements_path = base_path / params.measurements_file

    logger.info(f"Loading: {measurements_path}")
    measurements = pd.read_parquet(measurements_path)

    x_var = params.x_variable
    label_col = params.label_column
    curve_id_col = params.curve_id_column

    logger.info(
        f"Dataset: {len(measurements):,} rows, "
        f"{measurements[curve_id_col].nunique()} curves"
    )

    # Create A/Ci curves overview figure
    valid_data = measurements.dropna(subset=["AnetCO2", x_var, label_col])

    fig, ax = plt.subplots(figsize=(14, 9))
    for pathway, color in zip(["C3", "C4"], ["#3498db", "#e74c3c"]):
        pdata = valid_data[valid_data[label_col] == pathway]
        n_curves = pdata[curve_id_col].nunique()
        for cid in pdata[curve_id_col].unique():
            curve = pdata[pdata[curve_id_col] == cid].sort_values(x_var)
            ax.plot(
                curve[x_var], curve["AnetCO2"],
                color=color, alpha=0.6, linewidth=1, marker=".", markersize=3,
            )
        ax.plot([], [], color=color, linewidth=2, marker="o", markersize=5,
                label=f"{pathway} ({n_curves} curves)")

    ax.set_xlabel(f"{x_var} (ppm)", fontsize=18)
    ax.set_ylabel("AnetCO2 (umol m-2 s-1)", fontsize=18)
    ax.set_title("A/Ci Curves: C3 vs C4 Comparison", fontsize=20, fontweight="bold")
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.1)
    plt.tight_layout()
    plt.savefig(output_dir / "01_data_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: 01_data_overview.png")

    return measurements


def prepare_data(measurements, params):
    """Extract relevant columns and encode labels."""
    x_var = params.x_variable
    label_col = params.label_column
    curve_id_col = params.curve_id_column

    cols = [curve_id_col, "AnetCO2", x_var, label_col]
    data = measurements[cols].dropna().sort_values([curve_id_col, x_var])

    le = LabelEncoder()
    data["pathway_label"] = le.fit_transform(data[label_col])

    logger.info(f"Valid rows: {len(data):,}")
    logger.info(f"Labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return data, le


def resample_curves(data, params):
    """Resample all curves to uniform length."""
    curve_id_col = params.curve_id_column
    x_var = params.x_variable

    grouped = data.groupby(curve_id_col)
    num_curves = len(grouped)
    sample_len = min(len(df) for _, df in grouped)

    logger.info(f"Curves: {num_curves}, Sample length: {sample_len}")

    data_points = np.zeros((num_curves, sample_len, 2), dtype=np.float32)
    labels = np.zeros((num_curves, 1), dtype=np.float32)

    for i, (_, df) in enumerate(grouped):
        df = df.sort_values(x_var)
        x = df[x_var].values
        y = df["AnetCO2"].values
        x_target = np.linspace(x.min(), x.max(), sample_len)
        idx = np.clip(np.searchsorted(x, x_target), 0, len(x) - 1)
        data_points[i] = np.stack([y[idx], x[idx]], axis=1)
        labels[i] = df["pathway_label"].iloc[0]

    c3 = int((labels == 0).sum())
    c4 = int((labels == 1).sum())
    logger.info(f"C3: {c3}, C4: {c4}")

    return data_points, labels, sample_len


def preprocess_dataset(experiment_name, config_path="config.yaml"):
    """Main preprocessing pipeline."""
    logger.info(f"Starting preprocessing for: {experiment_name}")

    params = YParams(config_path, experiment_name)
    random_seed = params.random_seed
    test_size = params.test_size

    # Create output directory
    output_dir = Path(params.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    measurements = load_data(params, output_dir)

    # Step 2: Prepare data
    data, le = prepare_data(measurements, params)

    # Step 3: Resample curves
    data_points, labels, sample_len = resample_curves(data, params)

    # Step 4: Train/test split
    train_X, test_X, train_y, test_y = train_test_split(
        data_points, labels,
        test_size=test_size, random_state=random_seed, stratify=labels,
    )
    logger.info(f"Split: {train_X.shape[0]} train / {test_X.shape[0]} test")

    # Step 5: Normalize
    train_X_t = torch.from_numpy(train_X)
    test_X_t = torch.from_numpy(test_X)

    mean = train_X_t.mean(dim=(0, 1), keepdim=True)
    std = train_X_t.std(dim=(0, 1), keepdim=True)

    train_X_norm = ((train_X_t - mean) / std).numpy()
    test_X_norm = ((test_X_t - mean) / std).numpy()

    # Step 6: Save preprocessed data
    preprocessed = {
        "train_X": train_X_norm,
        "test_X": test_X_norm,
        "train_y": train_y,
        "test_y": test_y,
        "train_X_raw": train_X,
        "test_X_raw": test_X,
        "data_points": data_points,
        "labels": labels,
        "sample_len": sample_len,
        "mean": mean.numpy(),
        "std": std.numpy(),
        "label_encoder_classes": le.classes_.tolist(),
        "num_classes": len(le.classes_),
        "class_names": le.classes_.tolist(),
        "x_variable": params.x_variable,
        "config": params.params,
    }

    output_file = output_dir / f"{experiment_name}_preprocessed.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(preprocessed, f)

    stats = {"mean": mean.numpy(), "std": std.numpy(), "sample_len": sample_len}
    stats_file = output_dir / f"{experiment_name}_global_stats.pkl"
    with open(stats_file, "wb") as f:
        pickle.dump(stats, f)

    logger.info(f"Preprocessed data saved to {output_file}")
    logger.info("Preprocessing completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess C3/C4 classification data")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args()
    setup_preprocessing_logging(args.experiment)

    try:
        preprocess_dataset(args.experiment, args.config)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
