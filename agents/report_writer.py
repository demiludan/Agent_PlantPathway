from __future__ import annotations

"""
Report Writer for C3 vs C4 Photosynthetic Pathway Classification

Generates Markdown reports from evaluation artifacts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_json(obj, fallback="{}"):
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return fallback


def _find_figure(base_dir: Path, pattern: str) -> str:
    if not base_dir.exists():
        return ""
    matches = list(base_dir.rglob(f"*{pattern}*"))
    if matches:
        return str(matches[0])
    return ""


def generate_report_markdown(
    experiment: str,
    artifacts: Dict,
    config_path: Path,
) -> str:
    """Generate a Markdown report from evaluation artifacts."""
    # Find figures from evaluation output
    eval_dir = Path("experiments") / experiment / "evaluate" / "best_model"
    figures_dir = eval_dir / "figures" / "test"
    exp_dir = Path("experiments") / experiment

    # Load metrics
    metrics_path = eval_dir / "metrics" / f"{experiment}_test_metrics.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    # Load summary
    summary_path = eval_dir / "reports" / f"{experiment}_test_summary.txt"
    summary_text = ""
    if summary_path.exists():
        summary_text = summary_path.read_text()

    models_metrics = metrics.get("models", {})
    cv_results = metrics.get("cv_results", {})

    # Build model ranking
    sorted_models = sorted(models_metrics.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True)

    lines = [
        "# C3 vs C4 Photosynthetic Pathway Classification Report",
        "",
        f"This report evaluates 8 machine learning models for classifying C3 vs C4 "
        f"photosynthetic pathways using A/Ci curves.",
        "",
    ]

    # Data overview
    data_fig = _find_figure(exp_dir, "01_data_overview")
    if data_fig:
        lines.extend([f"![Data Overview]({data_fig})", ""])

    lines.extend([
        "## 1. Experiment Overview",
        "",
        "This experiment classifies plant photosynthetic pathways (C3 vs C4) from "
        "A/Ci response curves. Each curve is resampled to uniform length, normalized, "
        "then fed to 8 ML models: Neural Network (FNN), Logistic Regression, Random Forest, "
        "SVM, Gradient Boosting, AdaBoost, Naive Bayes, and Decision Tree.",
        "",
        "## 2. Model Comparison",
        "",
        "### Accuracy Ranking",
        "",
        "| Rank | Model | Accuracy | F1-Score | AUC-ROC |",
        "|------|-------|----------|----------|---------|",
    ])

    for rank, (name, m) in enumerate(sorted_models, 1):
        lines.append(
            f"| {rank} | {name} | {m.get('accuracy', 0)*100:.2f}% | "
            f"{m.get('f1', 0)*100:.2f}% | {m.get('auc', 0):.4f} |"
        )
    lines.append("")

    acc_fig = _find_figure(figures_dir, "03_accuracy_comparison")
    if acc_fig:
        lines.extend([f"![Accuracy Comparison]({acc_fig})", ""])

    hm_fig = _find_figure(figures_dir, "06_heatmap")
    if hm_fig:
        lines.extend(["### Performance Heatmap", "", f"![Heatmap]({hm_fig})", ""])

    cm_fig = _find_figure(figures_dir, "04_confusion_matrices")
    if cm_fig:
        lines.extend(["### Confusion Matrices", "", f"![Confusion Matrices]({cm_fig})", ""])

    # NN training
    nn_fig = _find_figure(figures_dir, "02_nn_training")
    if nn_fig:
        lines.extend([
            "## 3. Neural Network Analysis",
            "",
            f"![NN Training Curves]({nn_fig})",
            "",
        ])

    # CV results
    cv_fig = _find_figure(figures_dir, "05_cross_validation")
    lines.extend(["## 4. Cross-Validation Results", ""])
    if cv_fig:
        lines.extend([f"![Cross-Validation]({cv_fig})", ""])

    if cv_results:
        lines.extend([
            "| Model | CV Accuracy | Std Dev |",
            "|-------|-------------|---------|",
        ])
        cv_sorted = sorted(cv_results.items(), key=lambda x: x[1].get("mean", 0), reverse=True)
        for name, cv in cv_sorted:
            lines.append(f"| {name} | {cv['mean']*100:.2f}% | {cv['std']*100:.2f}% |")
        lines.append("")

    # Discussion
    lines.extend([
        "## 5. Discussion and Recommendations",
        "",
    ])

    if sorted_models:
        best_name, best_m = sorted_models[0]
        lines.append(
            f"**Best model by test accuracy**: {best_name} "
            f"({best_m.get('accuracy', 0)*100:.2f}%)"
        )

    if cv_results:
        cv_best = max(cv_results.items(), key=lambda x: x[1].get("mean", 0))
        lines.append(
            f"\n**Most reliable (CV)**: {cv_best[0]} "
            f"({cv_best[1]['mean']*100:.2f}% +/- {cv_best[1]['std']*100:.2f}%)"
        )

    lines.extend([
        "",
        "### Recommendations",
        "",
        "- Consider ensemble methods combining top-performing models",
        "- Increase dataset size with additional A/Ci curve measurements",
        "- Experiment with different curve resampling strategies",
        "- Test with additional x-variables (Ci, PAR) for robustness",
        "",
    ])

    return "\n".join(lines)


def final_report_writer(
    experiment: str,
    artifacts: Dict,
    output_dir: Path,
    config_path: Path,
    **kwargs,
) -> Path:
    """Generate the final Markdown report and write it to disk."""
    report_md = generate_report_markdown(experiment, artifacts, config_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    report_path.write_text(report_md)

    logger.info(f"Report written to {report_path}")
    return report_path
