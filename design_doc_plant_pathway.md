# Design Document: AI Agents for C3 vs C4 Photosynthetic Pathway Classification

## 1. Goal

Build an AI-agent-driven ML pipeline that classifies C3 vs C4 photosynthetic pathways
from A/Ci response curves (AnetCO2 vs CO2S), using natural-language prompts for
orchestration and 8 ML models for comparison.

## 2. Architecture

This project mirrors the Agent_GW groundwater forecasting system, adapted for classification:

| Agent_GW Component | Agent_PlantPathway Equivalent |
|---|---|
| QuantileLSTM (regression) | FNN + 7 sklearn models (classification) |
| Well selection (Son0551...) | Experiment selection (all_models_co2s...) |
| Year-based temporal splits | Stratified 80/20 train/test split |
| Quantile loss | BCEWithLogitsLoss (FNN) + sklearn losses |
| WSE prediction | C3/C4 binary classification |
| Hydrograph figures | Confusion matrices, accuracy bars, heatmap |

## 3. Data Source

- **Input**: `StandardDataForAI_measurements.parquet`
- **Columns**: `curve_id`, `AnetCO2`, `CO2S`, `Photosynthetic_pathway` (C3/C4)
- **Processing**: Resample all A/Ci curves to uniform length, normalize

## 4. Pipeline Stages

1. **Preprocessing** (`data_preprocessing.py`): Load parquet, extract curves, resample, split, normalize
2. **Training** (`train.py`): Train FNN (PyTorch) + 7 sklearn models, save checkpoints
3. **Inference** (`inference.py`): Generate predictions/probabilities, compute per-model metrics
4. **Evaluation** (`evaluate.py`): Cross-validation, 6 figures, CSV results, text report
5. **Report** (`agents/report_writer.py`): LLM-generated Markdown with embedded figures

## 5. Valid Experiments

| Experiment | Data Source | Models |
|---|---|---|
| `all_models_co2s` | `co2s_c3c4` | All 8 models |
| `all_models_ci` | `ci_c3c4` | All 8 models |
| `fnn_only_co2s` | `co2s_c3c4` | FNN only (200 epochs) |
| `random_forest_co2s` | `co2s_c3c4` | Random Forest only (tuned) |

## 6. Models

1. **Neural Network (FNN)**: 4-layer feedforward, BCEWithLogitsLoss, Adam optimizer
2. **Logistic Regression**: max_iter=1000
3. **Random Forest**: 100 estimators, max_depth=10
4. **SVM**: RBF kernel, probability=True
5. **Gradient Boosting**: 100 estimators, max_depth=5
6. **AdaBoost**: 100 estimators, SAMME algorithm
7. **Naive Bayes**: GaussianNB
8. **Decision Tree**: max_depth=10

## 7. Origin

Refactored from the monolithic `Classification_CO2S.py` into a 4-stage agent-orchestrated pipeline.
