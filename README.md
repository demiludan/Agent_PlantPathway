# Agent_PlantPathway

AI-agent-driven ML pipeline for **C3 vs C4 photosynthetic pathway classification** using A/Ci curves. 

## What it does

Classifies plant photosynthetic pathways (C3 vs C4) from **AnetCO2 vs CO2S** response curves using **8 ML models** simultaneously:

1. Neural Network (FNN)
2. Logistic Regression
3. Random Forest
4. SVM
5. Gradient Boosting
6. AdaBoost
7. Naive Bayes
8. Decision Tree

## Architecture

```
User Prompt ("Classify C3 vs C4 using CO2S curves")
    |
main.py (CLI)
    |
PathwayCrew.run()
    |-- Parse with LLM agents (experiment/config extraction)
    |-- Validate against allowed experiments
    |-- Resolve skip flags
    +-- Execute pipeline_tool
        |-- Generate config overlay
        |-- PipelineRunner.run_full_pipeline()
        |   |-- data_preprocessing.py   (load parquet, resample curves, split, normalize)
        |   |-- train.py                (train FNN + 7 sklearn models)
        |   |-- inference.py            (generate predictions + metrics for all models)
        |   +-- evaluate.py             (cross-validation, figures, report)
        |-- load_evaluation_artifacts()
        +-- final_report_writer() (Markdown report)
    |
Output: Run dir, config, report
    |
Web UI (optional)
    |-- FastAPI backend (SSE log streaming)
    +-- React frontend (report rendering)
```

## Quick Start

### 1. Setup Environment

```bash
conda env create -f environment_cpu.yml
conda activate plant-pathway-agents-cpu
```

### 2. Prepare Data

Place your parquet files in `data/`:
```
data/
  StandardDataForAI_measurements.parquet   # A/Ci curve data with curve_id, AnetCO2, CO2S, Photosynthetic_pathway
  StandardDataforAI_metadata.parquet       # Optional metadata
```

### 3. CLI Usage

```bash
# Interactive mode
python main.py

# Inline prompt (runs all 8 models on CO2S data)
python main.py --prompt "Classify C3 vs C4 using CO2S curves"

# With specific experiment
python main.py --experiment all_models_co2s

# Dry run (preview commands)
python main.py --dry-run --prompt "Classify C3 vs C4"

# Skip stages
python main.py --prompt "Classify C3 vs C4" --skip-preprocess
python main.py --prompt "Skip training, just evaluate"
```

### 4. Web UI

```bash
# Termianl 1: start backend
uvicorn webapp.api:app --reload --port 8000

# Terminal 2: Start frontend (in separate terminal)
cd frontend && npm install && npm run dev

Open browser: http://localhost:5173

Type in the prompt box: Classify C3 vs C4 using CO2S curves
and click Send.

```




## Supported Experiments

| Experiment | Key | Description |
|------------|-----|-------------|
| All models (CO2S) | `all_models_co2s` | All 8 models on AnetCO2 vs CO2S curves |
| All models (Ci) | `all_models_ci` | All 8 models on AnetCO2 vs Ci curves |
| FNN only | `fnn_only_co2s` | Neural Network only, more epochs |
| Random Forest only | `random_forest_co2s` | Random Forest with tuned hyperparams |

## Pipeline Stages

### Stage 1: Preprocessing (`data_preprocessing.py`)
- Load `StandardDataForAI_measurements.parquet`
- Extract columns: `curve_id`, `AnetCO2`, `CO2S`, `Photosynthetic_pathway`
- Resample all curves to uniform length
- Train/test split (80/20, stratified)
- Normalize using training set statistics
- Generate A/Ci curves overview figure

### Stage 2: Training (`train.py`)
- Train FNN (feedforward neural network) with BCEWithLogitsLoss
- Train 7 sklearn models (LR, RF, SVM, GB, AdaBoost, NB, DT)
- Save all model checkpoints

### Stage 3: Inference (`inference.py`)
- Load trained models
- Generate predictions + probabilities on test/train sets
- Compute per-model metrics (accuracy, precision, recall, F1, AUC, confusion matrix)

### Stage 4: Evaluation (`evaluate.py`)
- 5-fold cross-validation for all models (including manual NN CV)
- Generate 6 figures: data overview, NN training, accuracy comparison, confusion matrices, CV results, performance heatmap
- Save CSV results and text report

## Output Structure

```
experiments/{experiment}/
    {experiment}_preprocessed.pkl    # Preprocessed data
    nn_training_history.pkl          # FNN training curves
    checkpoints/                     # Model weights (.pth, .joblib)
    inference/best_model/            # Predictions
    evaluate/best_model/
        metrics/                     # JSON metrics
        figures/test/                # All 6 figures
        reports/                     # Text summary
    assistant_runs/{timestamp}/
        config.generated.yaml
        report.md
        logs/
```

## Configuration

`config.yaml` uses a three-tier YAML hierarchy:
- **base_experiment**: Common parameters (data files, model list, FNN architecture, sklearn hyperparams)
- **Data experiments** (`co2s_c3c4`, `ci_c3c4`): Which x-variable to use
- **Model experiments** (`all_models_co2s`, etc.): Which models to train, hyperparameter overrides

## Mapping from Classification_CO2S.py

| Original function | Agent_PlantPathway equivalent |
|---|---|
| `load_data()` | `data_preprocessing.py::load_data()` |
| `prepare_data()` | `data_preprocessing.py::prepare_data()` |
| `resample_curves()` | `data_preprocessing.py::resample_curves()` |
| `train_all_models()` | `train.py::train_all_models()` |
| Metrics computation | `inference.py::_compute_metrics()` |
| `cross_validate()` | `evaluate.py::cross_validate()` |
| `create_figures()` | `evaluate.py::create_figures()` |
| `save_report()` | `evaluate.py::save_report()` + `agents/report_writer.py` |

## Technology Stack

| Category | Technologies |
|----------|-------------|
| ML/DL | PyTorch, scikit-learn, NumPy, Pandas |
| Orchestration | CrewAI (LLM agents), Pydantic |
| Backend API | FastAPI, Uvicorn, asyncio |
| Frontend | React 18, Vite, React-Markdown |
| Visualization | Matplotlib, Seaborn |
| Data | Parquet (via pyarrow/pandas) |
| Environment | Conda (CPU) |
