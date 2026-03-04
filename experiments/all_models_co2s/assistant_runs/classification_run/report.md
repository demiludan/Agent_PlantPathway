# C3 vs C4 Photosynthetic Pathway Classification Report

This report evaluates 8 machine learning models for classifying C3 vs C4 photosynthetic pathways using A/Ci curves.

## 1. Experiment Overview

This experiment classifies plant photosynthetic pathways (C3 vs C4) from A/Ci response curves. Each curve is resampled to uniform length, normalized, then fed to 8 ML models: Neural Network (FNN), Logistic Regression, Random Forest, SVM, Gradient Boosting, AdaBoost, Naive Bayes, and Decision Tree.

## 2. Model Comparison

### Accuracy Ranking

| Rank | Model | Accuracy | F1-Score | AUC-ROC |
|------|-------|----------|----------|---------|
| 1 | Neural Network | 100.00% | 100.00% | 1.0000 |
| 2 | Random Forest | 100.00% | 100.00% | 1.0000 |
| 3 | SVM | 100.00% | 100.00% | 1.0000 |
| 4 | Logistic Regression | 97.67% | 97.78% | 0.9978 |
| 5 | AdaBoost | 97.67% | 97.78% | 1.0000 |
| 6 | Gradient Boosting | 95.35% | 95.65% | 0.9524 |
| 7 | Decision Tree | 95.35% | 95.65% | 0.9524 |
| 8 | Naive Bayes | 83.72% | 86.27% | 0.8810 |

![Accuracy Comparison](experiments/all_models_co2s/evaluate/best_model/figures/test/03_accuracy_comparison.png)

### Performance Heatmap

![Heatmap](experiments/all_models_co2s/evaluate/best_model/figures/test/06_heatmap.png)

### Confusion Matrices

![Confusion Matrices](experiments/all_models_co2s/evaluate/best_model/figures/test/04_confusion_matrices.png)

## 3. Neural Network Analysis

![NN Training Curves](experiments/all_models_co2s/evaluate/best_model/figures/test/02_nn_training.png)

## 4. Cross-Validation Results

![Cross-Validation](experiments/all_models_co2s/evaluate/best_model/figures/test/05_cross_validation.png)

| Model | CV Accuracy | Std Dev |
|-------|-------------|---------|
| AdaBoost | 98.57% | 1.90% |
| Random Forest | 97.63% | 1.51% |
| Decision Tree | 97.63% | 3.69% |
| Gradient Boosting | 97.15% | 3.50% |
| Neural Network | 97.15% | 1.78% |
| SVM | 95.26% | 1.51% |
| Logistic Regression | 93.83% | 3.24% |
| Naive Bayes | 79.11% | 9.26% |

## 5. Discussion and Recommendations

**Best model by test accuracy**: Neural Network (100.00%)

**Most reliable (CV)**: AdaBoost (98.57% +/- 1.90%)

### Recommendations

- Consider ensemble methods combining top-performing models
- Increase dataset size with additional A/Ci curve measurements
- Experiment with different curve resampling strategies
- Test with additional x-variables (Ci, PAR) for robustness
