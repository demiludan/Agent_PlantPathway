#!/usr/bin/env python3
"""
Model definitions for C3 vs C4 classification.

Contains:
- FNN (Feedforward Neural Network) for curve-based classification
- Factory for sklearn models (LR, RF, SVM, GB, AdaBoost, NB, DT)

Adapted from Classification_CO2S.py model definitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


class FNN(nn.Module):
    """Feedforward Neural Network for A/Ci curve classification."""

    def __init__(self, input_shape, layer1=128, layer2=256, layer3=128, layer4=64):
        super().__init__()
        self.lin1 = nn.Linear(input_shape[0] * input_shape[1], layer1)
        self.lin2 = nn.Linear(layer1, layer2)
        self.lin3 = nn.Linear(layer2, layer3)
        self.lin4 = nn.Linear(layer3, layer4)
        self.final = nn.Linear(layer4, 1)

    def forward(self, x):
        out = torch.flatten(x, start_dim=1)
        out = F.relu(self.lin1(out))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = torch.tanh(self.lin4(out))
        out = self.final(out)
        return out


def get_sklearn_models(params):
    """Return dictionary of configured sklearn models."""
    seed = getattr(params, "random_seed", 42)

    rf_cfg = getattr(params, "random_forest", {}) or {}
    gb_cfg = getattr(params, "gradient_boosting", {}) or {}
    ada_cfg = getattr(params, "adaboost", {}) or {}
    lr_cfg = getattr(params, "logistic_regression", {}) or {}
    dt_cfg = getattr(params, "decision_tree", {}) or {}
    svm_cfg = getattr(params, "svm", {}) or {}

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=lr_cfg.get("max_iter", 1000), random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 100),
            max_depth=rf_cfg.get("max_depth", 10),
            random_state=seed,
        ),
        "SVM": SVC(
            kernel=svm_cfg.get("kernel", "rbf"),
            probability=svm_cfg.get("probability", True),
            random_state=seed,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=gb_cfg.get("n_estimators", 100),
            max_depth=gb_cfg.get("max_depth", 5),
            random_state=seed,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=ada_cfg.get("n_estimators", 100),
            random_state=seed,
        ),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=dt_cfg.get("max_depth", 10),
            random_state=seed,
        ),
    }
