from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Valid experiment names (data experiments -> model experiments)
# co2s_c3c4 -> all_models_co2s, fnn_only_co2s, random_forest_co2s
# ci_c3c4   -> all_models_ci
VALID_EXPERIMENTS = [
    "all_models_co2s", "all_models_ci",
    "fnn_only_co2s", "random_forest_co2s",
]

VALID_X_VARIABLES = ["CO2S", "Ci"]

# Mapping from model experiment -> data experiment
DATA_EXPERIMENT_MAP = {
    "all_models_co2s": "co2s_c3c4",
    "fnn_only_co2s": "co2s_c3c4",
    "random_forest_co2s": "co2s_c3c4",
    "all_models_ci": "ci_c3c4",
}


@dataclass
class ParsedRequest:
    experiment: Optional[str] = None
    x_variable: str = "CO2S"
    errors: List[str] = field(default_factory=list)

    @property
    def data_experiment(self) -> Optional[str]:
        if not self.experiment:
            return None
        return DATA_EXPERIMENT_MAP.get(self.experiment, self.experiment)

    @property
    def model_experiment(self) -> Optional[str]:
        return self.experiment


def parse_experiment(prompt: str, explicit_experiment: Optional[str]) -> (Optional[str], List[str]):
    """Extract experiment from prompt or CLI override."""
    errors: List[str] = []
    if explicit_experiment:
        experiment = explicit_experiment
    else:
        prompt_lower = (prompt or "").lower()
        experiment = None
        # Try to match experiment names
        for exp in VALID_EXPERIMENTS:
            if exp.lower() in prompt_lower:
                experiment = exp
                break
        # Infer from keywords
        if not experiment:
            if "ci" in prompt_lower and "co2" not in prompt_lower:
                experiment = "all_models_ci"
            elif any(w in prompt_lower for w in ["co2s", "co2", "c3", "c4", "classify", "classification"]):
                experiment = "all_models_co2s"

    if experiment and experiment not in VALID_EXPERIMENTS:
        errors.append(f"Invalid experiment '{experiment}'. Valid: {', '.join(VALID_EXPERIMENTS)}.")
        experiment = None
    if not experiment:
        errors.append("No valid experiment found in prompt.")
    return experiment, errors


def parse_request(
    prompt: str,
    experiment: Optional[str] = None,
) -> ParsedRequest:
    """Parse a natural language prompt into a structured request."""
    exp_name, errors = parse_experiment(prompt, experiment)

    # Determine x_variable
    x_variable = "CO2S"
    if exp_name and "ci" in exp_name.lower() and "co2" not in exp_name.lower():
        x_variable = "Ci"
    elif "ci curve" in (prompt or "").lower():
        x_variable = "Ci"

    return ParsedRequest(
        experiment=exp_name,
        x_variable=x_variable,
        errors=errors,
    )
