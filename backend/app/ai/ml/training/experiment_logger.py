"""
experiment_logger.py

Task 1: Experiment Tracking utility.

Experiment Tracking is the systematic recording of every training run
so results can be reproduced and compared later. This module creates
(or updates) an experiments/<experiment_id>/ folder containing:

    experiment_config.json  - what was run (dataset, features, model, params)
    results.json            - what happened (metrics)
    notes.md                - human-written context/observations

Professional ML teams never train models without recording this
metadata - otherwise "why did accuracy change last month?" becomes
unanswerable.
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional


def get_experiments_root() -> Path:
    # backend/app/ai/ml/training -> ml -> ai -> app -> backend -> project root
    return Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "experiments"


def log_experiment(
    experiment_id: str,
    dataset_version: str,
    feature_version: str,
    model_used: str,
    parameters: dict,
    metrics: dict,
    engineer_name: str,
    notes: Optional[str] = None,
) -> Path:
    """
    Creates (or overwrites) an experiment folder with all required
    metadata. Returns the path to the experiment folder.
    """
    experiment_dir = get_experiments_root() / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment_id": experiment_id,
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "model_used": model_used,
        "parameters": parameters,
        "date": date.today().isoformat(),
        "engineer_name": engineer_name,
    }
    with open(experiment_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    results = {
        "experiment_id": experiment_id,
        "status": "complete",
        "metrics": metrics,
    }
    with open(experiment_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    notes_content = notes or (
        f"# {experiment_id} - Notes\n\n"
        f"Model: {model_used}\n"
        f"Dataset version: {dataset_version}\n"
        f"Feature version: {feature_version}\n\n"
        f"Metrics:\n"
        + "\n".join(f"- {k}: {v}" for k, v in metrics.items())
    )
    with open(experiment_dir / "notes.md", "w", encoding="utf-8") as f:
        f.write(notes_content)

    return experiment_dir