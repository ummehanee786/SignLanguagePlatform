"""
export_model.py

Every other training script (train_and_compare.py, hyperparameter_study.py,
error_analysis.py) trains a model in-memory and then throws it away - useful
for comparison and analysis, but none of them leave behind an actual file
the rest of the application could load and serve predictions from.

This script is the missing step between "we trained a good model" and
"a FastAPI route can load and use that model": it trains the chosen
production configuration (Random Forest, matching experiment_001) and
saves it as a VERSIONED artifact:

    backend/models/gesture_classifier_v1.joblib
    backend/models/registry.json

registry.json is what gives us "model loading with version management" -
it records, per version, exactly what Task 1's experiment tracking already
requires (dataset version, feature version, model used, parameters,
metrics, date, engineer name), plus the class list the model was trained
on. app/ai/ml/inference/model_loader.py reads this file so the inference
engine can load "latest" (or a specific pinned version) without ever
needing to know how the model was produced.

Run with:
    python -m app.ai.ml.training.export_model
"""

import json
import time
from datetime import date
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.ai.ml.training.data_utils import load_split, get_data_dir

DATASET_VERSION = "asl_alphabet_full_v1"
FEATURE_VERSION = "landmarks_normalized_v1"
MODEL_USED = "RandomForestClassifier"
PARAMETERS = {"n_estimators": 100, "random_state": 42}
ENGINEER_NAME = "T L UMME HANEE"


def get_models_dir() -> Path:
    # training/ -> ml -> ai -> app -> backend, then /models (sibling of /data)
    models_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_registry_path() -> Path:
    return get_models_dir() / "registry.json"


def load_registry() -> dict:
    registry_path = get_registry_path()
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"latest_version": None, "versions": {}}


def next_version_id(registry: dict) -> str:
    existing = [
        int(v[1:]) for v in registry["versions"].keys()
        if v.startswith("v") and v[1:].isdigit()
    ]
    next_number = (max(existing) + 1) if existing else 1
    return f"v{next_number}"


def main():
    print("Loading data...")
    X_train, y_train = load_split("train.csv")
    X_test, y_test = load_split("test.csv")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print(f"\nTraining {MODEL_USED} with parameters={PARAMETERS} ...")
    model = RandomForestClassifier(**PARAMETERS)
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, average="weighted", zero_division=0), 4),
        "f1_score": round(f1_score(y_test, predictions, average="weighted", zero_division=0), 4),
        "training_time_seconds": round(training_time, 4),
    }
    print(f"  Done in {training_time:.2f}s - accuracy={metrics['accuracy']:.4f}")

    registry = load_registry()
    version = next_version_id(registry)
    model_filename = f"gesture_classifier_{version}.joblib"
    model_path = get_models_dir() / model_filename

    joblib.dump(model, model_path)
    print(f"\n[i] Model saved to: {model_path}")

    classes = sorted(model.classes_.tolist())

    registry["versions"][version] = {
        "version": version,
        "file": model_filename,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_used": MODEL_USED,
        "parameters": PARAMETERS,
        "metrics": metrics,
        "date": date.today().isoformat(),
        "engineer_name": ENGINEER_NAME,
        "num_classes": len(classes),
        "classes": classes,
    }
    registry["latest_version"] = version

    with open(get_registry_path(), "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"[i] registry.json updated. Latest version is now: {version}")
    print("\nAny of the following will now work from app/ai/ml/inference/model_loader.py:")
    print("    load_model()            # loads 'latest' -> currently", version)
    print(f"    load_model(version='{version}')")


if __name__ == "__main__":
    main()