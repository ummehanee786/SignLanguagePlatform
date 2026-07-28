"""
model_loader.py

"Model loading with version management" for the AI inference engine.

export_model.py (app/ai/ml/training/) trains a model and writes:

    backend/models/gesture_classifier_v1.joblib
    backend/models/registry.json

This module is the ONLY place that knows about that file layout. Nothing
else in the application - not engine.py, not gesture_service.py, not any
FastAPI route - should ever construct a path into backend/models/ or parse
registry.json directly. If the storage format changes later (e.g. moving
to S3, or adding a staging/production split), only this file needs to
change.

Usage:
    from app.ai.ml.inference.model_loader import load_model

    model, metadata = load_model()                # latest version
    model, metadata = load_model(version="v1")     # pinned version

`metadata` is the exact registry entry for that version - dataset_version,
feature_version, model_used, parameters, metrics, date, engineer_name,
num_classes, classes - so callers (like engine.py) can log/report on it
without needing their own copy of that information.
"""

import json
from pathlib import Path
from typing import Optional

import joblib

_model_cache: dict[str, tuple] = {}


class ModelNotFoundError(RuntimeError):
    """Raised when no trained model / registry exists yet."""


class ModelVersionError(RuntimeError):
    """Raised when a specific requested version doesn't exist in the registry."""


def get_models_dir() -> Path:
    # inference/ -> ml -> ai -> app -> backend, then /models
    return Path(__file__).resolve().parent.parent.parent.parent.parent / "models"


def get_registry_path() -> Path:
    return get_models_dir() / "registry.json"


def _load_registry() -> dict:
    registry_path = get_registry_path()
    if not registry_path.exists():
        raise ModelNotFoundError(
            f"No model registry found at {registry_path}. "
            "Train and export a model first by running (from backend/):\n"
            "    python -m app.ai.ml.training.export_model"
        )
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_versions() -> list[str]:
    """Returns all available model versions, e.g. ['v1', 'v2']."""
    registry = _load_registry()
    return sorted(registry["versions"].keys(), key=lambda v: int(v[1:]))


def get_latest_version() -> str:
    registry = _load_registry()
    latest = registry.get("latest_version")
    if not latest:
        raise ModelNotFoundError(
            "Model registry exists but has no versions. "
            "Run: python -m app.ai.ml.training.export_model"
        )
    return latest


def load_model(version: Optional[str] = None, use_cache: bool = True):
    """
    Loads a trained model and its metadata.

    Args:
        version: A specific version string (e.g. "v1"), or None/"latest"
                  to use the most recently exported model.
        use_cache: If True (default), returns a previously-loaded model
                    instance instead of hitting disk again. Model files
                    can be tens of MB, so re-loading on every single
                    prediction call would be wasteful - see engine.py,
                    which calls this once and reuses the result.

    Returns:
        (model, metadata) where `model` is the deserialized scikit-learn
        estimator and `metadata` is its registry.json entry (dict).

    Raises:
        ModelNotFoundError: no registry.json / no exported model exists yet.
        ModelVersionError: a specific `version` was requested but isn't
                            in the registry.
    """
    registry = _load_registry()

    resolved_version = version
    if resolved_version is None or resolved_version == "latest":
        resolved_version = get_latest_version()

    if resolved_version not in registry["versions"]:
        available = sorted(registry["versions"].keys(), key=lambda v: int(v[1:]))
        raise ModelVersionError(
            f"Model version '{resolved_version}' not found. "
            f"Available versions: {available}"
        )

    if use_cache and resolved_version in _model_cache:
        return _model_cache[resolved_version]

    metadata = registry["versions"][resolved_version]
    model_path = get_models_dir() / metadata["file"]
    if not model_path.exists():
        raise ModelNotFoundError(
            f"registry.json references '{metadata['file']}' for version "
            f"'{resolved_version}', but that file is missing from "
            f"{get_models_dir()}. Was it deleted, or does registry.json "
            "belong to a different machine/checkout?"
        )

    model = joblib.load(model_path)
    result = (model, metadata)

    if use_cache:
        _model_cache[resolved_version] = result

    return result


def clear_cache() -> None:
    """Drops all cached in-memory models. Mainly useful for tests, or
    after exporting a new version you want the running process to pick
    up without a full restart."""
    _model_cache.clear()