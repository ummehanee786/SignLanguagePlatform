# AI Inference Module

This folder is a **self-contained AI module**. The rest of the backend
should only ever need one import from it:

```python
from app.ai.ml.inference.engine import predict, predict_from_bytes
```

Everything else in here - `model_loader.py`, `feature_pipeline.py`,
`result.py` - is an internal implementation detail. You should never need
to import them directly, and you should never need to touch `cv2`,
`mediapipe`, or the trained model file yourself.

## Quick start

```python
from app.ai.ml.inference.engine import predict_from_bytes

# image_bytes = raw bytes of an uploaded image (e.g. from a FastAPI
# UploadFile, or read from disk with open(path, "rb").read())
result = predict_from_bytes(image_bytes)

if result.success:
    print(result.predicted_class, result.confidence)
else:
    print("no prediction:", result.error)
```

If you already have a decoded OpenCV/BGR frame (e.g. from your own
`cv2.imread` or a webcam capture loop) instead of raw bytes, use `predict()`
instead:

```python
from app.ai.ml.inference.engine import predict
import cv2

frame = cv2.imread("some_photo.jpg")
result = predict(frame)
```

## The `PredictionResult` object

Both functions always return a `PredictionResult` (see `result.py`) -
**never raises for expected failure cases** (no hand in frame, corrupt
image, invalid landmarks). Always check `result.success` first.

| Field | Type | Notes |
|---|---|---|
| `success` | `bool` | `False` for expected failures - always check this first |
| `predicted_class` | `str \| None` | e.g. `"A"` - `None` if `success=False` |
| `confidence` | `float \| None` | 0.0-1.0, the top class's probability |
| `above_confidence_threshold` | `bool \| None` | whether `confidence` cleared the engine's configured threshold (default 0.6) |
| `probabilities` | `dict[str, float] \| None` | full class -> probability distribution |
| `error` | `str \| None` | human-readable reason when `success=False` |
| `model_version` | `str \| None` | which model version served this prediction (e.g. `"v1"`) |
| `model_inference_time_ms` | `float \| None` | time spent in `model.predict_proba()` only |
| `total_time_ms` | `float \| None` | time spent in the whole pipeline (detection -> result) |
| `timestamp` | `str` | ISO 8601 UTC timestamp |

Call `.to_dict()` on any result to get a plain JSON-serializable dict - this
is what `gesture_service.py` uses to build the API response.

## Before first use: export a model

The engine loads a trained model from `backend/models/`. If that folder
doesn't exist yet (fresh checkout), running `predict()` will raise
`ModelNotFoundError` with instructions. Fix it once with:

```bash
cd backend
python -m app.ai.ml.training.export_model
```

This trains the current production configuration (Random Forest, 100 trees)
on `data/train.csv` / `data/test.csv` and writes:

- `models/gesture_classifier_v1.joblib` - the actual model file
- `models/registry.json` - version history + metadata (dataset version,
  feature version, parameters, metrics, date, engineer name - the same
  fields Task 1's experiment tracking requires)

Re-running `export_model.py` later (e.g. after retraining on more data)
automatically creates `v2`, `v3`, etc. without touching `v1` - nothing gets
silently overwritten, and you can pin a specific version if needed:

```python
from app.ai.ml.inference.engine import GestureRecognitionEngine
engine = GestureRecognitionEngine(model_version="v1")  # pin instead of "latest"
```

## Performance & configuration notes

- The model and MediaPipe detector are loaded **once**, lazily, on the
  first `predict()` call, and reused for every call after that (see
  `get_engine()` in `engine.py`) - don't construct your own
  `GestureRecognitionEngine()` per-request.
- Default confidence threshold is `0.6`. To change it, either pass
  `confidence_threshold=` when constructing an engine directly, or edit
  `DEFAULT_CONFIDENCE_THRESHOLD` in `engine.py`.
- See `app/ai/ml/evaluation/benchmark_report.md` for real latency/memory/
  throughput numbers measured against this same model.

## File map

```
inference/
  engine.py           <- the public interface (predict / predict_from_bytes)
  result.py           <- PredictionResult dataclass
  feature_pipeline.py <- validation + normalization (internal)
  model_loader.py     <- versioned model loading + caching (internal)
```

```
../training/
  export_model.py     <- trains + saves a new versioned model
```

```
../../handtracking/
  detector.py          <- MediaPipe wrapper; the ONLY file allowed to
                          import cv2/mediapipe anywhere in the backend
```