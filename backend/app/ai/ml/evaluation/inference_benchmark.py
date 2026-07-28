"""
inference_benchmark.py

Task 5: Inference Benchmark.

The SRS repeatedly emphasizes real-time performance for gesture
assessment on a live learning platform - a model with excellent
accuracy but slow predictions is not suitable for a live webcam
feature. Accuracy alone (Task 2/3) says nothing about whether the
model can actually keep up with a webcam feed.

This script benchmarks the Random Forest classifier (the model
tracked in experiment_001) under realistic SINGLE-FRAME inference
conditions - one 63-feature landmark vector at a time, exactly what
the live pipeline produces per webcam frame (HandLandmarkDetector ->
normalize_landmarks -> this classifier) - not a batch-prediction
scenario.

Measures:
  - Average inference time per single prediction (ms)
  - Peak memory consumption during the inference loop (MB)
  - Model file size on disk (KB), via a joblib dump
  - CPU utilization during the benchmark loop (%) - optional, uses
    psutil if available, skipped gracefully if it isn't installed
  - Throughput (predictions per second)

Writes benchmark_report.md (this folder) and benchmark_metrics.json
(backend/data/, alongside confusion_matrix.csv from error_analysis.py).
"""

import json
import sys
import time
import tracemalloc
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from data_utils import load_split, get_data_dir  # noqa: E402

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

N_WARMUP = 50
N_TRIALS = 1000

# 30 FPS = 33.3ms/frame. Camera capture, MediaPipe landmark extraction,
# and UI rendering all have to share that budget too, so the model's
# own share of it should be well under the full frame time.
REALTIME_FRAME_BUDGET_MS = 33.3
TARGET_INFERENCE_MS = 20.0


def benchmark_single_frame_inference(model, X_test):
    """
    Times N_TRIALS individual single-row predictions (not one batch
    call) - this is what actually happens frame-by-frame in a live
    webcam pipeline, and it can behave very differently from batch
    prediction throughput.
    """
    # Keep rows as single-row DataFrames (matching what the model was
    # fit on) rather than converting to bare numpy - avoids sklearn's
    # "X does not have valid feature names" warning on every call and
    # matches how a single incoming frame would actually be shaped.
    n = len(X_test)
    single_frames = [X_test.iloc[i:i + 1] for i in range(min(N_TRIALS, n))]
    i = 0
    while len(single_frames) < N_TRIALS:
        idx = i % n
        single_frames.append(X_test.iloc[idx:idx + 1])
        i += 1

    for frame in single_frames[:N_WARMUP]:
        model.predict(frame)

    process = psutil.Process() if HAS_PSUTIL else None
    if process is not None:
        process.cpu_percent(interval=None)  # prime the counter

    tracemalloc.start()
    latencies_ms = []
    start = time.perf_counter()
    for frame in single_frames:
        t0 = time.perf_counter()
        model.predict(frame)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    total_seconds = time.perf_counter() - start
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cpu_percent = process.cpu_percent(interval=None) if process is not None else None

    latencies_ms = np.array(latencies_ms)
    return {
        "n_trials": len(single_frames),
        "avg_inference_time_ms": round(float(np.mean(latencies_ms)), 4),
        "p50_inference_time_ms": round(float(np.percentile(latencies_ms, 50)), 4),
        "p95_inference_time_ms": round(float(np.percentile(latencies_ms, 95)), 4),
        "p99_inference_time_ms": round(float(np.percentile(latencies_ms, 99)), 4),
        "peak_memory_mb": round(peak_mem_bytes / (1024 * 1024), 4),
        "cpu_utilization_percent": round(cpu_percent, 2) if cpu_percent is not None else None,
        "throughput_predictions_per_second": round(len(single_frames) / total_seconds, 1),
    }


def main():
    print("Loading data...")
    X_train, y_train = load_split("train.csv")
    X_test, y_test = load_split("test.csv")

    print("Training Random Forest (n_estimators=100, matching experiment_001)...")
    # n_jobs=1 on purpose: a single 63-feature frame has nothing to
    # parallelize across cores, so the training default of n_jobs=-1
    # would only add thread-dispatch overhead per call in a
    # one-frame-at-a-time serving scenario.
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)

    data_dir = get_data_dir()
    model_path = data_dir / "gesture_classifier_benchmark.joblib"
    joblib.dump(model, model_path)
    model_size_kb = round(model_path.stat().st_size / 1024, 2)

    print(f"Benchmarking {N_TRIALS} single-frame predictions...")
    results = benchmark_single_frame_inference(model, X_test)
    results["model"] = "RandomForestClassifier(n_estimators=100, n_jobs=1)"
    results["model_file_size_kb"] = model_size_kb
    if not HAS_PSUTIL:
        results["note"] = "psutil not installed - CPU utilization skipped (optional per task spec)."

    print(json.dumps(results, indent=2))

    metrics_path = data_dir / "benchmark_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[i] benchmark_metrics.json saved to: {metrics_path}")

    suitable = results["p95_inference_time_ms"] < TARGET_INFERENCE_MS
    write_report(results, suitable)


def write_report(r, suitable):
    verdict = "SUITABLE" if suitable else "BORDERLINE / NOT SUITABLE AS-IS"
    cpu_line = (
        f"| CPU utilization during benchmark | {r['cpu_utilization_percent']:.1f}% |\n"
        if r.get("cpu_utilization_percent") is not None
        else "| CPU utilization during benchmark | *(psutil not installed - skipped)* |\n"
    )

    content = f"""# Inference Benchmark Report

## Context
The SRS repeatedly emphasizes real-time performance for gesture assessment
on a live learning platform. Accuracy alone (see `train_and_compare.py` /
`comparison_report.csv` and `hyperparameter_study.py`) says nothing about
whether a model can keep up with a live webcam feed - a model that's 99%
accurate but takes 200ms per prediction will feel laggy and unusable. This
benchmark measures the Random Forest classifier (the model tracked as
`experiment_001`, 100 trees) under realistic **single-frame** inference
conditions: one 63-feature landmark vector at a time, exactly what the live
pipeline (`HandLandmarkDetector` -> `normalize_landmarks` -> classifier)
produces per webcam frame - not a batch-processing scenario.

## Method
- Model: `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)`,
  trained on `train.csv`, evaluated on `test.csv` (same split used
  throughout Tasks 2-4).
- `n_jobs=1` is used deliberately: a single incoming frame has nothing to
  parallelize across CPU cores, so `n_jobs=-1` (the training default) would
  only add repeated thread-dispatch overhead per call when serving one
  frame at a time.
- {r['n_trials']} single-row predictions were timed individually (not one
  batch `.predict()` call) after a {N_WARMUP}-prediction warmup, to avoid
  measuring one-time cache/JIT effects.
- Peak memory measured with Python's `tracemalloc` across the full
  benchmark loop.
- CPU utilization measured with `psutil.Process.cpu_percent()` over the
  same loop (optional - skipped automatically if `psutil` isn't installed).
- Model file size measured from a serialized `joblib` dump on disk - this
  is what would actually need to be loaded/shipped for serving.

## Results

| Metric | Value |
|---|---|
| Average inference time / prediction | {r['avg_inference_time_ms']:.3f} ms |
| p50 (median) inference time | {r['p50_inference_time_ms']:.3f} ms |
| p95 inference time | {r['p95_inference_time_ms']:.3f} ms |
| p99 inference time | {r['p99_inference_time_ms']:.3f} ms |
| Peak memory during inference loop | {r['peak_memory_mb']:.3f} MB |
| Model file size (on disk) | {r['model_file_size_kb']:.1f} KB |
{cpu_line}| Throughput | {r['throughput_predictions_per_second']:.1f} predictions/sec |

*(Absolute millisecond values depend on the machine this was run on -
re-run this script on the actual target deployment hardware, e.g. the
machine that will serve the FastAPI app, before treating these numbers as
final sign-off figures.)*

## Is this model suitable for real-time webcam-based sign recognition?

**Verdict: {verdict}**

At a {REALTIME_FRAME_BUDGET_MS:.1f} ms frame budget (30 FPS), the model's
own p95 latency of {r['p95_inference_time_ms']:.2f} ms {"leaves" if suitable else "does not leave much"}
headroom for camera capture, MediaPipe landmark extraction, and UI
rendering to share the same frame - all of which also cost time and happen
on every frame in addition to this classifier call. Throughput of
{r['throughput_predictions_per_second']:.0f} predictions/second is what a
30-60 FPS webcam stream would need if inference were the only bottleneck,
which in practice it usually isn't - MediaPipe hand detection is typically
the slower stage of the pipeline. The {r['model_file_size_kb']:.0f} KB model
file is small enough to load once at process startup (as `GestureService`
is already structured to do) with no meaningful startup-time cost, and peak
memory during inference is negligible.

### Why this benchmark matters on top of Tasks 2-4
Task 2 found this isn't even the most *accurate* model (SVM edged it out on
the comparison report), and Task 3 found more trees barely help accuracy.
Neither of those results says anything about whether the model can actually
keep up with a live webcam - that gap is exactly what this benchmark closes.
Picking a model on accuracy alone risks shipping something too slow for a
"real-time" feature; conversely, if this Random Forest turns out fast
enough here, that's a genuine point in its favor for this specific product,
even if a different classifier scores slightly higher on paper.

### Caveats worth flagging to the team
- These numbers were measured on pre-extracted, already-normalized landmark
  vectors - not through an actual webcam + MediaPipe + normalization
  pipeline end-to-end. Total perceived latency in the real app will be
  higher once capture, hand-landmark detection, and normalization are
  included; those stages should be benchmarked separately and added to this
  number for an honest end-to-end estimate.
- `n_estimators` matters here: Task 3 showed accuracy gains beyond ~50-100
  trees are marginal, so keeping the tree count modest is a "free" way to
  protect this real-time budget without giving up meaningful accuracy.
- Re-run this benchmark under concurrent load (multiple users/streams) and
  on the actual deployment/client hardware before launch - a single
  quiet-machine benchmark like this one is a first check, not a final
  sign-off.
- For serving, prefer `n_jobs=1` per prediction (as benchmarked here) and
  scale concurrent users with multiple worker processes rather than
  per-request thread parallelism.
"""
    report_path = Path(__file__).resolve().parent / "benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[i] benchmark_report.md saved to: {report_path}")


if __name__ == "__main__":
    main()