# Inference Benchmark Report

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
- 1000 single-row predictions were timed individually (not one
  batch `.predict()` call) after a 50-prediction warmup, to avoid
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
| Average inference time / prediction | 37.889 ms |
| p50 (median) inference time | 36.666 ms |
| p95 inference time | 42.612 ms |
| p99 inference time | 68.499 ms |
| Peak memory during inference loop | 31.145 MB |
| Model file size (on disk) | 69439.9 KB |
| CPU utilization during benchmark | 99.6% |
| Throughput | 26.4 predictions/sec |

*(Absolute millisecond values depend on the machine this was run on -
re-run this script on the actual target deployment hardware, e.g. the
machine that will serve the FastAPI app, before treating these numbers as
final sign-off figures.)*

## Is this model suitable for real-time webcam-based sign recognition?

**Verdict: BORDERLINE / NOT SUITABLE AS-IS**

At a 33.3 ms frame budget (30 FPS), the model's
own p95 latency of 42.61 ms does not leave much
headroom for camera capture, MediaPipe landmark extraction, and UI
rendering to share the same frame - all of which also cost time and happen
on every frame in addition to this classifier call. Throughput of
26 predictions/second is what a
30-60 FPS webcam stream would need if inference were the only bottleneck,
which in practice it usually isn't - MediaPipe hand detection is typically
the slower stage of the pipeline. The 69440 KB model
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
