# Live Recognition Architecture - Phase 1 Design

## Current system

```
Frame -> MediaPipe -> Random Forest -> Prediction
```

One gesture, one frame, no memory between calls. This is what
`app/ai/ml/inference/engine.py` implements today, and it's correct for the
current product need: assessing one static alphabet letter at a time.

## Target system

```
Webcam
  |
  v
Frame Capture
  |
  v
MediaPipe
  |
  v
Landmark Extraction
  |
  v
Frame Buffer
  |
  v
Sequence Builder
  |
  v
(Current) Random Forest  or  (Future) LSTM / GRU / Transformer
  |
  v
Gesture Prediction
```

The critical design property this diagram is built around: **the classifier
at the bottom is swappable, but everything above it isn't disposable
scaffolding - it's permanent infrastructure that both the current classifier
and any future sequence model share.**

---

## Component documentation

### Webcam
**Purpose:** Continuous video input from the user's camera, replacing the
single static image upload used today.
**Input:** Camera hardware.
**Output:** A raw, ongoing video stream.
**Why required:** Continuous recognition needs a continuous supply of
frames. A single upload (today's model) structurally cannot represent
motion - there's nothing to buffer or sequence from one image.

### Frame Capture
**Purpose:** Reads individual frames off the stream at a controlled rate
and feeds them into the pipeline one at a time.
**Input:** The webcam stream.
**Output:** Individual BGR frames - the same shape/format
`HandLandmarkDetector` already consumes today.
**Why required:** A raw stream can produce frames faster than the pipeline
should process. This is where sampling rate is decided (e.g. every frame,
or every 2nd/3rd frame) - a real tradeoff, since Sprint Task 1's benchmark
already showed per-frame inference isn't free, and oversampling wastes CPU
for no recognition benefit.

### MediaPipe
**Purpose:** Detects the hand(s) in one frame and locates the 21 landmark
points.
**Input:** One BGR frame.
**Output:** Raw landmark positions, or nothing if no hand is present (or,
per Sprint Task 1's new multi-hand handling, a count of how many hands were
found).
**Why required:** Unchanged - the same detection step used today.
**Reuse:** 100% reused, zero changes. This is
`HandLandmarkDetector.extract_landmarks_with_metadata()`, the exact method
built in Sprint Task 1 to support multi-hand rejection - it already reports
hand count per frame, which the Frame Buffer below needs anyway to decide
whether a frame is usable.

### Landmark Extraction
**Purpose:** Converts MediaPipe's detection into the fixed 63-value
`[x,y,z] x 21` vector, then validates and normalizes it exactly as done
during training.
**Input:** One frame's raw hand detection.
**Output:** One normalized 63-dimensional feature vector, or nothing if the
frame is unusable (no hand, multiple hands, invalid landmarks).
**Why required:** Unchanged - a sequence model needs the same
consistently-shaped, consistently-normalized input a static classifier
does; there's no reason to invent a different feature representation for
sequences.
**Reuse:** 100% reused. This is `feature_pipeline.validate_landmarks()` and
`feature_pipeline.build_feature_vector()`, called once per frame exactly as
`engine.py` calls them once per request today.

### Frame Buffer
**Purpose:** Holds the last N normalized feature vectors (e.g. 20-30) in a
rolling window, so a fixed-length slice of recent motion is always
available.
**Input:** One normalized 63-dim feature vector per usable frame, arriving
continuously.
**Output:** The last N vectors, in order - the raw material the Sequence
Builder below assembles into a model-ready tensor.
**Why required:** No single frame contains motion - motion only exists
*across* frames. Without this, every frame would still be evaluated in
total isolation, which is exactly why today's system can't represent
dynamic signs (the letter "J", for instance, is drawn as a motion, and a
single freeze-frame of it is genuinely ambiguous - already noted as a
limitation in `error_analysis.md`).
**New component - see Phase 2** for a working implementation (`FrameBuffer`
in `app/ai/ml/sequence/frame_buffer.py`).

### Sequence Builder
**Purpose:** Converts the Frame Buffer's contents into a fixed-shape tensor
a model can actually be called with, handling the case where the buffer
isn't full yet (e.g. right after starting a session).
**Input:** The Frame Buffer's current contents (0 to N vectors).
**Output:** A fixed-shape `(N, 63)` tensor - concretely:
```
Frame 1  -> 63 features
Frame 2  -> 63 features
Frame 3  -> 63 features
...
Frame 20 -> 63 features
   |
   v
Sequence Tensor (20 x 63)
```
**Why required:** Sequence models expect fixed-shape input; a live buffer's
fill level varies constantly (empty at startup, partially filled after a
gap where no hand was detected, full during steady signing). This is the
boundary between "variable, streaming reality" and "the fixed contract a
model needs" - the same *role* `feature_pipeline.build_feature_vector()`
plays for a single frame, scaled up to a sequence.
**New component - see Phase 2.**

### (Current) Random Forest / (Future) LSTM / GRU / Transformer
**Purpose:** Produces a prediction from the input. This is the only box in
the entire diagram that's expected to change.
**Input:**
  - Random Forest (today): one 63-dim vector.
  - Sequence model (future): the `(N, 63)` sequence tensor.
**Output:** Predicted class + per-class probabilities, either way.
**Why it's drawn as an "or":** This is the whole point of the redesign.
Everything above this box - capture, detection, extraction, buffering,
sequencing - doesn't know or care which classifier is plugged in below it.
Swapping Random Forest for an LSTM later is a **model change**, not an
**architecture change**. This is also why Sprint Task 1's
`model_loader.py` versioning (dataset version, feature version, model used,
parameters, metrics, date, engineer name) already generalizes to this
without modification - `model_used` becomes `"LSTMClassifier"` instead of
`"RandomForestClassifier"`, nothing else about the registry needs to change.

### Gesture Prediction
**Purpose:** Turns the classifier's raw output into a structured,
confidence-checked result - identical role to what `engine.py` already does
after `model.predict_proba()` today.
**Input:** Class probabilities from whichever classifier is in use.
**Output:** Predicted gesture, confidence, whether it cleared the
confidence threshold, full probability distribution.
**Why required:** Same reasons as today - raw probabilities aren't directly
usable; picking the top class and packaging it consistently is what makes
the output usable and loggable.
**Reuse:** `PredictionResult` (`result.py`) already has the right shape for
this and needs no changes - a sequence-based prediction is still
fundamentally "a class label plus a confidence plus metadata."

---

## What this buys us

Of the seven new/changed boxes in this diagram, only **two are genuinely
new infrastructure** (Frame Buffer, Sequence Builder) and **one is expected
to eventually change** (the classifier itself). Everything else - Webcam
capture aside, which is a thin loop - is Sprint Task 1's already-built,
already-tested code, reused unmodified. Phase 2 below builds and tests the
two new pieces now, without waiting for a sequence model to exist.