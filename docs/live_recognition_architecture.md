# Live Recognition Architecture: From Static Gestures to Continuous Sign Language

## Purpose

Today's system recognizes one static gesture from one still image:

```
Frame -> MediaPipe -> Random Forest -> Prediction
```

This is what `app/ai/ml/inference/engine.py` implements, and it's genuinely
sufficient for the current product surface: assessing whether a learner is
correctly forming a single static alphabet letter.

The long-term goal is different in kind, not just degree: recognizing
**continuous** sign language, where meaning lives in motion across many
frames (a word like "hello" or "thank you" is a gesture *trajectory*, not a
single handshape). This document designs the architecture for that future
system - **not the sequence model itself** - and, just as importantly,
explains which pieces of today's system carry over unchanged, so this is a
real migration plan rather than a rewrite.

```
Webcam Stream
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
Temporal Buffer
  |
  v
Sequence Generator
  |
  v
Sequence Model (Future: LSTM / GRU / Transformer)
  |
  v
Gesture / Word Prediction
  |
  v
Sentence Formation
```

---

## Component-by-component

### Webcam Stream
**Responsibility:** Provides a continuous, ongoing feed of video frames from
the user's camera, rather than one discrete upload.
**Input:** Hardware camera device.
**Output:** A raw video stream (e.g. an OpenCV `VideoCapture` object, or a
WebRTC stream if this becomes browser-based).
**Why needed:** Static recognition only ever needed *one* frame per request
(today's `POST /predict` takes a single uploaded image). Continuous
recognition needs an ongoing supply of frames to observe motion over time -
this is the component that makes "continuous" possible at all.
**Relationship to today's system:** Genuinely new as a first-class
component. `webcam_smoke_test.py` (used earlier to manually test the
engine) already opens a `cv2.VideoCapture(0)` loop - that ad hoc script is
effectively a prototype of this component, just without a purpose beyond
manual testing.

### Frame Capture
**Responsibility:** Pulls individual frames off the stream at a controlled
rate and hands each one to the recognition pipeline.
**Input:** The webcam stream.
**Output:** Individual BGR frames (`numpy.ndarray`), one at a time - the
exact same shape `HandLandmarkDetector.extract_landmarks()` already expects.
**Why needed:** A raw stream can produce frames faster than the pipeline can
usefully process them. This component decides the sampling rate (e.g. every
frame at 30fps, or every 3rd frame at ~10fps) - a real, deliberate
engineering choice, not an afterthought: Task 5's benchmark already showed
this system's per-frame cost isn't free, so oversampling wastes CPU for no
recognition benefit, and undersampling risks missing fast hand motion.
**Relationship to today's system:** New as a distinct step, but low-risk -
it's a thin loop around the webcam stream feeding frames one at a time into
exactly the same per-frame code path used today.

### MediaPipe (Hand Detection)
**Responsibility:** Detects the hand in a single frame and locates its 21
landmark points.
**Input:** One BGR frame.
**Output:** Raw landmark positions for the detected hand, or nothing if no
hand is present.
**Why needed:** Unchanged from today - still the foundation everything else
is built on.
**Relationship to today's system:** **Fully reused, zero changes.** This is
`HandLandmarkDetector` in `app/ai/handtracking/detector.py`, called exactly
as it is today. The only difference is *how often* it's called: once per
uploaded image today, once per captured frame in the future. The class
itself doesn't need to know or care which case it's in - `process()` and
`extract_landmarks()` are already stateless per-call.

### Landmark Extraction
**Responsibility:** Flattens MediaPipe's detected hand into the 63-value
`[x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]` vector the rest of the
pipeline uses.
**Input:** MediaPipe's raw per-frame detection result.
**Output:** A flat list of 63 floats (or `None` if no hand was detected).
**Why needed:** Unchanged from today - a consistent, fixed-size numeric
representation of "what did the hand look like in this frame" is exactly
as necessary for a sequence model as it is for a single-frame classifier.
**Relationship to today's system:** **Fully reused.** This is
`HandLandmarkDetector.extract_landmarks()`. Feature Validation and
Normalization (`feature_pipeline.py`) are reused too, applied to each
frame's landmarks individually before the vector enters the buffer below -
this matters because it means every vector inside the Temporal Buffer is
already on the same normalized scale as the model was trained on, with no
new normalization logic to write or validate.

### Temporal Buffer
**Responsibility:** Stores the last N frames' worth of (normalized) landmark
vectors, so motion over time - not just a single instant - is available for
analysis.
**Input:** One normalized 63-dimensional feature vector per frame (from
Landmark Extraction), arriving continuously.
**Output:** A rolling window of the last N feature vectors, e.g. a
fixed-size deque of shape `(N, 63)`.
**Why needed:** No single frame contains "motion" - motion only exists
*across* frames. Without a buffer, every frame would be evaluated in total
isolation (exactly what today's system does), which is precisely why
today's system can only recognize static handshapes and not dynamic signs
like the letter "J" (already flagged as a limitation in `error_analysis.md`
- J is a motion-based letter that today's single-frame pipeline can't
represent correctly). The buffer is what turns "a sequence of independent
predictions" into "one prediction problem over a sequence."
**Design notes:** N needs tuning against real sign duration (a "held" sign
lasts roughly 0.5-2 seconds; at 10fps sampling that's ~5-20 frames). The
buffer should also handle the "no hand detected" case gracefully - dropping
frames vs. inserting a placeholder both need a defined behavior, since
gaps in a real webcam feed (hand briefly leaving frame) are inevitable, not
exceptional.
**Relationship to today's system:** Entirely new - today's engine has no
concept of state between calls; `GestureRecognitionEngine.predict()` is a
pure function of its single input frame.

### Sequence Generator
**Responsibility:** Converts the buffer's raw window of individual landmark
vectors into a well-formed input the sequence model can actually consume -
a fixed-shape tensor, not just "whatever's currently in the buffer."
**Input:** The Temporal Buffer's rolling window (variable fill level,
especially near the start of a stream or after a gap).
**Output:** A fixed-shape sequence tensor, e.g. `(sequence_length, 63)`,
padded or truncated to a consistent length, optionally augmented with
frame-to-frame delta features (velocity: how much each landmark moved since
the previous frame - often more informative for motion than raw position).
**Why needed:** Sequence models (LSTM/GRU/Transformer) generally expect
fixed-shape batched input, and a live buffer's fill level naturally varies
(what's in the buffer right after startup isn't the same length as what's
in it 10 seconds in). This component is the boundary between "live,
variable, streaming data" and "the fixed-shape input contract a trained
model expects" - conceptually the same job `feature_pipeline.py` does
today (turning variable raw input into a consistent model-ready shape),
just operating on a sequence instead of a single frame.
**Relationship to today's system:** New logic, but a direct architectural
sibling of `feature_pipeline.build_feature_vector()` - same *role* in the
pipeline (raw-to-model-ready), scaled up from one vector to a sequence of
them.

### Sequence Model (Future: LSTM / GRU / Transformer)
**Responsibility:** Learns how a gesture *evolves* across the sequence,
instead of treating every frame as an independent, unrelated classification
problem.
**Input:** The fixed-shape sequence tensor from the Sequence Generator.
**Output:** A predicted class (a word/gesture) with per-class probabilities
- the same shape of output today's Random Forest already produces via
`model.predict_proba()`.
**Why needed:** This is the actual capability upgrade. A static classifier
fundamentally cannot represent "the hand moved from position A to position
B" - it only ever sees one position at a time. A sequence model can learn
that the *trajectory* itself is the signal (this is exactly why J, a
motion-drawn letter, and full dynamic words are out of reach for today's
architecture no matter how much static training data is added).
**Why this is deliberately NOT being built yet:** Introducing a sequence
model is a substantial, separate effort - it needs a labeled *video/sequence*
dataset (not single images), a different training pipeline, and its own
experiment-tracking runs (Task 1's framework - `experiment_config.json`,
`results.json`, `notes.md` - already generalizes to this without changes,
since it doesn't assume any particular model architecture).
**Relationship to today's system:** This is the piece that changes. But
critically, **the surrounding infrastructure doesn't need to change to
support it**: `model_loader.py`'s versioning already works for "any
scikit-learn-style estimator with `.predict_proba()`" - the same registry
pattern (dataset version, feature version, model used, parameters, metrics,
date, engineer name) applies whether "model used" is `RandomForestClassifier`
or an LSTM. Swapping in a sequence model later is a new registry version
and a new `model_used` string, not a new versioning system.

### Gesture / Word Prediction
**Responsibility:** Turns the sequence model's raw output into a structured,
confidence-checked prediction - identical role to what `engine.py` already
does today after `model.predict_proba()`.
**Input:** Class probabilities from the Sequence Model.
**Output:** A structured prediction object: predicted word/gesture,
confidence, whether it cleared the confidence threshold, full probability
distribution.
**Why needed:** Exactly the same reasons as today - a raw probability array
isn't useful to the rest of the app on its own; picking the top class,
checking it against a confidence threshold, and packaging it consistently
is what makes the output usable and loggable.
**Relationship to today's system:** **Directly reusable design, not just
reusable code.** `PredictionResult` (`result.py`) already has exactly the
right shape for this - `predicted_class`, `confidence`,
`above_confidence_threshold`, `probabilities`, `model_version`,
timing fields. A sequence-based prediction is still fundamentally "a class
label plus a confidence plus metadata"; the dataclass doesn't need to
change, only what produces it does.

### Sentence Formation
**Responsibility:** Accumulates individual recognized words/gestures, over
time, into a coherent sentence - handling the fact that a continuous stream
produces many overlapping/repeated recognitions, not one clean word per
sign.
**Input:** A stream of `PredictionResult`-shaped outputs from Gesture/Word
Prediction, arriving continuously as the user keeps signing.
**Output:** An assembled sentence (or sequence of words) ready to display or
process further - e.g. `["HELLO", "MY", "NAME"]`.
**Why needed:** None of the earlier components solve the problem of *when*
one sign ends and the next begins, or what to do about a sign held for 2
seconds producing 20 near-identical high-confidence predictions in a row.
Sentence Formation is where that gets resolved: deduplicating consecutive
repeats, deciding on a pause/gap as a word boundary, filtering out
low-confidence flickers instead of letting them corrupt the sentence, and
(eventually, out of scope for this design) accounting for the fact that ASL
grammar/word order doesn't map 1:1 onto English, which a naive
word-by-word concatenation won't capture.
**Relationship to today's system:** Entirely new - today's system has no
concept of "more than one prediction over time" at all; each `/predict`
call is independent and stateless by design (appropriate for assessing one
static letter at a time).

---

## How this evolves from today's system without breaking it

This isn't a replacement architecture - it's an additive one:

- **Nothing about today's `/predict` endpoint needs to change.** Static,
  single-frame alphabet assessment remains a legitimate, permanent use case
  (that's literally what the learning platform's practice/assessment
  features need), not a stepping stone to be discarded. It would continue
  to call `engine.predict()` / `engine.predict_from_bytes()` exactly as it
  does now.
- **A new, separate interface would be added alongside it** - something
  like `engine.predict_sequence(frames)` or a dedicated
  `LiveRecognitionSession` class that owns a Temporal Buffer and calls into
  the pipeline above - without modifying the existing single-frame path.
- **Three of today's four pipeline components are reused unmodified**
  (MediaPipe detection, landmark extraction, feature validation/
  normalization) and a fourth (confidence-thresholded structured
  prediction) is reused *by design*, even though its implementation will
  eventually need to change to accommodate a different model type. The
  genuinely new work is concentrated in exactly four components: Frame
  Capture, Temporal Buffer, Sequence Generator, and Sentence Formation -
  plus, eventually, training the Sequence Model itself.

## Suggested build order

1. **Frame Capture + Temporal Buffer + Sequence Generator**, wired up with
   *no* new model yet - just prove frames flow into a correctly-shaped
   sequence tensor. This can be validated end-to-end without any new
   training data.
2. **A rule-based or heuristic placeholder** where the Sequence Model will
   go (e.g. "if the hand hasn't moved for 500ms, run today's existing
   static classifier on the last frame") - this delivers partial value
   (stability against flicker, basic motion-vs-static detection) using
   zero new ML, while the real infrastructure gets exercised.
3. **Sentence Formation**, built against that heuristic output - so its
   deduplication/boundary logic gets tested against realistic, noisy input
   before a real sequence model exists.
4. **The actual Sequence Model**, once the above is stable and a labeled
   sequence dataset exists - dropped in via `model_loader.py`'s existing
   versioning, exactly as described above.

This mirrors how Task 1 was actually built (infrastructure - export,
versioning, validation, logging - built and tested before the model was the
hard part) and keeps every stage independently testable, rather than
betting the whole effort on a sequence model working correctly on the first
try.