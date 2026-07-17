# Normalization Strategy

## What normalize_landmarks.py does

Each training sample has 21 hand landmarks, each with (x, y, z)
coordinates as detected by MediaPipe - 63 numbers total per sample.
Two transformations are applied to every sample:

### 1. Wrist-relative translation (position invariance)

The wrist (landmark 0) position is subtracted from every landmark in
the sample, so the wrist becomes the origin (0, 0, 0).

**Why:** without this, the model could partly "cheat" by learning
where in the camera frame a hand tends to appear for a given sign,
rather than learning the actual hand *shape*. Making every sample
relative to the wrist removes the hand's absolute screen position as
a factor entirely.

### 2. Hand-size scaling (scale invariance)

After translation, every coordinate is divided by the distance from
the wrist to the middle finger's MCP knuckle (landmark 9).

**Why:** a hand held close to the camera produces larger raw
coordinate spreads than the same hand shape held farther away. Scaling
by a stable internal reference distance (wrist-to-middle-knuckle)
means the same gesture produces very similar normalized values
regardless of distance from the camera.

## Why this specific reference point (landmark 9)

Landmark 9 (middle finger MCP) sits in the central "palm" area of the
hand skeleton and is relatively stable across different hand poses -
less likely to be at an extreme position (like a fingertip) that
varies wildly between open and closed hand shapes.

## Verification

After normalization, on the real 12,697-sample dataset:
- Landmark 0 (wrist) equals exactly `(0, 0, 0)` for every sample
- Landmark 9's distance from the origin equals `1.0` for every sample

Both confirmed correct.

## Known limitation / alternative strategies not used

This is one reasonable, simple normalization approach - not the only
valid one. Alternatives considered but not implemented:
- Scaling by overall hand bounding-box size instead of a single
  landmark distance
- Rotation normalization (aligning hand orientation) - not applied,
  since hand rotation may itself be meaningful for some signs
- Per-axis normalization instead of a single uniform scale factor

These could be revisited later if model accuracy suggests the current
approach isn't capturing gestures well enough.