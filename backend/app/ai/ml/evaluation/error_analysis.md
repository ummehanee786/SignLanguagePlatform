# Error Analysis

## Top 5 Most Confused Gesture Pairs

| True Sign | Predicted As | Count | Likely Reason |
|---|---|---|---|
| N | M | 14 | Similar finger positions - M and N are the two closest handshapes in the ASL alphabet. Both are a closed fist with the thumb tucked underneath the fingers; the only difference is that M has three fingers over the thumb and N has two. That's a few millimeters of thumb visibility, which is genuinely hard to separate from 21 landmark points - not a data quality problem. |
| U | R | 9 | Similar finger positions - U and R use the same two extended fingers (index + middle). R crosses them; U keeps them together and straight. The crossing is a subtle depth/angle change that's easy to lose depending on hand rotation relative to the camera - likely compounded by mild occlusion of the crossing point at certain angles. |
| M | N | 5 | Same root cause as N->M above (this is the reverse direction of the same confusion) - confirms the M/N ambiguity is symmetric, i.e. a genuine boundary between the two classes rather than a one-off dataset artifact. |
| D | O | 4 | Similar finger positions, with a possible dataset consistency issue - D and O both use a thumb-to-finger circle shape; D adds an extended index finger while O curls it in. If the index finger was only partially extended in some captured images, it sits ambiguously between the two shapes - worth spot-checking a few D/O source images for inconsistent index-finger extension. |
| J | space | 3 | Data representation limitation, not a hand-similarity problem - J is a **dynamic** letter in ASL (traced in the air with the pinky), but this pipeline captures a single static frame per sample. Whatever moment mid-motion got photographed becomes J's "true" static shape, which varies sample to sample and can land close to a resting/space-like hand position. |

## Investigation: Why These Confusions Happen

Cross-referencing the top 5 pairs above against the five candidate causes:

- **Similar finger positions** - the dominant cause here. N/M, U/R, and D/O all differ by a small, specific detail (thumb depth, finger crossing, or degree of finger curl) rather than looking globally different. This explains 4 of the 5 pairs (23 of the 26 total confusions in the top 5).
- **Poor dataset quality** - a secondary factor for D/O specifically, where inconsistent index-finger extension across samples may blur the boundary between the two classes.
- **Occlusion** - a likely contributor to U/R: at certain hand angles, the crossing of the index and middle fingers can partially hide the exact crossing point from MediaPipe's landmark detector.
- **Incorrect labels** - no direct evidence of mislabeling found in the top 5; would need a manual sample review to fully rule out, but the fact that confusions concentrate on ASL's actually-similar letter pairs (rather than being spread randomly across unrelated letters) suggests labels are largely correct.
- **Background noise** - not a distinguishing factor for the top 5 pairs specifically, since these are landmark-coordinate features rather than raw pixels; background only matters as far as it degrades MediaPipe's landmark *detection* itself (see the note on failed extractions below).

**J/space is a distinct, separate case**: rather than a hand-shape confusion, it stems from J being a motion-based letter being represented as a single still frame. This is a data-representation limitation of the current pipeline (static landmark snapshots), not something more training data or a different classifier would necessarily fix on its own.

## Suggested follow-up (optional, strengthens this further)

`dataset_report.json` shows a 76.17% landmark-detection success rate (20,728 of 87,000 images had no hand detected). It would be worth checking whether failed-detection rates were disproportionately higher for M, N, U, and R specifically compared to the dataset average - that would tell us whether occlusion during image capture is compounding the finger-similarity problem for exactly the classes that are already hardest to separate.

## Conclusion

The dominant driver of this model's confusion is genuine ASL hand-shape similarity (N/M, U/R, D/O), not data quality or background noise. This has a concrete implication: switching classifiers or tuning hyperparameters (see Task 2/Task 3) is unlikely to resolve these specific confusions, since the ambiguity is in the *feature representation* (21 static landmarks) rather than the *model*. The most effective fixes would be (a) engineering features that better capture the specific distinguishing detail for each pair - e.g. thumb-to-finger distance for M/N, finger-crossing angle for U/R - and (b) treating J (and any other motion-based letters) as a separate case requiring multi-frame/temporal input rather than a single static snapshot.