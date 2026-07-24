# Error Analysis

## Top 5 Most Confused Gesture Pairs

| True Sign | Predicted As | Count | Likely Reason |
|---|---|---|---|
| N | M | 14 | *(fill in after reviewing signs below)* |
| U | R | 9 | *(fill in after reviewing signs below)* |
| M | N | 5 | *(fill in after reviewing signs below)* |
| D | O | 4 | *(fill in after reviewing signs below)* |
| J | space | 3 | *(fill in after reviewing signs below)* |

## Investigation: Why These Confusions Happen

For each pair above, consider these possible causes:

- **Similar finger positions** - do the two signs look nearly identical in hand shape, differing only in a subtle detail (e.g. thumb position)?
- **Poor dataset quality** - are there mislabeled or blurry images for either class in the training data?
- **Occlusion** - does one sign commonly hide key landmarks behind other fingers, making it genuinely ambiguous from a 2D landmark view?
- **Incorrect labels** - is it possible some images were labeled wrong during dataset creation/download?
- **Background noise** - could inconsistent backgrounds/lighting in the raw images be affecting landmark detection accuracy for these classes?

**Action needed:** manually review a few misclassified examples for each pair above (cross-reference `confusion_matrix.csv` and the original dataset images) and fill in the 'Likely Reason' column with your actual findings, rather than guessing from general ASL knowledge alone.
