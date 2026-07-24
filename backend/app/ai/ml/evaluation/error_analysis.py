"""
error_analysis.py

Task 4: Error Analysis.

Generates a confusion matrix from a trained model's predictions on
the test set, identifies the top 5 most-confused gesture pairs (the
highest off-diagonal counts), and writes error_analysis.md discussing
likely causes.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from data_utils import load_split, get_data_dir


def find_top_confused_pairs(cm, labels, top_n=5):
    """
    Looks at all off-diagonal cells in the confusion matrix (i.e.
    "true label X was predicted as label Y" where X != Y) and returns
    the top_n pairs with the highest confusion counts.
    """
    confused_pairs = []
    for i, true_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            if i != j and cm[i][j] > 0:
                confused_pairs.append((true_label, predicted_label, cm[i][j]))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    return confused_pairs[:top_n]


def main():
    print("Loading data and training a Random Forest for error analysis...")
    X_train, y_train = load_split("train.csv")
    X_test, y_test = load_split("test.csv")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)

    # Save the full confusion matrix as a CSV for reference
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_path = get_data_dir() / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"[i] Confusion matrix saved to: {cm_path}")

    top_confused = find_top_confused_pairs(cm, labels, top_n=5)

    print("\nTop 5 most confused gesture pairs:")
    for true_label, predicted_label, count in top_confused:
        print(f"  True: '{true_label}' -> Predicted: '{predicted_label}'  ({count} times)")

    # --- Write error_analysis.md ---
    md_path = Path(__file__).resolve().parent / "error_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Error Analysis\n\n")
        f.write("## Top 5 Most Confused Gesture Pairs\n\n")
        f.write("| True Sign | Predicted As | Count | Likely Reason |\n")
        f.write("|---|---|---|---|\n")
        for true_label, predicted_label, count in top_confused:
            f.write(f"| {true_label} | {predicted_label} | {count} | *(fill in after reviewing signs below)* |\n")

        f.write("\n## Investigation: Why These Confusions Happen\n\n")
        f.write(
            "For each pair above, consider these possible causes:\n\n"
            "- **Similar finger positions** - do the two signs look nearly identical "
            "in hand shape, differing only in a subtle detail (e.g. thumb position)?\n"
            "- **Poor dataset quality** - are there mislabeled or blurry images for "
            "either class in the training data?\n"
            "- **Occlusion** - does one sign commonly hide key landmarks behind other "
            "fingers, making it genuinely ambiguous from a 2D landmark view?\n"
            "- **Incorrect labels** - is it possible some images were labeled wrong "
            "during dataset creation/download?\n"
            "- **Background noise** - could inconsistent backgrounds/lighting in the "
            "raw images be affecting landmark detection accuracy for these classes?\n\n"
            "**Action needed:** manually review a few misclassified examples for each "
            "pair above (cross-reference `confusion_matrix.csv` and the original "
            "dataset images) and fill in the 'Likely Reason' column with your actual "
            "findings, rather than guessing from general ASL knowledge alone.\n"
        )

    print(f"\n[i] error_analysis.md saved to: {md_path}")


if __name__ == "__main__":
    main()