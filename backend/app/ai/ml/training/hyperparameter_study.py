"""
hyperparameter_study.py

Task 3: Hyperparameter Study.

A Hyperparameter is a configuration value chosen before training that
controls how the learning algorithm behaves (as opposed to a
parameter the model learns from data, like the actual tree splits).

For Random Forest, n_estimators (number of trees) is one such
hyperparameter. This script trains RF with 50, 100, and 200 trees to
observe whether more trees actually improves results, or whether
returns diminish (or even reverse) past a certain point.
"""

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from app.ai.ml.training.data_utils import load_split


def main():
    print("Loading data...")
    X_train, y_train = load_split("train.csv")
    X_test, y_test = load_split("test.csv")

    tree_counts = [50, 100, 200]
    results = []

    for n_trees in tree_counts:
        print(f"\nTraining Random Forest with {n_trees} trees...")
        model = RandomForestClassifier(n_estimators=n_trees, random_state=42)

        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

        print(f"  Training time: {elapsed:.2f}s | Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
        results.append({
            "n_estimators": n_trees,
            "training_time_seconds": round(elapsed, 4),
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
        })

    print("\n" + "=" * 55)
    print("HYPERPARAMETER STUDY SUMMARY")
    print("=" * 55)
    for r in results:
        print(f"  {r['n_estimators']:>4} trees | "
              f"time={r['training_time_seconds']:>7}s | "
              f"accuracy={r['accuracy']} | f1={r['f1_score']}")

    # --- Conclusion ---
    acc_50, acc_100, acc_200 = results[0]["accuracy"], results[1]["accuracy"], results[2]["accuracy"]
    time_50, time_100, time_200 = results[0]["training_time_seconds"], results[1]["training_time_seconds"], results[2]["training_time_seconds"]

    print("\nConclusion:")
    if acc_200 > acc_50 + 0.005:
        print(f"  More trees meaningfully helped: accuracy improved from {acc_50} (50 trees) "
              f"to {acc_200} (200 trees), though training time grew from {time_50}s to {time_200}s.")
    else:
        print(f"  More trees did NOT meaningfully help: accuracy only changed from {acc_50} (50 trees) "
              f"to {acc_200} (200 trees) - a difference of {round(acc_200 - acc_50, 4)}. "
              f"Meanwhile training time grew from {time_50}s to {time_200}s ({round(time_200/time_50, 1)}x slower) "
              f"for essentially no accuracy benefit. This suggests 50-100 trees is likely "
              f"the more efficient choice for this dataset.")


if __name__ == "__main__":
    main()