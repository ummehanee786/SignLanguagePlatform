"""
train_and_compare.py

Task 2: Trains Random Forest, Decision Tree, and Support Vector
Machine on the SAME dataset, measures training time and evaluation
metrics for each, and writes comparison_report.csv.

Also logs the Random Forest run as an experiment via experiment_logger
(Task 1's tracking system), since RF is typically the strongest
starting baseline for tabular landmark data like this.
"""

import csv
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.ai.ml.training.data_utils import load_split, get_data_dir
from app.ai.ml.training.experiment_logger import log_experiment


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }


def train_and_time(model, X_train, y_train):
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    return model, elapsed


def main():
    print("Loading data...")
    X_train, y_train = load_split("train.csv")
    X_test, y_test = load_split("test.csv")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results = []

    # --- Random Forest ---
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf, rf_time = train_and_time(rf, X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test)
    print(f"  Done in {rf_time:.2f}s - accuracy={rf_metrics['accuracy']:.4f}")
    results.append(("Random Forest", rf_time, rf_metrics))

    # --- Decision Tree ---
    print("\nTraining Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt, dt_time = train_and_time(dt, X_train, y_train)
    dt_metrics = evaluate_model(dt, X_test, y_test)
    print(f"  Done in {dt_time:.2f}s - accuracy={dt_metrics['accuracy']:.4f}")
    results.append(("Decision Tree", dt_time, dt_metrics))

    # --- SVM ---
    print("\nTraining SVM (this one can take a while)...")
    svm = SVC(random_state=42)
    svm, svm_time = train_and_time(svm, X_train, y_train)
    svm_metrics = evaluate_model(svm, X_test, y_test)
    print(f"  Done in {svm_time:.2f}s - accuracy={svm_metrics['accuracy']:.4f}")
    results.append(("Support Vector Machine", svm_time, svm_metrics))

    # --- Write comparison_report.csv ---
    report_path = get_data_dir() / "comparison_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Training Time", "Accuracy", "Precision", "Recall", "F1 Score"])
        for name, train_time, metrics in results:
            writer.writerow([
                name, round(train_time, 4),
                round(metrics["accuracy"], 4),
                round(metrics["precision"], 4),
                round(metrics["recall"], 4),
                round(metrics["f1_score"], 4),
            ])
    print(f"\n[i] comparison_report.csv saved to: {report_path}")

    # --- Log Random Forest as experiment_001 ---
    log_experiment(
        experiment_id="experiment_001",
        dataset_version="asl_alphabet_full_v1",
        feature_version="landmarks_normalized_v1",
        model_used="RandomForestClassifier",
        parameters={"n_estimators": 100, "random_state": 42},
        metrics={**rf_metrics, "training_time_seconds": round(rf_time, 4)},
        engineer_name="T L UMME HANEE",
    )
    print("[i] experiment_001 updated with real Random Forest results.")


if __name__ == "__main__":
    main()