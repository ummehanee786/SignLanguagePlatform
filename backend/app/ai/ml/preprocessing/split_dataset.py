"""
split_dataset.py

Task 2: Splits normalized_landmarks.csv into train.csv, validation.csv,
and test.csv using a STRATIFIED split - every gesture class is
proportionally represented in each split.

Task 3: Also generates training_report.json with overall dataset
statistics, since split sizes are needed for that report anyway.

Default split ratios: 70% train / 15% validation / 15% test.

Classes with fewer than MIN_SAMPLES_PER_CLASS samples are automatically
excluded before splitting, since they're too small to meaningfully
divide into train/validation/test (e.g. a single false-positive hand
detection on a background/"nothing" image out of thousands).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

MIN_SAMPLES_PER_CLASS = 3


def get_feature_columns() -> list[str]:
    return [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def stratified_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
):
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, \
        "Split fractions must add up to 1.0"

    train_df, temp_df = train_test_split(
        df, train_size=train_frac, stratify=df["label"], random_state=random_state,
    )

    remaining_frac = val_frac + test_frac
    val_relative_frac = val_frac / remaining_frac

    val_df, test_df = train_test_split(
        temp_df, train_size=val_relative_frac, stratify=temp_df["label"],
        random_state=random_state,
    )

    return train_df, val_df, test_df


def class_distribution(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().to_dict()
    return {str(k): int(v) for k, v in sorted(counts.items())}


def main():
    parser = argparse.ArgumentParser(description="Stratified train/validation/test split.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset-report", type=str, default=None,
                         help="Path to an existing dataset_report.json for failed-extraction count (optional)")
    parser.add_argument("--training-report", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    backend_root = script_dir.parent.parent.parent.parent

    input_csv = Path(args.input) if args.input else backend_root / "data" / "normalized_landmarks.csv"
    output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent
    training_report_path = (
        Path(args.training_report) if args.training_report
        else output_dir / "training_report.json"
    )

    if not input_csv.exists():
        print(f"[!] Input file not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    feature_cols = get_feature_columns()

    print(f"Loaded {len(df)} samples across {df['label'].nunique()} classes.")

    # Drop any class with too few samples to be meaningfully split into
    # train/validation/test.
    class_counts = df["label"].value_counts()
    excluded_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS].index.tolist()

    if excluded_classes:
        print(f"\n[!] Excluding classes with fewer than {MIN_SAMPLES_PER_CLASS} samples "
              f"(too few to split): {excluded_classes}")
        for cls in excluded_classes:
            print(f"    {cls}: only {class_counts[cls]} sample(s) - likely a rare "
                  f"false-positive detection, not enough data to train/validate/test on.")
        df = df[~df["label"].isin(excluded_classes)]

    train_df, val_df, test_df = stratified_split(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "validation.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSplit sizes: train={len(train_df)}  validation={len(val_df)}  test={len(test_df)}")

    print("\nClass distribution (train / validation / test):")
    train_dist = class_distribution(train_df)
    val_dist = class_distribution(val_df)
    test_dist = class_distribution(test_df)

    for label in sorted(df["label"].unique()):
        t = train_dist.get(str(label), 0)
        v = val_dist.get(str(label), 0)
        te = test_dist.get(str(label), 0)
        print(f"  {label:<10} train={t:<5} val={v:<5} test={te:<5}")

    failed_count = 0
    if args.dataset_report:
        report_path = Path(args.dataset_report)
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                existing_report = json.load(f)
            failed_count = (
                existing_report.get("no_hand_detected", 0)
                + existing_report.get("unreadable_or_corrupted", 0)
            )

    training_report = {
        "total_samples": len(df),
        "excluded_classes": excluded_classes,
        "num_classes": int(df["label"].nunique()),
        "samples_per_class": class_distribution(df),
        "num_features": len(feature_cols),
        "train_set_size": len(train_df),
        "validation_set_size": len(val_df),
        "test_set_size": len(test_df),
        "failed_landmark_extraction_count": failed_count,
        "class_distribution": {
            "train": train_dist,
            "validation": val_dist,
            "test": test_dist,
        },
    }

    with open(training_report_path, "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)

    print(f"\n[i] train.csv, validation.csv, test.csv saved to: {output_dir}")
    print(f"[i] training_report.json saved to: {training_report_path}")


if __name__ == "__main__":
    main()