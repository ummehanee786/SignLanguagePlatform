"""
normalize_landmarks.py

Task 1: Reads landmarks.csv (raw extracted landmark features + label
per row), normalizes every sample's coordinates using a consistent
strategy, and writes normalized_landmarks.csv.

Normalization strategy used (documented in detail in
NORMALIZATION_STRATEGY.md alongside this file):

    1. Wrist-relative translation: every landmark's (x, y, z) has the
       wrist's (landmark 0) position subtracted from it, so the wrist
       becomes the origin (0, 0, 0) for every sample.

    2. Hand-size scaling: every coordinate is then divided by the
       distance from the wrist to the middle finger's MCP joint
       (landmark 9) - a reasonably stable "hand size" reference.
"""

import argparse
from pathlib import Path

import pandas as pd


def get_feature_columns() -> list[str]:
    return [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def normalize_row(row_values: list[float]) -> list[float]:
    points = [row_values[i:i + 3] for i in range(0, len(row_values), 3)]
    wrist = points[0]

    translated = [
        [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]]
        for p in points
    ]

    middle_mcp = translated[9]
    scale = (middle_mcp[0] ** 2 + middle_mcp[1] ** 2 + middle_mcp[2] ** 2) ** 0.5
    if scale < 1e-6:
        scale = 1e-6

    normalized = [[p[0] / scale, p[1] / scale, p[2] / scale] for p in translated]

    flat = []
    for p in normalized:
        flat.extend(p)
    return flat


def normalize_dataset(input_csv: Path, output_csv: Path) -> dict:
    df = pd.read_csv(input_csv)
    feature_cols = get_feature_columns()

    missing = [c for c in feature_cols + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing expected columns: {missing}")

    normalized_rows = []
    for _, row in df.iterrows():
        values = row[feature_cols].tolist()
        normalized_rows.append(normalize_row(values))

    normalized_df = pd.DataFrame(normalized_rows, columns=feature_cols)
    normalized_df["label"] = df["label"].values

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_csv(output_csv, index=False)

    return {
        "input_rows": len(df),
        "output_rows": len(normalized_df),
        "input_file": str(input_csv),
        "output_file": str(output_csv),
    }


def main():
    parser = argparse.ArgumentParser(description="Normalize landmark coordinates.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    backend_root = script_dir.parent.parent.parent.parent

    input_csv = Path(args.input) if args.input else backend_root / "data" / "landmarks.csv"
    output_csv = Path(args.output) if args.output else input_csv.parent / "normalized_landmarks.csv"

    if not input_csv.exists():
        print(f"[!] Input file not found: {input_csv}")
        return

    print(f"Reading: {input_csv}")
    result = normalize_dataset(input_csv, output_csv)

    print(f"Normalized {result['output_rows']} samples.")
    print(f"Saved to: {result['output_file']}")


if __name__ == "__main__":
    main()