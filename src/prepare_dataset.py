"""
IMDB Dataset Preparation

Reads the raw IMDB 50k movie review dataset, removes duplicates,
precomputes cleaned text and length features, and writes the
analysis-ready CSV consumed by the training script and dashboard.
"""

import argparse
from pathlib import Path

import pandas as pd

from preprocessing import clean_text

PROJECT_ROOT = Path(__file__).parent.parent
RAW_PATH = PROJECT_ROOT / "IMDB Dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "movie_reviews.csv"


def prepare(raw_path: Path = RAW_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Load, clean, and persist the IMDB review dataset."""
    print(f"Loading {raw_path}...")
    df = pd.read_csv(raw_path)

    missing = {"review", "sentiment"} - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns 'review' and 'sentiment', missing: {missing}")

    print(f"  Loaded {len(df):,} reviews")

    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    print(f"  {len(df):,} reviews after removing duplicates")

    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df = df[df["sentiment"].isin(["positive", "negative"])].reset_index(drop=True)

    print("Cleaning text (this takes a couple of minutes on 50k reviews)...")
    df["clean_review"] = df["review"].apply(clean_text)
    df = df[df["clean_review"].str.strip().astype(bool)].reset_index(drop=True)

    df["word_count"] = df["review"].str.split().str.len()
    df["char_count"] = df["review"].str.len()
    df.insert(0, "review_id", range(1, len(df) + 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df):,} reviews to {output_path}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    print(f"Median review length: {df['word_count'].median():.0f} words")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH, help="Path to the raw IMDB CSV")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH, help="Path for the prepared CSV")
    args = parser.parse_args()
    prepare(args.raw, args.out)
