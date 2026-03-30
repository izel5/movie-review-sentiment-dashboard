"""
Model Training Pipeline

Trains a logistic regression classifier on TF-IDF features
extracted from movie reviews. Saves the trained model and
vectorizer for use by the Streamlit dashboard.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_dataframe


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"


def train(
    data_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    C: float = 1.0,
) -> dict:
    """
    End-to-end training pipeline.

    Parameters
    ----------
    data_path : str
        Path to the reviews CSV. Defaults to data/movie_reviews.csv.
    test_size : float
        Fraction of data held out for evaluation.
    random_state : int
        Seed for reproducibility.
    max_features : int
        Maximum vocabulary size for TF-IDF.
    ngram_range : tuple
        N-gram range for TF-IDF vectorizer.
    C : float
        Regularization strength for logistic regression.

    Returns
    -------
    dict
        Dictionary containing accuracy, classification report, and
        confusion matrix.
    """
    if data_path is None:
        data_path = DATA_DIR / "movie_reviews.csv"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load & preprocess ----
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} reviews")

    print("Preprocessing text...")
    df = preprocess_dataframe(df)
    print(f"  {len(df)} reviews after cleaning")

    # ---- Encode labels ----
    label_map = {"negative": 0, "positive": 1}
    y = df["sentiment"].map(label_map).values
    texts = df["clean_review"].values

    # ---- Train/test split ----
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train size: {len(X_train_text)}  |  Test size: {len(X_test_text)}")

    # ---- TF-IDF vectorization ----
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    # ---- Train model ----
    print("Training logistic regression...")
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["negative", "positive"])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    # ---- Save artifacts ----
    model_path = MODEL_DIR / "sentiment_model.joblib"
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    metrics_path = MODEL_DIR / "metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    metrics = {
        "accuracy": float(accuracy),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["negative", "positive"], output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
        "train_size": int(len(X_train_text)),
        "test_size": int(len(X_test_text)),
        "vocab_size": len(vectorizer.vocabulary_),
        "max_features": max_features,
        "ngram_range": list(ngram_range),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
    print(f"Metrics saved to {metrics_path}")

    return metrics


if __name__ == "__main__":
    metrics = train()
    if metrics["accuracy"] < 0.85:
        print(f"\nWARNING: Accuracy {metrics['accuracy']:.4f} is below 85% target.")
    else:
        print(f"\nSUCCESS: Accuracy {metrics['accuracy']:.4f} meets the 85% target.")
