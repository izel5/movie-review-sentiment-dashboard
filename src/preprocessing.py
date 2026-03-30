"""
Text Preprocessing Pipeline

Handles text cleaning, normalization, and TF-IDF vectorization
for the sentiment analysis model.
"""

import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (silent if already present)
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
# Keep negation words -- they carry sentiment signal
NEGATION_WORDS = {"not", "no", "nor", "neither", "never", "nobody", "nothing",
                  "nowhere", "hardly", "barely", "scarcely", "don", "don't",
                  "doesn", "doesn't", "didn", "didn't", "won", "won't",
                  "wouldn", "wouldn't", "couldn", "couldn't", "shouldn",
                  "shouldn't", "isn", "isn't", "aren", "aren't", "wasn",
                  "wasn't", "weren", "weren't", "hasn", "hasn't", "haven",
                  "haven't", "hadn", "hadn't"}
STOP_WORDS -= NEGATION_WORDS

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean and normalize a single review text.

    Steps:
        1. Lowercase
        2. Remove HTML tags
        3. Remove URLs
        4. Remove punctuation (keep apostrophes for contractions)
        5. Remove extra whitespace
        6. Remove stopwords (preserving negations)
        7. Lemmatize remaining tokens
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to the review column of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'review' column.

    Returns
    -------
    pd.DataFrame
        Copy of the input with an added 'clean_review' column.
    """
    df = df.copy()
    df["clean_review"] = df["review"].apply(clean_text)
    # Drop rows where cleaning produced empty strings
    df = df[df["clean_review"].str.strip().astype(bool)].reset_index(drop=True)
    return df
