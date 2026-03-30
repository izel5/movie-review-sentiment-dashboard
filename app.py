"""
Movie Review Sentiment Dashboard

An interactive Streamlit application for exploring sentiment analysis
results on movie reviews. Provides visualizations of sentiment trends
across genres, release periods, and rating distributions, plus a
live prediction interface for custom text input.
"""

import sys
from pathlib import Path

import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

# Allow imports from src/
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocessing import clean_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.joblib"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
DATA_PATH = PROJECT_ROOT / "data" / "movie_reviews.csv"

st.set_page_config(
    page_title="Movie Review Sentiment Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .metric-card h1 {
        margin: 0.25rem 0 0 0;
        font-size: 2rem;
    }
    .positive-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .negative-card {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .neutral-card {
        background: linear-gradient(135deg, #4568dc 0%, #b06ab3 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data & Model Loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the movie reviews dataset."""
    df = pd.read_csv(DATA_PATH)
    df["release_period"] = pd.cut(
        df["release_year"],
        bins=[1994, 2000, 2005, 2010, 2015, 2020, 2025],
        labels=["1995-2000", "2001-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2024"],
    )
    return df


@st.cache_resource
def load_model():
    """Load the trained model and vectorizer."""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


@st.cache_data
def load_metrics() -> dict:
    """Load saved model metrics."""
    with open(METRICS_PATH) as f:
        return json.load(f)


def predict_sentiment(text: str, model, vectorizer) -> dict:
    """Predict sentiment for a custom review text."""
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return {
        "label": "Positive" if prediction == 1 else "Negative",
        "confidence": float(max(probabilities)),
        "positive_prob": float(probabilities[1]),
        "negative_prob": float(probabilities[0]),
    }


# ---------------------------------------------------------------------------
# Dashboard Layout
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered DataFrame."""
    st.sidebar.header("Filters")

    selected_genres = st.sidebar.multiselect(
        "Select Genres",
        options=sorted(df["genre"].unique()),
        default=sorted(df["genre"].unique()),
    )

    year_range = st.sidebar.slider(
        "Release Year Range",
        min_value=int(df["release_year"].min()),
        max_value=int(df["release_year"].max()),
        value=(int(df["release_year"].min()), int(df["release_year"].max())),
    )

    selected_sentiments = st.sidebar.multiselect(
        "Sentiment",
        options=["positive", "negative"],
        default=["positive", "negative"],
    )

    filtered = df[
        (df["genre"].isin(selected_genres))
        & (df["release_year"].between(*year_range))
        & (df["sentiment"].isin(selected_sentiments))
    ]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Showing {len(filtered):,}** of {len(df):,} reviews")
    return filtered


def render_metrics_row(df: pd.DataFrame, metrics: dict):
    """Render the top-level KPI cards."""
    total = len(df)
    positive = (df["sentiment"] == "positive").sum()
    negative = (df["sentiment"] == "negative").sum()
    avg_rating = df["rating"].mean()

    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f"""<div class="metric-card neutral-card">
            <h3>Total Reviews</h3><h1>{total:,}</h1></div>""",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""<div class="metric-card positive-card">
            <h3>Positive Reviews</h3><h1>{positive:,}</h1></div>""",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f"""<div class="metric-card negative-card">
            <h3>Negative Reviews</h3><h1>{negative:,}</h1></div>""",
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            f"""<div class="metric-card">
            <h3>Model Accuracy</h3><h1>{metrics['accuracy']:.1%}</h1></div>""",
            unsafe_allow_html=True,
        )


def render_sentiment_distribution(df: pd.DataFrame):
    """Pie chart of overall sentiment distribution."""
    st.subheader("Overall Sentiment Distribution")
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(
        counts,
        values="count",
        names="sentiment",
        color="sentiment",
        color_discrete_map={"positive": "#38ef7d", "negative": "#f45c43"},
        hole=0.45,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
    st.plotly_chart(fig, use_container_width=True)


def render_genre_sentiment(df: pd.DataFrame):
    """Stacked bar chart of sentiment by genre."""
    st.subheader("Sentiment by Genre")
    genre_sent = (
        df.groupby(["genre", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        genre_sent,
        x="genre",
        y="count",
        color="sentiment",
        barmode="group",
        color_discrete_map={"positive": "#38ef7d", "negative": "#f45c43"},
    )
    fig.update_layout(
        xaxis_title="Genre",
        yaxis_title="Number of Reviews",
        margin=dict(t=20, b=20),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sentiment_by_period(df: pd.DataFrame):
    """Line chart of sentiment proportion over release periods."""
    st.subheader("Sentiment Trends by Release Period")
    period_sent = (
        df.groupby(["release_period", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    period_sent_pct = period_sent.div(period_sent.sum(axis=1), axis=0) * 100
    period_sent_pct = period_sent_pct.reset_index().melt(
        id_vars="release_period", var_name="sentiment", value_name="percentage"
    )
    fig = px.line(
        period_sent_pct,
        x="release_period",
        y="percentage",
        color="sentiment",
        markers=True,
        color_discrete_map={"positive": "#38ef7d", "negative": "#f45c43"},
    )
    fig.update_layout(
        xaxis_title="Release Period",
        yaxis_title="Percentage (%)",
        margin=dict(t=20, b=20),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_rating_distribution(df: pd.DataFrame):
    """Histogram of ratings split by sentiment."""
    st.subheader("Rating Distribution by Sentiment")
    fig = px.histogram(
        df,
        x="rating",
        color="sentiment",
        nbins=20,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={"positive": "#38ef7d", "negative": "#f45c43"},
    )
    fig.update_layout(
        xaxis_title="Rating",
        yaxis_title="Count",
        margin=dict(t=20, b=20),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_avg_rating_by_genre(df: pd.DataFrame):
    """Bar chart of average rating per genre."""
    st.subheader("Average Rating by Genre")
    avg = df.groupby("genre")["rating"].mean().sort_values(ascending=True).reset_index()
    fig = px.bar(
        avg,
        x="rating",
        y="genre",
        orientation="h",
        color="rating",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(
        xaxis_title="Average Rating",
        yaxis_title="Genre",
        margin=dict(t=20, b=20),
        height=400,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_yearly_volume(df: pd.DataFrame):
    """Area chart of review volume by year."""
    st.subheader("Review Volume by Year")
    yearly = (
        df.groupby(["release_year", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    fig = px.area(
        yearly,
        x="release_year",
        y="count",
        color="sentiment",
        color_discrete_map={"positive": "#38ef7d", "negative": "#f45c43"},
    )
    fig.update_layout(
        xaxis_title="Release Year",
        yaxis_title="Number of Reviews",
        margin=dict(t=20, b=20),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix(metrics: dict):
    """Display the model confusion matrix."""
    st.subheader("Model Confusion Matrix")
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)


def render_prediction_interface(model, vectorizer):
    """Live sentiment prediction for user-provided text."""
    st.subheader("Try It Yourself")
    st.write("Enter a movie review below to get a real-time sentiment prediction.")

    user_text = st.text_area(
        "Your Review",
        height=120,
        placeholder="e.g., This movie was a fantastic journey with stunning visuals...",
    )

    if st.button("Predict Sentiment", type="primary"):
        if user_text.strip():
            result = predict_sentiment(user_text, model, vectorizer)
            col1, col2 = st.columns(2)
            with col1:
                color = "#38ef7d" if result["label"] == "Positive" else "#f45c43"
                st.markdown(
                    f"""<div style="text-align:center; padding:1rem;
                    border-radius:0.5rem; background:{color}20;
                    border:2px solid {color};">
                    <h2 style="color:{color}; margin:0;">{result['label']}</h2>
                    <p style="margin:0.25rem 0 0 0;">Confidence: {result['confidence']:.1%}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col2:
                fig = go.Figure(
                    go.Bar(
                        x=[result["negative_prob"], result["positive_prob"]],
                        y=["Negative", "Positive"],
                        orientation="h",
                        marker_color=["#f45c43", "#38ef7d"],
                        text=[f"{result['negative_prob']:.1%}", f"{result['positive_prob']:.1%}"],
                        textposition="auto",
                    )
                )
                fig.update_layout(
                    xaxis_title="Probability",
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=200,
                    xaxis=dict(range=[0, 1]),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter a review to analyze.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("Movie Review Sentiment Dashboard")
    st.markdown(
        "Analyze sentiment patterns across **2,500 movie reviews** using a "
        "logistic regression model trained on TF-IDF features."
    )

    # Load resources
    df = load_data()
    model, vectorizer = load_model()
    metrics = load_metrics()

    # Sidebar filters
    filtered_df = render_sidebar(df)

    # KPI row
    render_metrics_row(filtered_df, metrics)
    st.markdown("---")

    # Tabs for organized content
    tab_overview, tab_genres, tab_temporal, tab_model, tab_predict = st.tabs(
        ["Overview", "Genre Analysis", "Temporal Trends", "Model Performance", "Live Prediction"]
    )

    with tab_overview:
        col1, col2 = st.columns(2)
        with col1:
            render_sentiment_distribution(filtered_df)
        with col2:
            render_rating_distribution(filtered_df)

    with tab_genres:
        render_genre_sentiment(filtered_df)
        render_avg_rating_by_genre(filtered_df)

    with tab_temporal:
        col1, col2 = st.columns(2)
        with col1:
            render_sentiment_by_period(filtered_df)
        with col2:
            render_yearly_volume(filtered_df)

    with tab_model:
        col1, col2 = st.columns(2)
        with col1:
            render_confusion_matrix(metrics)
        with col2:
            st.subheader("Model Metrics")
            report = metrics["classification_report"]
            metrics_df = pd.DataFrame(
                {
                    "Class": ["Negative", "Positive", "Weighted Avg"],
                    "Precision": [
                        report["negative"]["precision"],
                        report["positive"]["precision"],
                        report["weighted avg"]["precision"],
                    ],
                    "Recall": [
                        report["negative"]["recall"],
                        report["positive"]["recall"],
                        report["weighted avg"]["recall"],
                    ],
                    "F1-Score": [
                        report["negative"]["f1-score"],
                        report["positive"]["f1-score"],
                        report["weighted avg"]["f1-score"],
                    ],
                }
            ).set_index("Class")
            st.dataframe(
                metrics_df.style.format("{:.4f}").background_gradient(
                    cmap="Greens", axis=None
                ),
                use_container_width=True,
            )
            st.markdown("---")
            st.metric("Overall Accuracy", f"{metrics['accuracy']:.2%}")
            st.metric("Training Samples", f"{metrics['train_size']:,}")
            st.metric("Test Samples", f"{metrics['test_size']:,}")
            st.metric("Vocabulary Size", f"{metrics['vocab_size']:,}")

    with tab_predict:
        render_prediction_interface(model, vectorizer)


if __name__ == "__main__":
    main()
