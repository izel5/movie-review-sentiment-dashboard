"""
Movie Review Sentiment Dashboard

An interactive Streamlit application for exploring sentiment analysis
results on the IMDB 50k movie review dataset. Provides visualizations
of sentiment balance, review length patterns, vocabulary differences
between positive and negative reviews, and the terms the classifier
weights most heavily, plus a live prediction interface.
"""

import sys
from collections import Counter
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
from wordcloud import WordCloud

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

POSITIVE_COLOR = "#38ef7d"
NEGATIVE_COLOR = "#f45c43"
SENTIMENT_COLORS = {"positive": POSITIVE_COLOR, "negative": NEGATIVE_COLOR}

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
    """Load the prepared IMDB reviews dataset."""
    df = pd.read_csv(DATA_PATH)
    df["clean_review"] = df["clean_review"].fillna("")
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


# The leading underscore tells Streamlit not to hash the DataFrame; cache
# identity comes from filter_key, which is cheap to hash.
@st.cache_data(show_spinner=False)
def word_frequencies(filter_key, sentiment: str, _df: pd.DataFrame, sample: int = 6000) -> Counter:
    """Count cleaned-token frequencies for one sentiment class."""
    subset = _df.loc[_df["sentiment"] == sentiment, "clean_review"]
    if len(subset) > sample:
        subset = subset.sample(sample, random_state=42)
    counts = Counter()
    for doc in subset:
        counts.update(doc.split())
    return counts


def distinctive_frequencies(
    counts: Counter, other: Counter, min_total: int = 25, min_share: float = 0.55
) -> Counter:
    """
    Keep terms that lean toward this class rather than terms that are merely common.

    Without this, both word clouds are dominated by 'film', 'movie', and 'one',
    which appear at nearly identical rates in positive and negative reviews.
    """
    if not other:
        return counts
    leaning = Counter()
    for term, count in counts.items():
        total = count + other.get(term, 0)
        if total >= min_total and count / total >= min_share:
            leaning[term] = count
    return leaning or counts


@st.cache_data(show_spinner=False)
def make_wordcloud(filter_key, sentiment: str, _freqs: Counter) -> np.ndarray:
    """Render a word cloud image from token frequencies."""
    colormap = "YlGn" if sentiment == "positive" else "OrRd"
    cloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=colormap,
        max_words=150,
        random_state=42,
    ).generate_from_frequencies(dict(_freqs.most_common(300)))
    return cloud.to_array()


# ---------------------------------------------------------------------------
# Dashboard Layout
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame):
    """Render sidebar filters; return the filtered DataFrame and a cache key."""
    st.sidebar.header("Filters")

    selected_sentiments = st.sidebar.multiselect(
        "Sentiment",
        options=["positive", "negative"],
        default=["positive", "negative"],
    )

    min_words, max_words = int(df["word_count"].min()), int(df["word_count"].max())
    length_range = st.sidebar.slider(
        "Review Length (words)",
        min_value=min_words,
        max_value=max_words,
        value=(min_words, max_words),
    )

    keyword = st.sidebar.text_input(
        "Keyword Search",
        placeholder="e.g. soundtrack",
        help="Show only reviews containing this text.",
    )

    filtered = df[
        df["sentiment"].isin(selected_sentiments)
        & df["word_count"].between(*length_range)
    ]
    if keyword.strip():
        filtered = filtered[
            filtered["review"].str.contains(keyword.strip(), case=False, regex=False)
        ]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Showing {len(filtered):,}** of {len(df):,} reviews")

    filter_key = (tuple(sorted(selected_sentiments)), length_range, keyword.strip().lower())
    return filtered, filter_key


def render_metrics_row(df: pd.DataFrame, metrics: dict):
    """Render the top-level KPI cards."""
    total = len(df)
    positive = (df["sentiment"] == "positive").sum()
    negative = (df["sentiment"] == "negative").sum()

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
        color_discrete_map=SENTIMENT_COLORS,
        hole=0.45,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
    st.plotly_chart(fig, width="stretch")


def render_length_distribution(df: pd.DataFrame):
    """Histogram of review length in words, split by sentiment."""
    st.subheader("Review Length Distribution")
    capped = df[df["word_count"] <= df["word_count"].quantile(0.99)]
    fig = px.histogram(
        capped,
        x="word_count",
        color="sentiment",
        nbins=60,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_layout(
        xaxis_title="Words per Review",
        yaxis_title="Count",
        margin=dict(t=20, b=20),
        height=350,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Top 1% longest reviews trimmed so the bulk of the distribution stays readable.")


def render_length_summary(df: pd.DataFrame):
    """Box plot comparing review length across sentiment classes."""
    st.subheader("Length by Sentiment")
    capped = df[df["word_count"] <= df["word_count"].quantile(0.99)]
    fig = px.box(
        capped,
        x="sentiment",
        y="word_count",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        points=False,
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Words per Review",
        margin=dict(t=20, b=20),
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def render_sample_reviews(df: pd.DataFrame):
    """Show a handful of real reviews from the current filter."""
    st.subheader("Sample Reviews")
    if df.empty:
        st.info("No reviews match the current filters.")
        return
    sample = df.sample(min(10, len(df)), random_state=42)[
        ["sentiment", "word_count", "review"]
    ].copy()
    sample["review"] = sample["review"].str.slice(0, 400) + "..."
    st.dataframe(sample, width="stretch", hide_index=True)


def render_wordclouds(df: pd.DataFrame, filter_key):
    """Side-by-side word clouds of the vocabulary distinctive to each class."""
    st.subheader("Distinctive Vocabulary by Sentiment")
    st.caption(
        "Terms that appear disproportionately in one class, drawn from cleaned text "
        "(stopwords removed, negations kept) and sampled to at most 6,000 reviews per "
        "class. Words common to both — 'film', 'movie', 'one' — are filtered out."
    )
    counters = {
        sentiment: word_frequencies(filter_key, sentiment, df)
        if (df["sentiment"] == sentiment).any()
        else Counter()
        for sentiment in ("positive", "negative")
    }

    cols = st.columns(2)
    for col, sentiment in zip(cols, ["positive", "negative"]):
        with col:
            st.markdown(f"**{sentiment.capitalize()} Reviews**")
            if not counters[sentiment]:
                st.info(f"No {sentiment} reviews in the current filter.")
                continue
            other = counters["negative" if sentiment == "positive" else "positive"]
            freqs = distinctive_frequencies(counters[sentiment], other)
            st.image(make_wordcloud(filter_key, sentiment, freqs), width="stretch")

    if not all(counters.values()):
        st.info(
            "Only one sentiment is selected, so there is nothing to contrast against — "
            "the cloud above falls back to raw word frequency."
        )


def render_top_terms(df: pd.DataFrame, filter_key):
    """Bar chart of the most frequent terms in each sentiment class."""
    st.subheader("Top Terms by Raw Frequency")
    st.caption(
        "Unfiltered counts, for corpus context. The two classes look alike here — that "
        "similarity is the point, and it is why the clouds above weight by distinctiveness."
    )
    cols = st.columns(2)
    for col, sentiment in zip(cols, ["positive", "negative"]):
        with col:
            if not (df["sentiment"] == sentiment).any():
                continue
            freqs = word_frequencies(filter_key, sentiment, df)
            top = pd.DataFrame(freqs.most_common(20), columns=["term", "count"])
            fig = px.bar(
                top.sort_values("count"),
                x="count",
                y="term",
                orientation="h",
                color_discrete_sequence=[SENTIMENT_COLORS[sentiment]],
            )
            fig.update_layout(
                title=f"{sentiment.capitalize()} Reviews",
                xaxis_title="Occurrences",
                yaxis_title="",
                margin=dict(t=40, b=20),
                height=500,
            )
            st.plotly_chart(fig, width="stretch")


def render_predictive_terms(metrics: dict):
    """Bar charts of the terms with the strongest model coefficients."""
    st.subheader("Most Predictive Terms")
    st.markdown(
        "Terms with the largest logistic regression coefficients. These are what the "
        "model actually leans on, as opposed to what simply appears most often."
    )
    cols = st.columns(2)
    specs = [
        (cols[0], "top_positive_terms", "Pushes Toward Positive", POSITIVE_COLOR, True),
        (cols[1], "top_negative_terms", "Pushes Toward Negative", NEGATIVE_COLOR, False),
    ]
    for col, key, title, color, ascending in specs:
        with col:
            terms = pd.DataFrame(metrics[key])
            terms["magnitude"] = terms["coefficient"].abs()
            fig = px.bar(
                terms.sort_values("magnitude"),
                x="magnitude",
                y="term",
                orientation="h",
                color_discrete_sequence=[color],
            )
            fig.update_layout(
                title=title,
                xaxis_title="Coefficient Magnitude",
                yaxis_title="",
                margin=dict(t=40, b=20),
                height=600,
            )
            st.plotly_chart(fig, width="stretch")


def render_vocabulary_stats(df: pd.DataFrame, filter_key):
    """Summary statistics on vocabulary overlap between the two classes."""
    st.subheader("Vocabulary Overlap")
    if not df["sentiment"].nunique() == 2:
        st.info("Select both sentiments to compare vocabularies.")
        return

    pos = word_frequencies(filter_key, "positive", df)
    neg = word_frequencies(filter_key, "negative", df)
    shared = set(pos) & set(neg)

    cols = st.columns(4)
    cols[0].metric("Unique Terms (Positive)", f"{len(pos):,}")
    cols[1].metric("Unique Terms (Negative)", f"{len(neg):,}")
    cols[2].metric("Shared Terms", f"{len(shared):,}")
    cols[3].metric(
        "Overlap",
        f"{len(shared) / len(set(pos) | set(neg)):.1%}" if (pos or neg) else "n/a",
    )

    distinct = [
        {
            "term": term,
            "positive_share": pos[term] / (pos[term] + neg[term]),
            "total": pos[term] + neg[term],
        }
        for term in shared
        if pos[term] + neg[term] >= 200
    ]
    if not distinct:
        return

    skew = pd.DataFrame(distinct)
    skew["skew"] = (skew["positive_share"] - 0.5).abs()
    top = skew.nlargest(20, "skew").sort_values("positive_share")
    fig = px.bar(
        top,
        x="positive_share",
        y="term",
        orientation="h",
        color="positive_share",
        color_continuous_scale=[NEGATIVE_COLOR, "#dddddd", POSITIVE_COLOR],
        range_color=[0, 1],
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Most Sentiment-Skewed Common Terms",
        xaxis_title="Share of Occurrences in Positive Reviews",
        yaxis_title="",
        margin=dict(t=40, b=20),
        height=550,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, width="stretch")


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
                color = POSITIVE_COLOR if result["label"] == "Positive" else NEGATIVE_COLOR
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
                        marker_color=[NEGATIVE_COLOR, POSITIVE_COLOR],
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
                st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Please enter a review to analyze.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("Movie Review Sentiment Dashboard")

    # Load resources
    df = load_data()
    model, vectorizer = load_model()
    metrics = load_metrics()

    st.markdown(
        f"Analyze sentiment patterns across **{len(df):,} IMDB movie reviews** using a "
        "logistic regression model trained on TF-IDF features."
    )

    # Sidebar filters
    filtered_df, filter_key = render_sidebar(df)

    # KPI row
    render_metrics_row(filtered_df, metrics)
    st.markdown("---")

    if filtered_df.empty:
        st.warning("No reviews match the current filters. Widen them in the sidebar.")
        return

    tab_overview, tab_words, tab_insights, tab_model, tab_predict = st.tabs(
        ["Overview", "Word Analysis", "Model Insights", "Model Performance", "Live Prediction"]
    )

    with tab_overview:
        col1, col2 = st.columns(2)
        with col1:
            render_sentiment_distribution(filtered_df)
        with col2:
            render_length_distribution(filtered_df)
        render_sample_reviews(filtered_df)

    with tab_words:
        render_wordclouds(filtered_df, filter_key)
        st.markdown("---")
        render_top_terms(filtered_df, filter_key)
        st.markdown("---")
        render_length_summary(filtered_df)

    with tab_insights:
        render_predictive_terms(metrics)
        st.markdown("---")
        render_vocabulary_stats(filtered_df, filter_key)

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
                width="stretch",
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
