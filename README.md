# Movie Review Sentiment Dashboard

An end-to-end sentiment analysis pipeline that classifies movie reviews as positive or negative using logistic regression on TF-IDF features, paired with an interactive Streamlit dashboard for exploring sentiment trends.

## Features

- **Sentiment Classification** -- Logistic regression model achieving 85%+ accuracy on binary sentiment classification of movie reviews.
- **Text Processing Pipeline** -- Full NLP preprocessing with tokenization, lemmatization, stopword removal (preserving negation words), and TF-IDF vectorization with bigram support.
- **Interactive Dashboard** -- Streamlit app with filterable visualizations covering sentiment distribution, genre analysis, temporal trends, and rating distributions.
- **Live Prediction** -- Enter any custom review text and get real-time sentiment prediction with confidence scores.
- **Genre & Temporal Analysis** -- Explore how sentiment varies across 10 movie genres and release periods from 1995 to 2024.

## Tech Stack

- **Python** -- Core language
- **Scikit-Learn** -- TF-IDF vectorization and logistic regression classifier
- **Streamlit** -- Interactive web dashboard
- **Pandas** -- Data manipulation and analysis
- **Matplotlib / Seaborn** -- Statistical visualizations
- **Plotly** -- Interactive charts
- **NLTK** -- Text preprocessing and lemmatization

## Project Structure

```
movie-review-sentiment-dashboard/
├── app.py                    # Streamlit dashboard application
├── requirements.txt          # Python dependencies
├── README.md
├── .gitignore
├── data/
│   └── movie_reviews.csv     # Dataset (generated)
├── models/
│   ├── sentiment_model.joblib    # Trained classifier
│   ├── tfidf_vectorizer.joblib   # Fitted TF-IDF vectorizer
│   └── metrics.json              # Evaluation metrics
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py   # Synthetic dataset generator
│   ├── preprocessing.py      # Text cleaning pipeline
│   └── train_model.py        # Model training script
└── notebooks/
    └── (exploratory analysis)
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd movie-review-sentiment-dashboard
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate the dataset:**
   ```bash
   python src/generate_dataset.py
   ```

5. **Train the model:**
   ```bash
   cd src && python train_model.py && cd ..
   ```

6. **Launch the dashboard:**
   ```bash
   streamlit run app.py
   ```

## Model Performance

| Metric     | Score  |
|------------|--------|
| Accuracy   | 85%+   |
| Precision  | 85%+   |
| Recall     | 85%+   |
| F1-Score   | 85%+   |

The model uses TF-IDF features with unigrams and bigrams (max 10,000 features) fed into a logistic regression classifier with balanced class weights.

## Screenshots

> Screenshots of the dashboard can be added here after running the application.

## Dataset

The project uses a synthetic dataset of 2,500 movie reviews designed to mirror realistic IMDB review patterns. Each review includes:

- **Review text** -- Varied sentence structures with sentiment-appropriate vocabulary
- **Sentiment label** -- Binary classification (positive / negative)
- **Rating** -- Numerical score (1.0--10.0) correlated with sentiment
- **Genre** -- One of 10 major film genres
- **Release year** -- Ranging from 1995 to 2024 with realistic distribution

## License

This project is for educational and portfolio demonstration purposes.
