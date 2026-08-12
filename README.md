# Movie Review Sentiment Dashboard

An end-to-end sentiment analysis pipeline that classifies IMDB movie reviews as positive or negative using logistic regression on TF-IDF features, paired with an interactive Streamlit dashboard for exploring the corpus and the model.

## Features

- **Sentiment Classification** -- Logistic regression model achieving ~91% accuracy on the IMDB 50k benchmark.
- **Text Processing Pipeline** -- Full NLP preprocessing with HTML stripping, lemmatization, stopword removal (preserving negation words), and TF-IDF vectorization with bigram support.
- **Interactive Dashboard** -- Streamlit app with filterable visualizations covering sentiment balance, review length, and vocabulary differences between the two classes.
- **Distinctive Vocabulary Analysis** -- Word clouds weighted by how much a term leans toward one class, rather than by raw frequency, so the two classes are actually distinguishable.
- **Model Interpretability** -- The terms with the largest logistic regression coefficients, showing what the classifier actually keys on.
- **Live Prediction** -- Enter any custom review text and get a real-time sentiment prediction with confidence scores.

## Tech Stack

- **Python** -- Core language
- **Scikit-Learn** -- TF-IDF vectorization and logistic regression classifier
- **Streamlit** -- Interactive web dashboard
- **Pandas** -- Data manipulation and analysis
- **Matplotlib / Seaborn** -- Statistical visualizations
- **Plotly** -- Interactive charts
- **WordCloud** -- Vocabulary visualizations
- **NLTK** -- Text preprocessing and lemmatization

## Project Structure

```
movie-review-sentiment-dashboard/
├── app.py                    # Streamlit dashboard application
├── requirements.txt          # Python dependencies
├── README.md
├── .gitignore
├── IMDB Dataset.csv          # Raw dataset (not tracked in git)
├── data/
│   └── movie_reviews.csv     # Prepared dataset (generated)
├── models/
│   ├── sentiment_model.joblib    # Trained classifier
│   ├── tfidf_vectorizer.joblib   # Fitted TF-IDF vectorizer
│   └── metrics.json              # Evaluation metrics & top model terms
└── src/
    ├── __init__.py
    ├── prepare_dataset.py    # Loads and cleans the raw IMDB CSV
    ├── preprocessing.py      # Text cleaning pipeline
    └── train_model.py        # Model training script
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

4. **Add the dataset:**

   Download the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place `IMDB Dataset.csv` in the project root. It needs a `review` column and a `sentiment` column containing `positive` / `negative`.

5. **Prepare the dataset:**
   ```bash
   python src/prepare_dataset.py
   ```
   Deduplicates, cleans, and precomputes the tokenized text into `data/movie_reviews.csv`. Takes a couple of minutes on 50k reviews; only needs to run once.

6. **Train the model:**
   ```bash
   python src/train_model.py
   ```

7. **Launch the dashboard:**
   ```bash
   streamlit run app.py
   ```

## Model Performance

Measured on a stratified 20% held-out test set (9,917 reviews):

| Metric     | Score  |
|------------|--------|
| Accuracy   | 90.7%  |
| Precision  | 90.7%  |
| Recall     | 90.7%  |
| F1-Score   | 90.7%  |

The model uses TF-IDF features with unigrams and bigrams (max 40,000 features) fed into a logistic regression classifier with balanced class weights.

The strongest positive coefficients are terms like *great*, *excellent*, *perfect*, *must see*, and *one best*; the strongest negative ones are *worst*, *awful*, *boring*, and *waste*.

## Dashboard Tabs

- **Overview** -- Sentiment balance, review length distribution, and a sample of real reviews.
- **Word Analysis** -- Distinctive vocabulary word clouds, raw term frequencies, and review length by sentiment.
- **Model Insights** -- Most predictive terms by coefficient, plus vocabulary overlap between the two classes.
- **Model Performance** -- Confusion matrix and per-class precision / recall / F1.
- **Live Prediction** -- Free-text sentiment prediction with confidence breakdown.

## Dataset

The project uses the **IMDB Dataset of 50K Movie Reviews**, a standard binary sentiment benchmark of real user-written reviews. Each row has:

- **Review text** -- Full-length review, averaging around 230 words (median 173)
- **Sentiment label** -- Binary classification (positive / negative), evenly balanced

After removing 418 exact duplicates, 49,582 reviews remain (24,884 positive / 24,698 negative).

Note that the raw data contains only review text and labels -- there is no genre, rating, or release year -- so the dashboard analyzes the text itself rather than movie metadata.

## License

This project is for educational and portfolio demonstration purposes.
