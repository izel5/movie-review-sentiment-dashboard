"""
Synthetic IMDB Movie Review Dataset Generator

Generates a realistic dataset of movie reviews with sentiment labels,
genres, ratings, and release years for training and dashboard visualization.
"""

import random
import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

GENRES = [
    "Action", "Comedy", "Drama", "Horror", "Sci-Fi",
    "Romance", "Thriller", "Animation", "Documentary", "Adventure",
]

POSITIVE_TEMPLATES = [
    "An absolutely {adj} film that {verb} from start to finish. The {element} was outstanding and I couldn't look away. {extra}",
    "This movie was {adj}! The {element} alone makes it worth watching. {extra} Highly recommended.",
    "I was blown away by how {adj} this film turned out. The {element} is top-notch. {extra}",
    "One of the best movies I've seen this year. {adj} {element} and a story that keeps you hooked. {extra}",
    "Truly {adj} cinema. The director crafted something special here with incredible {element}. {extra}",
    "What a {adj} experience! Every scene was carefully composed. The {element} elevated everything. {extra}",
    "This is filmmaking at its finest. The {adj} {element} paired with excellent pacing makes it a must-see. {extra}",
    "Loved every minute. The {element} was {adj} and the story resonated deeply. {extra}",
    "A {adj} masterpiece that showcases brilliant {element}. I'd gladly watch it again. {extra}",
    "Rarely do you find a movie this {adj}. The {element} was perfectly executed. {extra}",
    "Superb film with {adj} {element}. The cast delivered powerhouse performances throughout. {extra}",
    "An unexpectedly {adj} movie. The {element} caught me off guard in the best way possible. {extra}",
    "Heartfelt and {adj}, this film nails every emotional beat. The {element} is unforgettable. {extra}",
    "A must-watch for any movie fan. The {adj} {element} sets a new standard. {extra}",
    "Incredible storytelling backed by {adj} {element}. One of the strongest releases in recent memory. {extra}",
]

NEGATIVE_TEMPLATES = [
    "A {adj} film that fails on almost every level. The {element} was particularly disappointing. {extra}",
    "I expected much more but was left feeling {adj}. The {element} needed serious work. {extra}",
    "What a {adj} waste of time. The {element} was below average and the story went nowhere. {extra}",
    "Deeply {adj} movie with poor {element}. I struggled to stay awake through the second half. {extra}",
    "The {element} was {adj} and the plot was predictable from the first scene. {extra} Skip this one.",
    "Unfortunately {adj} in every department. The {element} felt lazy and uninspired. {extra}",
    "A {adj} attempt that misses the mark entirely. The {element} is the weakest link. {extra}",
    "This movie was painfully {adj}. Poor {element} and a script that needed several more rewrites. {extra}",
    "Not worth your time. The {adj} {element} drags the entire film down. {extra}",
    "I wanted to like this but the {adj} {element} made it impossible to enjoy. {extra}",
    "One of the worst films this year. The {element} was {adj} and the direction was aimless. {extra}",
    "Boring, {adj}, and forgettable. The {element} offered nothing new to the genre. {extra}",
    "A {adj} mess from beginning to end. The {element} never comes together coherently. {extra}",
    "Struggled through the entire runtime. The {adj} {element} made every scene feel tedious. {extra}",
    "Disappointing on all fronts. The {element} was {adj} and the ending was laughable. {extra}",
]

POSITIVE_ADJS = [
    "brilliant", "stunning", "captivating", "remarkable", "phenomenal",
    "outstanding", "excellent", "wonderful", "riveting", "compelling",
    "gripping", "beautiful", "powerful", "engaging", "mesmerizing",
    "spectacular", "extraordinary", "breathtaking", "fantastic", "superb",
]

NEGATIVE_ADJS = [
    "terrible", "disappointing", "dreadful", "mediocre", "uninspired",
    "dull", "tedious", "lackluster", "flat", "forgettable",
    "painful", "atrocious", "abysmal", "weak", "generic",
    "lifeless", "clumsy", "tiresome", "stale", "sloppy",
]

ELEMENTS = [
    "cinematography", "acting", "screenplay", "direction", "score",
    "pacing", "dialogue", "visual effects", "character development",
    "plot", "editing", "sound design", "production value", "casting",
    "narrative structure",
]

POSITIVE_EXTRAS = [
    "The supporting cast was equally impressive.",
    "I found myself emotionally invested throughout.",
    "The soundtrack perfectly complemented every scene.",
    "It's the kind of film that stays with you long after the credits roll.",
    "A welcome surprise in a year of lackluster releases.",
    "The third act is where the film truly shines.",
    "Every technical aspect was polished to perfection.",
    "I left the theater wanting more in the best way possible.",
    "This deserves all the praise it's been getting.",
    "A refreshing take on a familiar formula.",
    "",
    "",
    "",
]

NEGATIVE_EXTRAS = [
    "The supporting cast couldn't salvage this mess.",
    "I found myself checking my phone constantly.",
    "The soundtrack felt out of place and distracting.",
    "I wish I could get those two hours of my life back.",
    "Even the trailer was misleading about what this film actually is.",
    "The third act completely falls apart.",
    "It felt like a first draft that nobody bothered to revise.",
    "I walked out of the theater feeling genuinely annoyed.",
    "I have no idea how this got greenlit.",
    "A tired rehash of better films that came before it.",
    "",
    "",
    "",
]

POSITIVE_VERBS = [
    "captivates", "delivers", "impresses", "entertains", "thrills",
    "enchants", "excels", "dazzles", "resonates", "soars",
]

NEGATIVE_VERBS = [
    "disappoints", "stumbles", "drags", "fails", "bores",
    "frustrates", "falters", "collapses", "meanders", "flounders",
]

# Ambiguous templates -- mix positive/negative language to create harder examples
AMBIGUOUS_POSITIVE_TEMPLATES = [
    "Despite some {neg_adj} moments, the film ultimately delivers a {pos_adj} experience. The {element} had rough patches but came together well.",
    "Not a perfect movie by any means. The {element} was uneven and sometimes {neg_adj}, but overall I found it {pos_adj} enough to recommend.",
    "I went in expecting something {neg_adj} but was pleasantly surprised. The {element} won me over despite a slow start.",
    "It has its flaws and some scenes felt {neg_adj}, yet the {pos_adj} {element} makes up for the weaker parts.",
    "A mixed bag that leans positive. Some {neg_adj} choices in the first act, but the {element} carries it through to a satisfying finish.",
    "The movie struggles in places and the {element} is occasionally {neg_adj}, but when it works it's genuinely {pos_adj}.",
]

AMBIGUOUS_NEGATIVE_TEMPLATES = [
    "While the {element} showed {pos_adj} potential, the film ultimately disappoints. Too many {neg_adj} choices dragged it down.",
    "There were {pos_adj} moments scattered throughout, but the overall experience was {neg_adj}. The {element} couldn't save the weak script.",
    "I wanted to love this movie. The {element} had flashes of something {pos_adj}, but the execution was {neg_adj} and left me cold.",
    "An ambitious film with some {pos_adj} ideas that never quite work. The {element} starts strong but becomes {neg_adj} by the end.",
    "On paper this should have been {pos_adj}, but in practice the {element} is {neg_adj} and the pacing kills any momentum.",
    "Promising premise with {pos_adj} production values that are wasted on a {neg_adj} script. The {element} deserved better material.",
]


def generate_review(sentiment: str, ambiguous: bool = False) -> str:
    """Generate a single synthetic movie review."""
    if ambiguous:
        if sentiment == "positive":
            template = random.choice(AMBIGUOUS_POSITIVE_TEMPLATES)
        else:
            template = random.choice(AMBIGUOUS_NEGATIVE_TEMPLATES)
        adj = random.choice(POSITIVE_ADJS)
        neg_adj = random.choice(NEGATIVE_ADJS)
        element = random.choice(ELEMENTS)
        review = template.format(pos_adj=adj, neg_adj=neg_adj, element=element)
        return review.strip()

    if sentiment == "positive":
        template = random.choice(POSITIVE_TEMPLATES)
        adj = random.choice(POSITIVE_ADJS)
        verb = random.choice(POSITIVE_VERBS)
        extra = random.choice(POSITIVE_EXTRAS)
    else:
        template = random.choice(NEGATIVE_TEMPLATES)
        adj = random.choice(NEGATIVE_ADJS)
        verb = random.choice(NEGATIVE_VERBS)
        extra = random.choice(NEGATIVE_EXTRAS)

    element = random.choice(ELEMENTS)
    review = template.format(adj=adj, verb=verb, element=element, extra=extra)
    return review.strip()


def generate_rating(sentiment: str) -> float:
    """Generate a rating consistent with sentiment."""
    if sentiment == "positive":
        return round(np.clip(np.random.normal(7.5, 1.2), 5.0, 10.0), 1)
    else:
        return round(np.clip(np.random.normal(3.5, 1.2), 1.0, 6.0), 1)


def generate_dataset(n_samples: int = 2500, output_path: str = None) -> pd.DataFrame:
    """
    Generate a complete synthetic movie review dataset.

    Parameters
    ----------
    n_samples : int
        Total number of reviews to generate.
    output_path : str, optional
        Path to save the CSV. If None, saves to data/movie_reviews.csv.

    Returns
    -------
    pd.DataFrame
        Generated dataset.
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "movie_reviews.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentiments = (["positive"] * (n_samples // 2)) + (
        ["negative"] * (n_samples - n_samples // 2)
    )
    random.shuffle(sentiments)

    records = []
    for i, sentiment in enumerate(sentiments):
        genre = random.choice(GENRES)
        year = random.choices(
            range(1995, 2025),
            weights=[max(1, y - 1994) for y in range(1995, 2025)],
            k=1,
        )[0]

        # ~15% of reviews use ambiguous mixed-sentiment language
        is_ambiguous = random.random() < 0.15

        # ~4% of reviews have flipped labels (simulates annotator disagreement),
        # making the task realistically imperfect
        flip_label = random.random() < 0.04
        review_sentiment = (
            ("negative" if sentiment == "positive" else "positive")
            if flip_label
            else sentiment
        )

        records.append(
            {
                "review_id": i + 1,
                "review": generate_review(review_sentiment, ambiguous=is_ambiguous),
                "sentiment": sentiment,  # keep original label
                "rating": generate_rating(sentiment),
                "genre": genre,
                "release_year": year,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} reviews and saved to {output_path}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    print(f"Genre distribution:\n{df['genre'].value_counts()}")
    return df


if __name__ == "__main__":
    generate_dataset()
