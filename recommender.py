"""
Phase 2: The Recommendation Engine
Movie Recommendation System — Item-Based Collaborative Filtering
Depends on: data_pipeline.py (Phase 1)
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_pipeline import build_pipeline


# ── Build the pipeline once at import time ─────────────────────────────────────
matrix, pivot, movie_index = build_pipeline()


# ── Similarity Matrix ──────────────────────────────────────────────────────────
def build_similarity_matrix():
    """
    Compute cosine similarity between every pair of movies.

    - Input  : CSR matrix  (movies × users)
    - Output : 2-D array   (movies × movies)  values in [0, 1]

    cosine_similarity works directly on sparse matrices — no need to
    convert to dense first, which keeps memory usage low.
    """
    return cosine_similarity(matrix)


similarity_matrix = build_similarity_matrix()


# ── Core Recommendation Function ───────────────────────────────────────────────
def get_recommendations(movie_name: str, top_n: int = 5) -> list[str] | str:
    """
    Return the top-N most similar movies for a given title.

    Parameters
    ----------
    movie_name : str   Exact movie title as it appears in the dataset.
    top_n      : int   Number of recommendations to return (default 5).

    Returns
    -------
    list[str]  Titles of the top-N recommended movies.
    str        Friendly error message if the movie is not found.
    """
    # ── Guard: movie not in database ───────────────────────────────────────────
    if movie_name not in movie_index:
        return (
            f"❌ '{movie_name}' was not found in the database.\n"
            f"Tip: Make sure the title matches exactly, including the year — "
            f"e.g. 'Toy Story (1995)'."
        )

    # ── Find the row index of this movie ──────────────────────────────────────
    movie_idx = movie_index.get_loc(movie_name)

    # ── Grab its similarity scores against all other movies ───────────────────
    scores = similarity_matrix[movie_idx]

    # ── Sort descending; skip index 0 (the movie itself, score = 1.0) ─────────
    similar_indices = np.argsort(scores)[::-1][1 : top_n + 1]

    # ── Map indices back to titles ────────────────────────────────────────────
    recommendations = movie_index[similar_indices].tolist()

    return recommendations


# ── Helper: list all valid movie titles ───────────────────────────────────────
def get_all_titles() -> list[str]:
    """Return every movie title in the filtered dataset (sorted A→Z)."""
    return sorted(movie_index.tolist())


# ── Quick Test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_movies = [
        "Toy Story (1995)",
        "Fargo (1996)",
        "This Movie Does Not Exist (2099)",   # error-handling test
    ]

    for title in test_movies:
        print(f"\n🎬 Recommendations for: {title}")
        result = get_recommendations(title)

        if isinstance(result, list):
            for i, rec in enumerate(result, 1):
                print(f"  {i}. {rec}")
        else:
            print(f"  {result}")