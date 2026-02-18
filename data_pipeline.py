"""
Phase 1: Data Pipeline & Noise Filtering
Movie Recommendation System — Item-Based Collaborative Filtering
Dataset: MovieLens Small (data/movies.csv, data/ratings.csv)
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


# ── Constants ──────────────────────────────────────────────────────────────────
MIN_MOVIE_RATINGS = 50   # drop movies rated fewer than this many times
MIN_USER_RATINGS  = 10   # drop users who rated fewer than this many movies
DATA_DIR          = "data/"


# ── Loaders ────────────────────────────────────────────────────────────────────
def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load movies and ratings CSVs from disk."""
    movies  = pd.read_csv(f"{DATA_DIR}movies.csv")
    ratings = pd.read_csv(f"{DATA_DIR}ratings.csv")
    return movies, ratings


# ── Noise Filtering ────────────────────────────────────────────────────────────
def filter_noise(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Remove statistical noise by enforcing minimum interaction thresholds.

    Keeps only:
      - Movies with > MIN_MOVIE_RATINGS ratings  (removes obscure/cold-start movies)
      - Users  with > MIN_USER_RATINGS  ratings  (removes casual/sparse raters)
    """
    # Step 1 — filter by movie popularity
    movie_counts   = ratings["movieId"].value_counts()
    popular_movies = movie_counts[movie_counts > MIN_MOVIE_RATINGS].index
    ratings        = ratings[ratings["movieId"].isin(popular_movies)]

    # Step 2 — filter by user activity
    user_counts   = ratings["userId"].value_counts()
    active_users  = user_counts[user_counts > MIN_USER_RATINGS].index
    ratings       = ratings[ratings["userId"].isin(active_users)]

    return ratings


# ── Pivot Table ────────────────────────────────────────────────────────────────
def build_pivot_table(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a Movies × Users pivot table.

    Rows    → movie titles
    Columns → user IDs
    Values  → star ratings (0 where no rating exists)
    """
    # Merge to get human-readable titles
    merged = ratings.merge(movies[["movieId", "title"]], on="movieId")

    pivot = merged.pivot_table(
        index="title",
        columns="userId",
        values="rating",
        aggfunc="mean",   # handles duplicate (user, movie) pairs gracefully
    ).fillna(0)

    return pivot


# ── CSR Matrix ─────────────────────────────────────────────────────────────────
def build_csr_matrix(pivot: pd.DataFrame) -> csr_matrix:
    """
    Convert the pivot table to a Scipy CSR (Compressed Sparse Row) matrix.

    CSR format stores only non-zero values → large memory savings and
    faster cosine-similarity computation in Phase 2.
    """
    return csr_matrix(pivot.values)


# ── Orchestrator ───────────────────────────────────────────────────────────────
def build_pipeline() -> tuple[csr_matrix, pd.DataFrame, pd.Index]:
    """
    Run the full Phase-1 pipeline and return everything Phase 2 needs.

    Returns
    -------
    matrix      : CSR matrix  (movies × users)
    pivot       : DataFrame   (same data, keeps index/column labels)
    movie_index : Index       (movie titles aligned with matrix rows)
    """
    movies, ratings = load_raw_data()

    print(f"[raw]      movies={len(movies):,}  ratings={len(ratings):,}")

    filtered = filter_noise(ratings)

    print(
        f"[filtered] ratings={len(filtered):,}  "
        f"unique_movies={filtered['movieId'].nunique():,}  "
        f"unique_users={filtered['userId'].nunique():,}"
    )

    pivot  = build_pivot_table(filtered, movies)
    matrix = build_csr_matrix(pivot)

    print(
        f"[matrix]   shape={matrix.shape}  "
        f"stored_elements={matrix.nnz:,}  "
        f"sparsity={1 - matrix.nnz / np.prod(matrix.shape):.2%}"
    )

    return matrix, pivot, pivot.index


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    matrix, pivot, movie_index = build_pipeline()

    print("\nSample movie titles in filtered dataset:")
    print(movie_index[:10].tolist())