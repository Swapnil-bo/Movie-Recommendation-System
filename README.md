# 🎬 CineMatch — Movie Recommendation System

A content-based movie recommender built with **Item-Based Collaborative Filtering** on the MovieLens Small dataset.

## Demo
> Select any movie → get 5 personalised recommendations instantly.

## How It Works
1. Loads and filters the MovieLens dataset (movies with >50 ratings, users with >10 ratings)
2. Builds a Movies × Users pivot table and converts it to a CSR sparse matrix
3. Computes cosine similarity between every pair of movies
4. Returns the top 5 most similar movies for any selected title

## Tech Stack
| Layer | Tools |
|---|---|
| Data | Pandas, NumPy, Scipy |
| ML | Scikit-learn (Cosine Similarity) |
| App | Streamlit |

## Run Locally
```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
[MovieLens Small](https://grouplens.org/datasets/movielens/latest/) — 100,000 ratings across 9,000 movies.

Hope y'all like it. 
