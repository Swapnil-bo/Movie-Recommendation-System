"""
Phase 3 (v2): Streamlit Application — upgraded with real-time search
Movie Recommendation System — Item-Based Collaborative Filtering
Depends on: recommender.py (Phase 2) → data_pipeline.py (Phase 1)
"""

import streamlit as st
from recommender import get_recommendations, get_all_titles

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=Outfit:wght@300;400;500;600&display=swap');

/* ── CSS Variables ── */
:root {
    --bg:        #090909;
    --surface:   #111111;
    --surface2:  #181818;
    --border:    #222222;
    --border2:   #2e2e2e;
    --gold:      #d4a843;
    --gold-dim:  #8a6a24;
    --text:      #ede9e0;
    --text-muted:#666666;
    --text-dim:  #999999;
    --red:       #e05555;
    --red-bg:    #1a0d0d;
    --red-border:#4a1818;
    --green:     #4caf82;
}

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    max-width: 800px;
}

/* ── Hero ── */
.hero-wrap {
    margin-bottom: 2.8rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid var(--border);
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 500;
    margin-bottom: 1rem;
}
.hero-eyebrow::before {
    content: "";
    display: inline-block;
    width: 20px;
    height: 1px;
    background: var(--gold);
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.6rem;
    font-weight: 700;
    line-height: 1.05;
    color: var(--text);
    margin-bottom: 0.8rem;
    letter-spacing: -0.01em;
}
.hero-title em {
    font-style: italic;
    font-weight: 400;
    color: var(--gold);
}
.hero-sub {
    font-size: 0.92rem;
    color: var(--text-muted);
    font-weight: 300;
    line-height: 1.6;
    max-width: 480px;
}

/* ── Stats Bar ── */
.stats-bar {
    display: flex;
    gap: 2.5rem;
    margin-top: 1.8rem;
}
.stat-item { display: flex; flex-direction: column; gap: 0.15rem; }
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.stat-label {
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* ── Section label ── */
.search-label {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
    font-weight: 500;
}

/* ── Text input ── */
input[type="text"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
input[type="text"]:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(212, 168, 67, 0.1) !important;
}
input[type="text"]::placeholder { color: var(--text-muted) !important; }

/* ── Match counter pill ── */
.match-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 0.22rem 0.75rem;
    font-size: 0.73rem;
    color: var(--text-dim);
    margin: 0.5rem 0 0.6rem;
}
.match-pill .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    flex-shrink: 0;
}
.match-pill.no-match .dot { background: var(--red); }

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-baseweb="menu"] {
    background-color: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
}
[role="option"] {
    background-color: var(--surface2) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
}
[role="option"]:hover { background-color: #252525 !important; }

/* ── No results ── */
.no-results {
    background: var(--surface);
    border: 1px dashed var(--border2);
    border-radius: 8px;
    padding: 1.3rem;
    text-align: center;
    color: var(--text-muted);
    font-size: 0.88rem;
    margin-top: 0.5rem;
    line-height: 1.6;
}
.no-results strong { color: var(--text-dim); }

/* ── Keyboard hint ── */
.hint-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-top: 0.55rem;
    font-size: 0.71rem;
    color: var(--text-muted);
}
.kbd {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    font-size: 0.67rem;
    color: var(--text-dim);
    font-family: monospace;
}

/* ── Button ── */
.stButton > button {
    background: var(--gold);
    color: #080808;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: none;
    border-radius: 8px;
    padding: 0.68rem 2rem;
    margin-top: 1.1rem;
    width: 100%;
    cursor: pointer;
    transition: background 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease;
}
.stButton > button:hover {
    background: #c49a38;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(212, 168, 67, 0.18);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button:disabled {
    background: var(--surface2) !important;
    color: var(--text-muted) !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Section divider ── */
.section-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2.5rem 0 1.8rem;
}
.section-divider .line { flex: 1; height: 1px; background: var(--border); }
.section-divider .label {
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-muted);
    white-space: nowrap;
}

/* ── Selected movie badge ── */
.selected-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(212, 168, 67, 0.07);
    border: 1px solid var(--gold-dim);
    border-radius: 6px;
    padding: 0.45rem 1rem;
    margin-bottom: 1.4rem;
    font-size: 0.85rem;
    color: var(--gold);
    font-weight: 500;
}

/* ── Movie card ── */
.movie-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease, transform 0.2s ease;
    animation: fadeSlideIn 0.35s ease both;
}
.movie-card::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--gold);
    opacity: 0;
    transition: opacity 0.2s ease;
}
.movie-card:hover { border-color: var(--border2); transform: translateX(5px); }
.movie-card:hover::before { opacity: 1; }
.movie-card:nth-child(1) { animation-delay: 0.00s; }
.movie-card:nth-child(2) { animation-delay: 0.07s; }
.movie-card:nth-child(3) { animation-delay: 0.14s; }
.movie-card:nth-child(4) { animation-delay: 0.21s; }
.movie-card:nth-child(5) { animation-delay: 0.28s; }

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0);    }
}

.card-rank {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--border2);
    min-width: 2.4rem;
    text-align: center;
    line-height: 1;
    user-select: none;
}
.card-body { flex: 1; }
.card-title {
    font-size: 0.97rem;
    font-weight: 500;
    color: var(--text);
    line-height: 1.3;
    margin-bottom: 0.18rem;
}
.card-year {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 300;
}
.card-arrow {
    font-size: 0.9rem;
    color: var(--border2);
    transition: color 0.2s ease, transform 0.2s ease;
}
.movie-card:hover .card-arrow { color: var(--gold); transform: translateX(3px); }

/* ── Error box ── */
.error-box {
    background: var(--red-bg);
    border: 1px solid var(--red-border);
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    color: #e07070;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Footer ── */
.footer {
    margin-top: 5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
}
.footer-brand {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    color: var(--gold);
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)


# ── Cached Data Loader ─────────────────────────────────────────────────────────
@st.cache_data
def load_titles() -> list[str]:
    """Load and cache all movie titles — runs once per session."""
    return get_all_titles()


# ── Helper: split title and year ───────────────────────────────────────────────
def split_title_year(full_title: str) -> tuple[str, str]:
    """'Toy Story (1995)'  →  ('Toy Story', '1995')"""
    if full_title.endswith(")") and "(" in full_title:
        name, year = full_title.rsplit("(", 1)
        return name.strip(), year.rstrip(")")
    return full_title, ""


# ── Load data ──────────────────────────────────────────────────────────────────
all_titles   = load_titles()
total_movies = len(all_titles)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-eyebrow">Collaborative Filtering</div>
    <div class="hero-title">Your next<br><em>great watch</em><br>awaits.</div>
    <div class="hero-sub">
        Tell us a film you love and we'll find five more you'll probably obsess over —
        powered by item-based cosine similarity on real viewer ratings.
    </div>
    <div class="stats-bar">
        <div class="stat-item">
            <div class="stat-value">{total_movies:,}</div>
            <div class="stat-label">Films indexed</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">100K+</div>
            <div class="stat-label">Ratings analysed</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">Top 5</div>
            <div class="stat-label">Recommendations</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Search Bar ─────────────────────────────────────────────────────────────────
st.markdown('<div class="search-label">🔍 &nbsp; Search for a movie</div>', unsafe_allow_html=True)

query = st.text_input(
    label="",
    placeholder="Start typing — e.g. Pulp Fiction, The Matrix, Toy Story…",
    label_visibility="collapsed",
    key="search_query",
)

# ── Real-time filter ───────────────────────────────────────────────────────────
q = query.strip().lower()
filtered = [t for t in all_titles if q in t.lower()] if q else all_titles
match_count = len(filtered)

# ── Match counter pill (only shown when user has typed something) ──────────────
if q:
    pill_class = "match-pill" if match_count > 0 else "match-pill no-match"
    pill_text  = f"{match_count} match{'es' if match_count != 1 else ''} found" if match_count > 0 else "No matches"
    st.markdown(f"""
    <div class="{pill_class}">
        <span class="dot"></span>{pill_text}
    </div>
    """, unsafe_allow_html=True)


# ── Selectbox or no-results message ───────────────────────────────────────────
selected_movie = None

if match_count == 0:
    st.markdown(f"""
    <div class="no-results">
        No films matching <strong>"{query}"</strong> were found.<br>
        Try a shorter keyword or check the spelling.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(
        '<div class="search-label" style="margin-top:0.7rem">↳ &nbsp; Select from results</div>',
        unsafe_allow_html=True,
    )
    selected_movie = st.selectbox(
        label="",
        options=filtered,
        label_visibility="collapsed",
        key="movie_select",
    )
    st.markdown("""
    <div class="hint-row">
        <span class="kbd">↑</span><span class="kbd">↓</span> to navigate &nbsp;·&nbsp;
        <span class="kbd">Enter</span> to confirm &nbsp;·&nbsp;
        <span class="kbd">Esc</span> to close
    </div>
    """, unsafe_allow_html=True)


# ── Recommend Button ───────────────────────────────────────────────────────────
recommend_clicked = st.button(
    "✦  Get Recommendations",
    disabled=(selected_movie is None),
)


# ── Results ────────────────────────────────────────────────────────────────────
if recommend_clicked and selected_movie:

    name, year = split_title_year(selected_movie)
    year_html  = f"&nbsp;<span style='opacity:0.45; font-weight:300'>({year})</span>" if year else ""

    st.markdown(f"""
    <div class="section-divider">
        <div class="line"></div>
        <div class="label">Your recommendations</div>
        <div class="line"></div>
    </div>
    <div class="selected-badge">
        🎬 &nbsp; Because you liked &nbsp;<strong>{name}</strong>{year_html}
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Calculating similarity scores…"):
        results = get_recommendations(selected_movie, top_n=5)

    if isinstance(results, str):
        st.markdown(
            f'<div class="error-box">⚠️ &nbsp;{results}</div>',
            unsafe_allow_html=True,
        )
    else:
        for rank, title in enumerate(results, start=1):
            card_name, card_year = split_title_year(title)
            year_div = f"<div class='card-year'>{card_year}</div>" if card_year else ""
            st.markdown(f"""
            <div class="movie-card">
                <div class="card-rank">{rank:02d}</div>
                <div class="card-body">
                    <div class="card-title">{card_name}</div>
                    {year_div}
                </div>
                <div class="card-arrow">→</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1.2rem; font-size:0.73rem; color:#3a3a3a;
                    text-align:center; letter-spacing:0.06em;">
            Ranked by cosine similarity score
        </div>
        """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-brand">CineMatch</div>
    <div>MovieLens Small &nbsp;·&nbsp; Scikit-learn &nbsp;·&nbsp; Streamlit</div>
</div>
""", unsafe_allow_html=True)