from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests

app = Flask(__name__)

# ===================== CONFIG =====================
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

# ===================== LOAD DATA =====================
movies = pd.read_csv("movies.csv")
movies = movies[['title', 'overview']]
movies['overview'] = movies['overview'].fillna('')
movies['title_lower'] = movies['title'].str.lower()

# ===================== ML MODEL =====================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['title_lower']).drop_duplicates()

# ===================== HELPERS =====================
def get_poster(title):
    if not TMDB_API_KEY:
        return None

    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title
        }

        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        for result in data.get("results", []):
            poster_path = result.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except Exception as e:
        print("Poster error:", e)

    return None

# ===================== ROUTES =====================
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    error = ""

    if request.method == "POST":
        movie_input = request.form.get("movie", "").strip().lower()

        if movie_input in indices.index:
            recommendations = recommend(movie_input)
        else:
            error = "Movie not found. Try another title."

    return render_template(
        "index.html",
        recommendations=recommendations,
        error=error
    )

@app.route("/autocomplete")
def autocomplete():
    query = request.args.get("q", "").lower()

    if len(query) < 2:
        return jsonify([])

    matches = movies[movies['title_lower'].str.contains(query, na=False)]
    return jsonify(matches['title'].head(10).tolist())

# ===================== RUN =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
