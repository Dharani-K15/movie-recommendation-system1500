from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies = pd.read_csv("movies.csv")
movies = movies[['title', 'overview']]
movies['overview'] = movies['overview'].fillna('')

# ML logic
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend(movie_title):
    idx = indices[movie_title]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:6]   # top 5 recommendations
    movie_indices = [i[0] for i in scores]

    return movies['title'].iloc[movie_indices].tolist()

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    error = ""

    if request.method == "POST":
        movie_name = request.form["movie"]

        if movie_name in indices:
            recommendations = recommend(movie_name)
        else:
            error = "Movie not found."

    return render_template(
        "index.html",
        recommendations=recommendations,
        error=error
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

