import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import os

# ------------------ DOWNLOAD LARGE FILE ------------------

SIMILARITY_FILE = "similarity.pkl"

GDRIVE_FILE_ID = "1wK_XlGY7trmEbmu7Wp_sB5qzr-CP06yC"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

def download_similarity():
    if not os.path.exists(SIMILARITY_FILE):
        with st.spinner("Downloading recommendation model... ⏳"):
            response = requests.get(GDRIVE_URL, stream=True)
            response.raise_for_status()
            with open(SIMILARITY_FILE, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

# Download only once
download_similarity()

# ------------------ LOAD DATA ------------------

similarity = pickle.load(open(SIMILARITY_FILE, "rb"))
df = pickle.load(open("movies_dict.pkl", "rb"))
df = pd.DataFrame(df)

# ------------------ FUNCTIONS ------------------

def fetch_poster(id):
    url = f"https://api.themoviedb.org/3/movie/{id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"
    except requests.exceptions.RequestException:
        return "https://via.placeholder.com/500x750?text=No+Image"


def recommend(movie):
    index = df[df["title"] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    rec_movies = []
    rec_posters = []

    for i in movie_list:
        movie_id = df.iloc[i[0]].id
        rec_movies.append(df.iloc[i[0]].title)
        rec_posters.append(fetch_poster(movie_id))

    return rec_movies, rec_posters

# ------------------ STREAMLIT UI ------------------

st.header("🎬 Movie Recommender System")

select_movie_name = st.selectbox(
    "Type or select a movie from the dropdown",
    df["title"].values
)

if st.button("Recommend"):
    rec_movies, rec_posters = recommend(select_movie_name)

    cols = st.columns(5)
    for col, movie, poster in zip(cols, rec_movies, rec_posters):
        with col:
            st.text(movie)
            st.image(poster)
