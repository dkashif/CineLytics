import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import ast

st.title("🎬 Cinelytics")
st.write("Predict a movie's box office revenue based on its features!")


# ----- Model Loading -----
# If you haven't saved the model yet, you can re-train quickly here
@st.cache_data
def load_data_and_model():
    movies = pd.read_csv("tmdb_5000_movies.csv", low_memory=False)
    movies = movies[(movies["budget"] > 0) & (movies["revenue"] > 0)]
    movies = movies.dropna(subset=["runtime", "popularity", "vote_average"])

    # Extract genres
    all_genres = set()
    for row in movies["genres"]:
        try:
            genres = [g["name"] for g in ast.literal_eval(row)]
            all_genres.update(genres)
        except:
            pass
    for genre in all_genres:
        movies[f"genre_{genre}"] = 0
    for i, row in movies.iterrows():
        try:
            genres = [g["name"] for g in ast.literal_eval(row["genres"])]
            for g in genres:
                movies.loc[i, f"genre_{g}"] = 1
        except:
            pass

    features = ["budget", "popularity", "runtime", "vote_average"] + [
        col for col in movies.columns if col.startswith("genre_")
    ]
    X = movies[features]
    y = movies["revenue"]

    model = LinearRegression()
    model.fit(X, y)
    # return the trained model, the set of genres and the ordered feature list
    features = ["budget", "popularity", "runtime", "vote_average"] + [
        col for col in movies.columns if col.startswith("genre_")
    ]
    return model, all_genres, features


model, all_genres, features = load_data_and_model()

# ----- User Inputs -----
budget = st.number_input("Budget ($)", min_value=0, step=1000000, value=100000000)
runtime = st.number_input("Runtime (minutes)", min_value=30, max_value=300, value=120)
popularity = st.number_input("Popularity", min_value=0.0, value=10.0)
vote_avg = st.slider("Average Rating", 0.0, 10.0, 7.5)
selected_genres = st.multiselect("Genres", sorted(list(all_genres)))

# ----- Prepare Input for Prediction -----
# Build input DataFrame using the exact feature names & order used during training
input_dict = {f: 0 for f in features}
input_dict["budget"] = budget
input_dict["popularity"] = popularity
input_dict["runtime"] = runtime
input_dict["vote_average"] = vote_avg
# Set genre flags according to user selection (only for genres present in features)
for genre in all_genres:
    col = f"genre_{genre}"
    if col in features:
        input_dict[col] = 1 if genre in selected_genres else 0

input_data = pd.DataFrame([input_dict], columns=features)

# ----- Predict -----
if st.button("Predict Revenue"):
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Estimated Box Office Revenue: **${prediction:,.0f}**")
