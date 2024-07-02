import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Load movies data
movies_data = pd.read_csv(r"C:\Users\user\Documents\TAB\movies.csv")  # Replace with your path

# Selected features
selected_features = ["genres", "keywords", "tagline", "cast", "director"]

# Fill null values
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna("")

# Combine features
combined_features = movies_data["genres"] + " " + movies_data["keywords"] + " " + movies_data[
    "tagline"
] + " " + movies_data["cast"] + " " + movies_data["director"]

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)


def recommend_movies(movie_name):
    list_of_all_titles = movies_data["title"].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if not find_close_match:
        return "Movie not found!"
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]["index"].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    recommendations = []
    for movie in sorted_similar_movies[:6]:  # Limit recommendations
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]["title"].values[0]
        recommendations.append(title_from_index)
    return recommendations


# Streamlit App
movie_name = st.text_input("Enter movie name")
if st.button("Get Recommendation"):  # Added button
    if movie_name:
        recommendations = recommend_movies(movie_name)
        if isinstance(recommendations, str):
            st.write(recommendations)
        else:
            st.header("Recommendations")
            for movie in recommendations:
                st.write(movie)
