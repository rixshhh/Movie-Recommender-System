import streamlit as st
from src.data_preprocessing import load_and_process_data
from src.recommender import build_similarity_matrix, load_similarity_matrix, recommend
import os

st.set_page_config(page_title="Movie Recommender System", layout='centered')

# Check if similarity already exists
if not os.path.exists("models/similarity.pkl"):
    st.write("🔄 Processing dataset... please wait")
    new_df = load_and_process_data("data/tmdb_5000_movies.csv", "data/tmdb_5000_credits.csv")
    similarity = build_similarity_matrix(new_df)
else:
    similarity, new_df = load_similarity_matrix()

st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Choose a movie:", new_df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie, similarity, new_df)
    st.subheader("Top 5 Recommendations:")
    for movie in recommendations:
        st.write("👉", movie)



