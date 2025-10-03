import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

def build_similarity_matrix(new_df, save_path="models/similarity.pkl"):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    # Save similarity for reuse
    with open(save_path, "wb") as f:
        pickle.dump((similarity, new_df), f)
    return similarity

def load_similarity_matrix(path="models/similarity.pkl"):
    with open(path, "rb") as f:
        similarity, new_df = pickle.load(f)
    return similarity, new_df


def recommend(movie, similarity, new_df, top_n=5):
    if movie not in new_df['title'].values:
        return ["Movie not found in database"]
    
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:top_n+1]

    return [new_df.iloc[i[0]].title for i in movies_list]
