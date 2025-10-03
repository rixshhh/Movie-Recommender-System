import pandas as pd
import numpy as np
import ast
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert_cast(obj):
    L = []
    for idx, i in enumerate(ast.literal_eval(obj)):
        if idx < 3:
            L.append(i['name'])
    return L

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def stem(text):
    return " ".join([ps.stem(i) for i in text.split()])

def load_and_process_data(movies_path, credits_path):
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    movies = movies.merge(credits, on='title')

    movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
    movies.dropna(inplace=True)

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    # remove spaces
    for col in ['genres','keywords','cast','crew']:
        movies[col] = movies[col].apply(lambda x:[i.replace(" ","") for i in x])

    # combine
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id','title','tags']]

    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(stem)

    return new_df
