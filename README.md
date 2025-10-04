
# 🎬 Movie Recommender System  

This repository contains a **Content-Based Movie Recommendation System** that suggests similar movies to users based on their input. It uses **Natural Language Processing (NLP)** techniques and **cosine similarity** for recommendations.  

---

## 📑 Table of Contents  

- [Introduction](#-introduction)  
- [Why We Need This Project](#-why-we-need-this-project)  
- [Types of Recommender Systems](#-types-of-recommender-systems)  
- [Project Flow](#-project-flow)  
- [Building the Project](#-building-the-project)  
- [Tech Stack](#-tech-stack)  
- [Summary](#-summary)  

---

## 📘 Introduction  

Recommender systems are widely used in modern applications like **Netflix, YouTube, and Amazon** to provide personalized suggestions.  
This project implements a **content-based recommendation system** where recommendations are based on the similarity of movie features such as genres, cast, crew, keywords, and overviews.  

---

## ❓ Why We Need This Project  

- Helps users quickly find relevant movies.  
- Enhances user experience with personalized recommendations.  
- Provides hands-on experience with **machine learning & NLP concepts**.  
- Demonstrates end-to-end project flow: **data → preprocessing → model → frontend → deployment**.  

---

## 📂 Types of Recommender Systems  

1. **Content-Based Filtering**  
   - Uses item metadata (genres, keywords, etc.).  
   - Example: If you like *Inception*, it recommends similar    sci-fi movies.  

2. **Collaborative Filtering**  
   - Uses user behavior (ratings, clicks).  
   - Example: “Users who liked X also liked Y.”  

3. **Hybrid Systems**  
   - Combines both content & collaborative filtering.  
   - Example: **YouTube recommendations**.  

👉 In this project, we are using a **Content-Based System**.  

---

## 🚀 Project Flow  

1. **Data** → Import `movies.csv` and `credits.csv`.  
2. **Preprocessing** → Merge datasets, keep relevant columns, create `tags` column.  
3. **Model Building** → Text preprocessing, vectorization, similarity calculation.  
4. **Frontend** → Build a user interface (Streamlit).  
5. **Deployment** → Deploy using Streamlit Cloud / Heroku.  

---

## 🛠️ Building the Project  

### 1. Load Data  
```python
movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')
```

### 2. Merge Data  
```python
movies = movies.merge(credits, on='id')
```

### 3. Select Important Columns  
- `id`  
- `title`  
- `genres`  
- `keywords`  
- `overview`  
- `cast`  
- `crew`  

### 4. Create New DataFrame  
```text
movie_id | title | tags
```
Where `tags = overview + genres + keywords + cast + crew`.  

### 5. Preprocessing  
- Remove extra spaces.  
- Lowercase conversion.  
- Stemming (reduce words to root).  
- Remove stop words.  

### 6. Vectorization  
- Use **Bag of Words** to represent text.  
- Example:  
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
```

### 7. Similarity Calculation  
- Use **Cosine Similarity**:  
```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
```

### 8. Recommendation Function  
```python
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
```

### 9. Deployment  
- Frontend built with **Streamlit**.  
- Deploy on **Streamlit Cloud / Heroku**.  

---

## ⚙️ Tech Stack  

- **Language:** Python 🐍  
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK  
- **Frontend:** Streamlit  
- **Deployment:** Streamlit Cloud / Heroku  

---

## 📊 Summary  

This project builds a **Content-Based Movie Recommendation System** using **NLP and cosine similarity**. It suggests movies based on textual metadata and follows a complete workflow:  

**Data → Preprocessing → Model → Frontend → Deployment**.  

It demonstrates the practical use of **machine learning in real-world applications** like movie recommendations.  

---
 
