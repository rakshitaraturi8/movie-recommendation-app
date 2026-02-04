# import libraries
import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


st.set_page_config(page_title="Movie Recommendation", layout="wide")



#API for movie poster
def fetch_poster(movie_title):
    api_key = "b67c4d83" 
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    
    try:
        data = requests.get(url).json()
        if data.get('Poster') and data.get('Poster') != 'N/A':
            return data.get('Poster')
    except:
        pass        
    return "https://cdn-icons-png.flaticon.com/512/864/864818.png"



#-------------Loading and cleaning data---------------

@st.cache_resource
def load_data():
    
    # datasets loading
    md = pd.read_csv('movies_metadata.csv', low_memory=False)
    ratings = pd.read_csv('ratings_small.csv')
    links = pd.read_csv('links_small.csv')

    # taking smaller size of dataset for speed
    md = md.iloc[0:20000]

    # missing plot of movies = empty string
    md['overview'] = md['overview'].fillna('')

    # cleaning messyy ID column
    md['id'] = pd.to_numeric(md['id'], errors = 'coerce')
    md = md.dropna(subset=['id'])
    md['id'] = md['id'].astype(int)

    # filter 'links' file to match IDs
    links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')

    return md, ratings, links

md, ratings, links = load_data()


#---------Content-based recommendation system---------

@st.cache_resource
def build_models(md, ratings):
    # TF-IDF matrix counting word frequency in every plot
    tfidf = TfidfVectorizer(stop_words = 'english')
    tfidf_matrix = tfidf.fit_transform(md['overview'])

    # computing similarity score (cosine similarity) it creates a matric which compares every movie to every ther movie
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    #create a reverse map of indices and movie titles
    indices = pd.Series(md.index, index=md['title']).drop_duplicates()


    # collaborative-based recommendation system
    print("Training Collaborative recommendation system....")

    # prepare data for the surprise library
    reader = Reader()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # train the SVD model 
    # it looks at a user A's history and predicts: "If user A watched this movie, hat would they rate it (1-5)?"
    svd = SVD()
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    return cosine_sim, indices, svd


with st.spinner("Please Wait...."):
    cosine_sim, indices, svd = build_models(md, ratings)

#---------hybrid recommendation system(combining both the recommendation system)---------------

def hybrid_recommendation(user_id, movie_title):
    
    if movie_title not in indices:
        return None
    
    idx = indices[movie_title] 

    sim_scores = list(enumerate(cosine_sim[idx])) #getting similarity scores of all movie with that movie 

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) #sorting on the basis of similarity scores

    sim_scores = sim_scores[1:26] #scores of 25 most similar movies
    movie_indices = [i[0] for i in sim_scores]

    movies = md.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'release_date', 'id']]

    #applying svd model to each of the 25 movies
    movies['est'] = movies['id'].apply(lambda x: svd.predict(user_id, x).est)

    movies = movies.sort_values('est', ascending=False) #sort by predicted rating

    return movies.head(15)


#-----------UI--------------

st.title("Hybrid Movie Recommendation System")

col1, col2 = st.columns(2)

with col1: 
    user_id = st.number_input("Enter User ID", min_value=1, max_value=600, value=1)

with col2:
    movie_list = md['title'].values
    selected_movie = st.selectbox("Select a Movie you like", options=["Toy Story", "The Dark Knight", "Iron Man"] + list(movie_list))

if st.button("Get Recommendations"):
    st.divider()
    results = hybrid_recommendation(user_id, selected_movie)

    if results is None:
        st.error("Movie not Found!")
    else:
        st.subheader("Top recommendations based on:")

        cols = st.columns(5)

        for i, (index, row) in enumerate(results.head(5).iterrows()):
            with cols[i]:
                poster_url = fetch_poster(row['title'])
                st.image(poster_url, use_container_width=True) 
                st.markdown(f"{row['title']}")
                st.write(f"Predicted Rating: ⭐ {row['est']:.2f} / 5")