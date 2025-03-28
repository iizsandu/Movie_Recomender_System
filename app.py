import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

def load_data():
    # load movie data and compute similarity matrix
    movies_df = pickle.load(open('movies.pkl','rb'))
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return movies_df, similarity

def recommend(movie, movies_df, similarity):
    # recommend movies based on cosine similarity
    movie_index = movies_df[movies_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key = lambda x: x[1])[1:6]

    recommended_movies = [movies_df.iloc[i[0]].title for i in movies_list]
    return recommended_movies

def fetch_poster(movie_id):
    api_key = "42748772d2f1aa6fd0f36112309ffebb"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"

    response = requests.get(url)
    data = response.json()

    if 'poster_path' in data:
        return f"https://image.tmdb.org/t/p/w500/{data['poster_path']}"
    else:
        return f"poster not found"

if __name__ == '__main__':
    st.title('🎬 Movie Recommendation System')

    # Load data
    movies_df, similarity = load_data()
    movies_list = movies_df['title'].values

    selected_movie = st.selectbox('🎥 Select a Movie from the dropdown', movies_list)

    if st.button('🔍 Recommend'):
        recommendations = recommend(selected_movie, movies_df, similarity)
        st.write("## 🎞 Recommended Movies:")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        cols = [col1, col2, col3, col4, col5]
        
        for idx, movie in enumerate(recommendations):
            movie_id = movies_df[movies_df['title'] == movie]['id'].values[0]  
            poster_url = fetch_poster(movie_id)
            
            with cols[idx]:
                st.image(poster_url, use_container_width=True, caption=movie)