import pandas as pd
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#Applying CSS
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f5;
    }
    .stApp {
        background-color: #f0f0f5;
    }
    h1 {
        color: #4CAF50;
        font-size: 2.5rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    .stTextInput>div>div>input {
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True
)

#Loading the dataset from the pickle file
with open('netflix_movies.pkl', 'rb') as file:
    df = pickle.load(file)

#Function to prepare the text column
def prepare_text_column(df):
    required_columns = ['title', 'genres', 'description']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from DataFrame")
    
    #Combining columns into one text column
    df['text'] = df['title'].astype(str) + ' ' + df['genres'].astype(str) + ' ' + df['description'].astype(str)
    df['text'] = df['text'].str.lower().str.replace('[^\w\s]', '', regex=True)
    return df

df = prepare_text_column(df)

#Initializing the SentenceTransformer pre-trained model all-Mini-L6-v2 from Huggingface
model = SentenceTransformer('all-MiniLM-L6-v2')

#Saved embeddings and IDs
def save_embeddings_and_ids(df, model, filename='embeddings.pkl'):
    if 'text' not in df.columns:
        raise ValueError("DataFrame does not have a 'text' column")
    
    embeddings = model.encode(df['text'].tolist())
    ids = df['id'].tolist()
    with open(filename, 'wb') as file:
        pickle.dump((embeddings, ids), file)

#Loading those embeddings and IDs
def load_embeddings_and_ids(filename='embeddings.pkl'):
    with open(filename, 'rb') as file:
        embeddings, ids = pickle.load(file)
    return np.array(embeddings), ids

try:
    all_embeddings, all_ids = load_embeddings_and_ids()
except FileNotFoundError:
    save_embeddings_and_ids(df, model)
    all_embeddings, all_ids = load_embeddings_and_ids()

#Initialized ChromaDB vector store and loaded the collection
client = chromadb.Client()
collection_name = 'movie_titles'
collection = client.get_or_create_collection(name=collection_name)

#Function to perform the query with cosine similarity
def find_similar_movies(query, top_k=5):
    #Encoded the user query
    query_embedding = model.encode([query])
    
    #Computed cosine similarity between the query embedding and all document embeddings
    similarities = cosine_similarity(query_embedding, all_embeddings)
    
    #Getting the values of the top-k number of most similar items
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    
    #Retrieved the IDs of the top-k number of most similar movies
    top_ids = [all_ids[i] for i in top_indices]
    
    #Returning movie titles based on IDs
    movie_titles = df.loc[df['id'].isin(top_ids), 'title'].tolist()
    return movie_titles

#Setting up Streamlit app
st.markdown("<h1>Movie Recommendation System</h1>", unsafe_allow_html=True)
st.write("Enter movie characteristics to find similar movies:")

#User input for movie characteristics or description
user_query = st.text_input("Describe the movie you're looking for (e.g., 'heartfelt romantic comedy')")

#Button "Find Movies" to trigger search
if st.button("Find Movies") or user_query:
    if user_query:
        #Finding similar movies
        similar_movies = find_similar_movies(user_query)
        
        if similar_movies:
            st.write("### Top Recommendations:")
            for movie in similar_movies:
                st.write(f"- {movie}")
        else:
            st.write("No similar movies found.")
    else:
        st.write("Please enter a movie description.")
