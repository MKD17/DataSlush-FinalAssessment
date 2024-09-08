# DataSlush-FinalAssessment (Movie-Recommendation)

## Overview

This is a Movie Recommendation System built with Python and Streamlit. It leverages semantic search using pre-trained models from Hugging Face and cosine similarity to recommend movies based on user input. The system searches through a dataset of movies, considering titles, genres, and descriptions to return movies that match the query.

## Features
- **Semantic Search**: Utilizes a pre-trained model (`all-MiniLM-L6-v2`) from Hugging Face to generate embeddings for movie descriptions.
- **Cosine Similarity**: Finds the most similar movies based on user queries by calculating cosine similarity between query embeddings and movie embeddings.
- **Streamlit UI**: Provides a user-friendly interface for entering movie preferences and receiving recommendations.
- **Custom UI**: Includes basic styling for buttons, text input, and background color.

## Technologies Used
- **Streamlit**: Frontend interface for user interaction.
- **Sentence Transformers**: Pre-trained model for generating embeddings.
- **Scikit-learn**: Used for computing cosine similarity.
- **Pandas**: Used for data manipulation.

## Dataset
The dataset is a collection of movie titles, genres, descriptions, and other metadata stored in a `pickle` file and loaded directly into the application.

## How It Works
1. **Preprocessing**: The dataset is cleaned, and a text column is created by combining the movie title, genre, and description.
2. **Embeddings**: The pre-trained model encodes the text into embeddings.
3. **Cosine Similarity**: When a user enters a search query, its embedding is compared to the movie embeddings using cosine similarity, and the top results are returned.
4. **Streamlit UI**: Users can input movie descriptions (e.g., "action-packed thriller") and receive movie recommendations.

## How to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/MKD17/DataSlush-FinalAssessment.git
    cd DataSlush-FinalAssessment
    cd movie-recommendation-app
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4. Open the app in your browser at:
    ```
    http://localhost:8501
    ```

## Example

Simply type in your desired movie characteristics, such as `"action-packed thriller"` or `"heartfelt romantic comedy"`, and the app will return a list of the most relevant movies.

## Folder Structure
- **app.py**: The main script that runs the Streamlit application.
- **netflix_movies.pkl**: A pickled file containing the dataset of Netflix movies, including titles, genres, and descriptions.
- **embeddings.pkl**: A pickled file that stores precomputed embeddings and IDs for the movie dataset.
- **requirements.txt**: A file listing the Python dependencies required to run the project.
- **rawcode_withoutUI.ipynb**: This file contains a raw code which recommends the output without UI design.
- **urldataset_to_picklefile**: This file contains the code for conversion of URL dataset(.csv) link to a pickle(.pkl) file.
- **README.md**: This file, providing an overview and instructions for the project.
