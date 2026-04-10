from src.embeddings.embedding_initialization import load_embeddings_model, encode_texts, normalize_embeddings
from src.embeddings.load_embeddings import load_embeddings
from src.indexing.initialize_indexes import load_index, index_file
from src.data.load import load_data
from src.data.dummyUser import load_user_data
from src.util.util import get_indices_for_user
from src.service.app_context import app_context

#initializing model

def initialize_services():
    print("loading model....")
    app_context.model = load_embeddings_model()

    print("loading embeddings...")
    app_context.movie_ids,app_context.embeddings = load_embeddings()

    print("loading movie data...")
    app_context.movies = load_data()
    app_context.movies_by_id = app_context.movies.set_index("movie_id")

    print("loading FAISS index...")
    app_context.index = load_index(index_file)


    print("All services initialized...")
