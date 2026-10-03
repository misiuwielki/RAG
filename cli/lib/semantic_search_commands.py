from lib.semantic_search import SemanticSearch, ChunkedSemanticSearch
from lib.utils import load_movies


def verify_model():
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")

def embed_text(text: str):
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    model = SemanticSearch()
    movies = load_movies()
    movies_list = [k for k in movies["movies"]]
    emb = model.load_or_create_embeddings(movies_list)
    print(f"Number of docs: {len(movies_list)}")
    print(f"Embeddings shape: {emb.shape[0]} vectors in {emb.shape[1]} dimensions")

def embed_query(query: str):
    model = SemanticSearch()
    emb = model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {emb[:3]}")
    print(f"Shape: {emb.shape}")

def search_command(query: str, limit:int = 5):
    model = SemanticSearch()
    movies = load_movies()["movies"]
    model.load_or_create_embeddings(movies)
    result_list = model.search(query, limit)
    for i in range(len(result_list)):
        result = result_list[i]
        print(f"{i + 1}. {result["title"]} ({result["score"]})\n{result["description"]}\n)")

def embed_chunks_command():
    movies = load_movies()["movies"]
    model = ChunkedSemanticSearch()
    embeddings = model.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked_command(query: str, limit:int = 5):
    movies = load_movies()["movies"]
    model = ChunkedSemanticSearch()
    model.load_or_create_chunk_embeddings(movies)
    result_list = model.search_chunks(query, limit)
    for i in range(len(result_list)):
        result = result_list[i]
        print(f"\n{i + 1}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")

