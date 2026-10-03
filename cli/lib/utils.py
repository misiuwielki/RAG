import json
from typing import TypedDict, TYPE_CHECKING
if TYPE_CHECKING:
    from lib.inverted_index import InvertedIndex

class Movie(TypedDict):
    id: int
    title: str
    description: str

class MovieData(TypedDict):
    movies: list[Movie]

def load_movies() -> MovieData:
    with open("data/movies.json") as f:
            file = f.read()
            return json.loads(file)

def search_for_tokens(input: list[str], index: "InvertedIndex", list_length: int | None = None):
    search_result: list[Movie] = []
    for token in input:
        movie_ids = index.get_document(token)
        for movie_id in movie_ids:
            movie = index.docmap[movie_id]
            if movie not in search_result:
                search_result.append(movie)
            if list_length is not None and len(search_result) >= list_length:
                break
        if list_length is not None and len(search_result) >= list_length:
            break
    return search_result