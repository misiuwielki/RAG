import json
from typing import TypedDict

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