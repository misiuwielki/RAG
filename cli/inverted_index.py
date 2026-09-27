import pickle, os
from process_text import process_text, remove_stop_words_and_stem, load_stop_words
from utils import load_movies, Movie, MovieData

class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Movie] = {}
        self.stop_words = load_stop_words()

    def __add_document(self, doc_id: int, text: str):
        tokens: list[str] = remove_stop_words_and_stem(process_text(text), self.stop_words)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

    def get_document(self, term: str):
        ids = self.index.get(term, set())
        return sorted(ids)
    
    def build(self):
        movies: MovieData = load_movies()
        for movie in movies['movies']:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie['id']] = movie

    def save(self):
        os.makedirs("cache/", exist_ok=True)
        with open("cache/index.pkl", 'wb') as i:
            pickle.dump(self.index, i)
        with open("cache/docmap.pkl", 'wb') as d:
            pickle.dump(self.docmap, d)

    def load(self):
        if not os.path.isfile("cache/index.pkl") or not os.path.isfile("cache/docmap.pkl"):
            raise FileNotFoundError("at least one file is missing")
        with open("cache/index.pkl", 'rb') as i:
            self.index = pickle.load(i)
        with open("cache/docmap.pkl", 'rb') as d:
            self.docmap = pickle.load(d)
    def build_command(self):
        self.build()
        self.save()




    