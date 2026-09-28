import pickle, os, math
from process_text import process_text, remove_stop_words_and_stem, load_stop_words
from utils import load_movies, Movie, MovieData, search_for_tokens
from collections import Counter

class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Movie] = {}
        self.stop_words = load_stop_words()
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}
        self.avg_length = 0.0

    def __add_document(self, doc_id: int, text: str):
        tokens: list[str] = remove_stop_words_and_stem(process_text(text), self.stop_words)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
            if not self.term_frequencies.get(doc_id):
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        length_sum = 0.0
        if len(self.doc_lengths) == 0:
            return length_sum
        for id in self.doc_lengths:
            length_sum += self.doc_lengths[id]
        return length_sum/len(self.doc_lengths)
        
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
        with open("cache/term_frequencies.pkl", 'wb') as t:
            pickle.dump(self.term_frequencies, t)
        with open("cache/doc_lengths.pkl", 'wb') as dl:
            pickle.dump(self.doc_lengths, dl)

    def load(self):
        if (not os.path.isfile("cache/index.pkl") or not os.path.isfile("cache/docmap.pkl")
            or not os.path.isfile("cache/term_frequencies.pkl") or not os.path.isfile("cache/doc_lengths.pkl")):
            raise FileNotFoundError("at least one file is missing, use 'build' command")
        with open("cache/index.pkl", 'rb') as i:
            self.index = pickle.load(i)
        with open("cache/docmap.pkl", 'rb') as d:
            self.docmap = pickle.load(d)
        with open("cache/term_frequencies.pkl", 'rb') as t:
            self.term_frequencies = pickle.load(t)
        with open("cache/doc_lengths.pkl", 'rb') as dl:
                self.doc_lengths = pickle.load(dl)

    def build_command(self):
        self.build()
        self.save()

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

    def get_bm25_idf(self, term: str) -> float:
        search_result = search_for_tokens([term], self)
        bm25idf = math.log((len(self.docmap) - len(search_result) + 0.5) / (len(search_result) + 0.5) + 1)
        return bm25idf

    def get_bm25_tf(self, doc_id: int, term: str, k1:float=1.5, b:float=0.75):
        if self.avg_length == 0.0:
            self.avg_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.avg_length)
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1) / (tf + k1 * length_norm))

    def bm25(self, doc_id: int, term: str):
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit:int):
        tokens = remove_stop_words_and_stem(process_text(query), self.stop_words)
        scores: dict[int, float] = {}
        for token in tokens:
            idf = self.get_bm25_idf(token)
            if token in self.index:
                for doc_id in self.index[token]:
                    if scores.get(doc_id) == None:
                        scores[doc_id] = 0.0
                    scores[doc_id] += (self.get_bm25_tf(doc_id, token) * idf)
        return dict(sorted(scores.items(), reverse=True, key=lambda item:item[1])[:limit])




    