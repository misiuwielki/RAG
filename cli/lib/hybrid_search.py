import os
from lib.inverted_index import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from lib.utils import Movie, load_movies
from typing import TypedDict

class HybridSearchResult(TypedDict):
    id: int
    bm25: float
    semantic: float
    hybrid: float
    title: str
    description: str

class RRFSearchResult(TypedDict):
    id: int
    bm25: int | None
    semantic: int | None
    rrf: float
    title: str
    description: str


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5) -> list[HybridSearchResult]:
        bm25_results = self._bm25_search(query, limit * 50)
        semantic_results = self.semantic_search.search_chunks(query, limit * 50)
        bm25_results_normalized = dict(zip(list(bm25_results.keys()), normalize_scores(list(bm25_results.values()))))
        s_scores = [s["score"] for s in semantic_results]
        s_dict:dict[int, float] = {}
        for s, score in zip(semantic_results, normalize_scores(s_scores)):
            s_dict[s["id"]] = score
        final_results: list[HybridSearchResult] = []
        for id in set(bm25_results_normalized) | set(s_dict):
            bm25 = bm25_results_normalized.get(id, 0.0)
            semantic = s_dict.get(id, 0.0)
            hybrid = hybrid_score(bm25, semantic, alpha)
            h_result:HybridSearchResult = {
                "id": id,
                "bm25": bm25,
                "semantic": semantic,
                "hybrid": hybrid,
                "title": self.idx.docmap[id]["title"],
                "description": self.idx.docmap[id]["description"][:100]
            }
            final_results.append(h_result)
        final_sorted = sorted(final_results, key=lambda d:d["hybrid"], reverse=True)
        return final_sorted
            

    def rrf_search(self, query: str, k: int, limit: int = 10, debug: bool = False) -> list[RRFSearchResult]:
        bm25_results = self._bm25_search(query, limit * 50)
        semantic_results = self.semantic_search.search_chunks(query, limit * 50)
        if debug:
            print(f"Keyword search results: '{", ".join([self.idx.docmap[t]['title'] for t in bm25_results])}'")
            print(f"Semantic search results: '{", ".join(r['title'] for r in semantic_results)}'")
        final_results: dict[int, RRFSearchResult] = {}
        for rank, id in enumerate(bm25_results, start=1):
            rrf = rrf_score(rank, k)
            rrf_result: RRFSearchResult = {
                "id": id,
                "bm25": rank,
                "semantic": None,
                "rrf": rrf,
                "title": self.idx.docmap[id]["title"],
                "description": self.idx.docmap[id]["description"][:100]
            }
            final_results[id] = rrf_result
        for rank, result in enumerate(semantic_results, start=1):
            id = result["id"]
            rrf = rrf_score(rank, k)
            rrf_res = final_results.get(id)
            if rrf_res is None:
                final_results[id] = {
                    "id": id,
                    "bm25": None,
                    "semantic": rank,
                    "rrf": rrf,
                    "title": self.idx.docmap[id]["title"],
                    "description": self.idx.docmap[id]["description"][:100]
                }
            else:
                final_results[id]["semantic"] = rank
                final_results[id]["rrf"] += rrf
        return sorted(final_results.values(), key=lambda r:r["rrf"], reverse=True)



def normalize_scores(scores:list[float]) -> list[float]:
    if not scores:
        return []
    ordered = sorted(scores)
    min_v = ordered[0]
    max_v = ordered[-1]
    if min_v == max_v:
        return [1.0 for _ in scores]
    return [normalize(min_v, max_v, score) for score in scores]

def normalize(min: float, max: float, value: float) -> float:
    return (value - min) / (max - min)

def hybrid_score(bm25: float, semantic: float, alpha: float = 0.5) -> float:
    return alpha * bm25 + (1 - alpha) * semantic

def rrf_score(rank: int, k: int = 60) -> float:
    assert rank > 0, f"Non-positive rank {rank}"
    return 1 / (k + rank)

def weighted_search_command(query: str, alpha: float=0.5, limit: int = 5):
    movies = load_movies()["movies"]
    hybrid = HybridSearch(movies)
    return hybrid.weighted_search(query, alpha, limit)

def rrf_search_command(query: str, k: int = 60, limit: int = 5, debug:bool = False):
    if debug:
        print(f"Modified query:'{query}'")
    movies = load_movies()["movies"]
    hybrid = HybridSearch(movies)
    return hybrid.rrf_search(query, k, limit, debug)
