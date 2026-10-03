from sentence_transformers import SentenceTransformer
from typing import TypedDict
import numpy as np, os, json
from lib.utils import Movie
from lib.semantic_search_utils import cosine_similarity, semantic_chunk_command


class ChunkScore(TypedDict):
    chunk_idx: int
    movie_idx: int
    score: float

class FinalResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, int]

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("empty text was provided")
        embedding = self.model.encode([text.strip()]) # pyright: ignore[reportUnknownMemberType]
        return embedding[0]

    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        document_list: list[str] = []
        for document in documents:
            self.document_map[document["id"]] = document
            document_list.append(f"{document['title']}: {document['description']}")
        embedding = self.model.encode(document_list, show_progress_bar=True) # pyright: ignore[reportUnknownMemberType]
        self.embeddings = embedding
        np.save("cache/movie_embeddings.npy", embedding)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]):
        self.documents = documents
        document_list: list[str] = []
        for document in documents:
            self.document_map[document["id"]] = document
            document_list.append(f"{document['title']}: {document['description']}")
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int=5) -> list[dict[str, str | float]]:
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        comparison:list[tuple[float, Movie]] = []
        q_emb = self.generate_embedding(query)
        for i in range(len(self.embeddings)):
            d_emb = self.embeddings[i]
            score = cosine_similarity(q_emb, d_emb)
            comparison.append((score, self.documents[i]))
        result_t = sorted(comparison, key=lambda x:x[0], reverse=True)
        result_d: list[dict[str, str | float]] = [{"score": t[0], "title": t[1]['title'], "description": t[1]['description']} for t in result_t[:limit]]
        return result_d



class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        chunk_list:list[str] = []
        chunk_metadata:list[dict[str, int]] = []
        for document in documents:
            self.document_map[document["id"]] = document
            if document["description"].strip() != "":
                chunks = semantic_chunk_command(document["description"], 4, 1)
                for i in range(len(chunks)):
                    chunk = chunks[i]
                    chunk_list.append(chunk)
                    metadata = {
                        "movie_idx": document["id"],
                        "chunk_idx": i,
                        "total_chunks": len(chunks)
                    }
                    chunk_metadata.append(metadata)
        chunk_embeddings = self.model.encode(chunk_list) # pyright: ignore[reportUnknownMemberType]
        self.chunk_embeddings = chunk_embeddings
        self.chunk_metadata = chunk_metadata
        np.save("cache/chunk_embeddings.npy", chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(chunk_list)}, f, indent=2)
        return chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        document_list: list[str] = []
        for document in documents:
            self.document_map[document["id"]] = document
            document_list.append(f"{document['title']}: {document['description']}")
        if os.path.exists("cache/chunk_embeddings.npy"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                metadata:dict[str, list[dict[str, int]]] = json.load(f)
            self.chunk_metadata = metadata["chunks"]
            if len(self.chunk_embeddings) == len(metadata['chunks']):
                return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        q_emb = self.generate_embedding(query)
        result_list:list[ChunkScore] = []
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("load or generate chunk embeddings first")
        for i in range(len(self.chunk_embeddings)):
            ch_emb = self.chunk_embeddings[i]
            score = cosine_similarity(q_emb, ch_emb)
            result_list.append({
                "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": score
            })
        best_results:dict[int, ChunkScore] = {}
        for result in result_list:
            best_so_far = best_results.get(result["movie_idx"])
            if best_so_far is None:
                best_results[result["movie_idx"]] = result
            elif result['score'] > best_so_far["score"]:
                best_results[result["movie_idx"]] = result
        sorted_result = sorted(best_results.items(), key=lambda item:item[1]["score"], reverse=True)[:limit]
        end_result:list[FinalResult] = []
        for result in sorted_result:
            end_result.append({
                "id": result[0],
                "title": self.document_map[result[0]]["title"],
                "document": self.document_map[result[0]]["description"][:100],
                "score": round(result[1]["score"], 4),
                "metadata": self.chunk_metadata[result[1]["chunk_idx"]] or {}
            })
        return end_result