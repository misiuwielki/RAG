import os
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from sentence_transformers import SentenceTransformer
from typing import TypedDict
from lib.inverted_index import InvertedIndex
from lib.semantic_search_utils import cosine_similarity

class MultimodalResult(TypedDict):
    id: int
    title: str
    description: str
    score: float

def query_with_image(query: str, data_url: str):
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client  = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages= [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
                    - Synthesize visual and textual information
                    - Focus on movie-specific details (actors, scenes, style, etc.)
                    - Return only the rewritten query, without any additional commentary"""},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": query}
        ]}
            ]
        )
    print(f"Prompt tokens: {response.usage.prompt_tokens if response.usage else 0}")
    print(f"Response tokens: {response.usage.completion_tokens if response.usage else 0}")
    print(f"Total tokens: {response.usage.total_tokens if response.usage else 0}")
    if response.choices and response.choices[0].message.content is not None and "User Safety:" not in response.choices[0].message.content:
        return response.choices[0].message.content.strip()
    else:
        query_with_image(query, data_url)

class MultimodalSearch:
    def __init__(self, model_name:str="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()
        self.texts:list[str] = []
        self.text_embeddings = self.build_text_embeddings()

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        image_embedding = self.model.encode([image]) # pyright: ignore[reportUnknownMemberType]
        return image_embedding[0]

    def build_texts(self):
        texts:list[str] = []
        print(len(self.idx.docmap))
        for id in self.idx.docmap:
            movie = self.idx.docmap[id]
            texts.append(f"{movie['title']}: {movie['description']}")
        self.texts = texts

    def build_text_embeddings(self):
        embeddings = self.model.encode(self.texts) # pyright: ignore[reportUnknownMemberType]
        self.text_embeddings =  embeddings

    def search_with_image(self, image_path: str):
        img_emb = self.embed_image(image_path)[0]
        cos_results:dict[int, float] = {}
        if self.text_embeddings is None:
            raise ValueError
        for id, t_emb in enumerate(self.text_embeddings, start=1):
            cos = cosine_similarity(t_emb, img_emb)
            cos_results[id] = cos
        result = sorted(cos_results.items(), key=lambda d:d[1], reverse=True)
        search_results:list[MultimodalResult] = []
        for id, score in result:
            search_results.append({
                "id": id,
                "title": self.idx.docmap[id]["title"],
                "description": self.idx.docmap[id]["description"],
                "score": score
            })
        return search_results[:5]

def verify_embedding(image_path: str):
    model = MultimodalSearch()
    embedding = model.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path: str):
    model = MultimodalSearch()
    model.idx.load()
    model.build_texts()
    model.build_text_embeddings()
    result = model.search_with_image(image_path)
    return result