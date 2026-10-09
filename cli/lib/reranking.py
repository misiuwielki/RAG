import os, json
from dotenv import load_dotenv
from openai import OpenAI
from lib.hybrid_search import RRFSearchResult
from sentence_transformers import CrossEncoder

class RerankedRRF(RRFSearchResult):
    rerank: float | None

def rerank_individual(query: str, title: str, description: str) -> float | None:
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
                "content": f"""Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {title} - {description}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Output ONLY the number in your response, no other text or explanation.

                Score:""",
                }
            ]
        )
    print(f"Prompt tokens: {response.usage.prompt_tokens if response.usage else 0}")
    print(f"Response tokens: {response.usage.completion_tokens if response.usage else 0}")
    if response.choices[0].message.content:
        text = response.choices[0].message.content.strip()
        try:
            score = float(text)
        except(TypeError, ValueError):
            score = None
    else:
        score =  None
    return score

def rerank_batch(query: str, movies:list[RRFSearchResult]) -> list[int] | None:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    doc_list_str = str(movies)
    client  = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages= [
                {
                "role": "user",
                "content": f"""Rank the movies listed below by relevance to the following search query.

                Query: "{query}"

                Movies:
                {doc_list_str}

                Return the movie IDs in order of relevance, best match first.

                Your response must be a raw JSON array of integers.
                Do not wrap the JSON in Markdown. Do not use a ```json code block.
                Do not include any explanatory text.

                For example:
                [75, 12, 34, 2, 1]

                Ranking:""",
                }
            ]
        )
    print(f"Prompt tokens: {response.usage.prompt_tokens if response.usage else 0}")
    print(f"Response tokens: {response.usage.completion_tokens if response.usage else 0}")
    if response.choices[0].message.content:
        result:list[int] = json.loads(response.choices[0].message.content.strip())
        try:
            _ = [r * 2 for r in result]
            score = result
        except(TypeError, ValueError):
            score = None
    else:
        score =  None
    return score

def rerank_cross_encode(search_results: list[RRFSearchResult], query:str):
    pairs:list[list[str]] = []
    for t_result in search_results:
        pairs.append([query, f"{t_result["title"]} - {t_result["description"]}"])
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs) # pyright: ignore[reportUnknownMemberType]
    return scores

def validate_rankings(search_results: list[str], query: str):
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
                "content": f"""Rate how relevant each result is to this query on a 0-3 scale:

                Query: "{query}"

                Results:
                {chr(10).join(search_results)}

                Scale:
                - 3: Highly relevant
                - 2: Relevant
                - 1: Marginally relevant
                - 0: Not relevant

                Do NOT give any numbers other than 0, 1, 2, or 3.

                Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                [2, 0, 3, 2, 0, 1]""",
                    }
                ]
            )
    print(f"Prompt tokens: {response.usage.prompt_tokens if response.usage else 0}")
    print(f"Response tokens: {response.usage.completion_tokens if response.usage else 0}")
    if response.choices[0].message.content:
        result:list[int] = json.loads(response.choices[0].message.content.strip())
        try:
            _ = [r * 2 for r in result]
            score = result
        except(TypeError, ValueError):
            score = None
    else:
        score =  None
    return score