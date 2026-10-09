import os
from dotenv import load_dotenv
from openai import OpenAI
from lib.hybrid_search import RRFSearchResult
from lib.reranking import RerankedRRF
from lib.utils import format_results

def rag(query: str, search_results: list[RRFSearchResult] | list[RerankedRRF]):
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    docs = format_results(search_results)
    client  = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages= [
                {
                "role": "user",
                "content": f"""You are a RAG agent for Webflyx, a movie streaming service.
                Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
                Provide a comprehensive answer that addresses the user's query. Don't use markdown.

                Query: {query}

                Documents:
                {chr(10).join(docs)}

                Answer:""",
                }
            ]
        )
    print(f"Prompt tokens: {response.usage.prompt_tokens if response.usage else 0}")
    print(f"Response tokens: {response.usage.completion_tokens if response.usage else 0}")
    if response.choices[0].message.content and response.choices[0].message.content != "User Safety: safe":
        return response.choices[0].message.content.strip()
    else:
        rag(query, search_results)

def summarize(query: str, search_results: list[RRFSearchResult] | list[RerankedRRF]):
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    docs = format_results(search_results)
    client  = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages= [
                {
                "role": "user",
                "content": f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

                The goal is to provide comprehensive information so that users know what their options are.
                Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

                This should be tailored to Webflyx users. Webflyx is a movie streaming service. Don't use markdown.

                Query: {query}

                Search results:
                {chr(10).join(docs)}

                Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:""",
                }
            ]
        )
    print(f"Prompt tokens: {response.usage.prompt_tokens if response.usage else 0}")
    print(f"Response tokens: {response.usage.completion_tokens if response.usage else 0}")
    if response.choices[0].message.content and response.choices[0].message.content != "User Safety: safe":
        return response.choices[0].message.content.strip()
    else:
        summarize(query, search_results)

def citations(query: str, search_results: list[RRFSearchResult] | list[RerankedRRF]):
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    docs = format_results(search_results)
    client  = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages= [
                {
                "role": "user",
                "content": f"""Answer the query below and give information based on the provided documents.

                The answer should be tailored to users of Webflyx, a movie streaming service.
                If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

                Query: {query}

                Documents:
                {chr(10).join(docs)}

                Instructions:
                - Provide a comprehensive answer that addresses the query
                - Cite sources in the format [1], [2], etc. when referencing information
                - If sources disagree, mention the different viewpoints
                - If the answer isn't in the provided documents, say "I don't have enough information"
                - Be direct and informative

                Answer:""",
                }
            ]
        )
    print(f"Prompt tokens: {response.usage.prompt_tokens if response.usage else 0}")
    print(f"Response tokens: {response.usage.completion_tokens if response.usage else 0}")
    if response.choices[0].message.content and response.choices[0].message.content != "User Safety: safe":
        return response.choices[0].message.content.strip()
    else:
        citations(query, search_results)

def question(query: str, search_results: list[RRFSearchResult] | list[RerankedRRF]):
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    docs = format_results(search_results)
    client  = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    response = client.chat.completions.create(
        model="openrouter/free",
        messages= [
                {
                "role": "user",
                "content": f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

                Question: {query}

                Documents:
                {chr(10).join(docs)}

                Instructions:
                - Answer questions directly and concisely
                - Be casual and conversational
                - Don't be cringe or hype-y
                - Talk like a normal person would in a chat conversation

                Answer:""",
                }
            ]
        )
    print(f"Prompt tokens: {response.usage.prompt_tokens if response.usage else 0}")
    print(f"Response tokens: {response.usage.completion_tokens if response.usage else 0}")
    if response.choices[0].message.content and "User Safety:" not in response.choices[0].message.content:
        return response.choices[0].message.content.strip()
    else:
        question(query, search_results)