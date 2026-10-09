import argparse
from lib.hybrid_search import rrf_search_command
from lib.augmented_generation import rag, summarize, citations, question

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    query_parser = argparse.ArgumentParser(add_help=False)
    query_parser.add_argument("query", type=str, help="Search query for RAG")
    limit_parser = argparse.ArgumentParser(add_help=False)
    limit_parser.add_argument("--limit", type=int, default=5)

    subparsers.add_parser("rag", help="Perform RAG (search + generate answer)", parents=[query_parser])
    subparsers.add_parser("summarize", help="Perform RAG to get summarized response", parents=[query_parser, limit_parser])
    subparsers.add_parser("citations", help="Perform RAG to get summarized response with citations", parents=[query_parser, limit_parser])
    subparsers.add_parser("question", help="Perform RAG to answer a question", parents=[query_parser, limit_parser])

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            result = rrf_search_command(query)
            result = result[:5]
            response = rag(query, result)
            assert response is not None
            print("Search Results:")
            for i in range(len(result)):
                print(f"- {result[i]['title']}")
            print("RAG Response:")
            print(response)

        case "summarize":
            query = args.query
            limit = args.limit
            result = rrf_search_command(query, limit=limit)
            result = result[:limit]
            response = summarize(query, result)
            assert response is not None
            print("Search Results:")
            for i in range(len(result)):
                print(f"- {result[i]['title']}")
            print("LLM Summary:")
            print(response)

        case "citations":
            query = args.query
            limit = args.limit
            result = rrf_search_command(query, limit=limit)
            result = result[:limit]
            response = citations(query, result)
            assert response is not None
            print("Search Results:")
            for i in range(len(result)):
                print(f"- {result[i]['title']}")
            print("LLM Answer:")
            print(response)

        case "question":
            query = args.query
            limit = args.limit
            result = rrf_search_command(query, limit=limit)
            result = result[:limit]
            response = question(query, result)
            assert response is not None
            print("Search Results:")
            for i in range(len(result)):
                print(f"- {result[i]['title']}")
            print("LLM Answer:")
            print(response)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()