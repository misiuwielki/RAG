import argparse
from lib.semantic_search_commands import (verify_model, embed_text, verify_embeddings, embed_query,
    search_command, embed_chunks_command, search_chunked_command)
from lib.semantic_search_utils import chunk_command, semantic_chunk_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    text_parser = argparse.ArgumentParser(add_help=False)
    text_parser.add_argument("text", type=str)
    query_parser = argparse.ArgumentParser(add_help=False)
    query_parser.add_argument("query", type=str)
    limit_parser = argparse.ArgumentParser(add_help=False)
    limit_parser.add_argument("--limit", type=int, default=5)

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the model")
    subparsers.add_parser("embed_text", help="Create embedding for text", parents=[text_parser])
    subparsers.add_parser("verify_embeddings", help="Verify embeddings for db")
    subparsers.add_parser("embed_query", help="Create embedding for query", parents=[query_parser])
    subparsers.add_parser("search", help="Search for movies by a query", parents=[query_parser, limit_parser])
    chunk = subparsers.add_parser("chunk", help="Chunk a text into smaller fragments, with customizable chunk size and overlap", parents=[text_parser])
    chunk.add_argument("--chunk-size", type=int, default=200)
    chunk.add_argument("--overlap", type=int, default=0)
    sem_chunk = subparsers.add_parser("semantic_chunk", help="Chunk a text semantically into smaller fragments, with customizable chunk size and overlap", parents=[text_parser])
    sem_chunk.add_argument("--max-chunk-size", type=int, default=4)
    sem_chunk.add_argument("--overlap", type=int, default=0)
    subparsers.add_parser("embed_chunks", help="Create chunk embeddings for db")
    subparsers.add_parser("search_chunked", help="Search for movies by a query using chunking", parents=[query_parser, limit_parser])
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query(args.query)

        case "search":
            search_command(args.query, args.limit)

        case "chunk":
            result = chunk_command(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i in range(len(result)):
                print(f"{i + 1}. {result[i]}")

        case "semantic_chunk":
            result = semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i in range(len(result)):
                print(f"{i + 1}. {result[i]}")

        case "embed_chunks":
            embed_chunks_command()

        case "search_chunked":
            search_chunked_command(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()