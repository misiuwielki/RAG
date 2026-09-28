import argparse, sys, math
from process_text import process_text, remove_stop_words_and_stem, process_single_term
from utils import search_for_tokens
from inverted_index import InvertedIndex
from constants import BM25_K1, BM25_B

def has_matching_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for q_token in query_tokens:
        for t_token in title_tokens:
            if q_token in t_token:
                return True
    return False

def load_index_and_process_term_to_token(index: InvertedIndex, term: str):
    try:
        index.load()
    except Exception as e:
        print(e)
        sys.exit(1)
    try:
        token = process_single_term(term, index.stop_words)
        return token
    except Exception as e:
        print(e)
        empty:list[str] = []
        return empty

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    term_parser = argparse.ArgumentParser(add_help=False)
    term_parser.add_argument("term", type=str, help="Term to be checked")
    id_parser = argparse.ArgumentParser(add_help=False)
    id_parser.add_argument("id", type=int, help="Document id")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build the inverted index")
    subparsers.add_parser("tf", help="Check term frequency in a given document by id", parents=[id_parser, term_parser])
    subparsers.add_parser("idf", help="Check inverse document frequency for term", parents=[term_parser])
    subparsers.add_parser("tfidf", help="Check TF-IDF of term in document by id", parents=[id_parser, term_parser])
    subparsers.add_parser("bm25idf", help="Check BM25 IDF for term", parents=[term_parser])
    bm25tf = subparsers.add_parser("bm25tf", help="Check BM25 TF of term in document by id, with customizable K1", parents=[id_parser, term_parser])
    bm25tf.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25tf.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 B parameter")
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs="?", default=5, help="Controls amount of results")


    args = parser.parse_args()
    index = InvertedIndex()

    match args.command:
        case "search":
            try:
                index.load()
            except Exception as e:
                print(e)
                sys.exit(1)
            print(f"Searching for: {args.query}")
            input = remove_stop_words_and_stem(process_text(args.query), index.stop_words)
            search_result = search_for_tokens(input, index, 5)
            for result in search_result:
                print(f"{result['title'], result['id']}")
                
        case "build":
            index.build_command()

        case "tf":
            token = load_index_and_process_term_to_token(index, args.term)
            if token:
                print(f"Frequency of {args.term} in document {args.id}: {index.get_tf(args.id, token[0])}")
            
        case "idf":
            token = load_index_and_process_term_to_token(index, args.term)
            if token:
                search_result = search_for_tokens(token, index)
                idf = math.log((len(index.docmap) + 1) / (len(search_result) + 1))
                print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            token = load_index_and_process_term_to_token(index, args.term)
            if token:
                search_result = search_for_tokens(token, index)
                idf = math.log((len(index.docmap) + 1) / (len(search_result) + 1))
                tf = index.get_tf(args.id, token[0])
                tfidf = idf * tf
                print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tfidf:.2f}")

        case "bm25idf":
            token = load_index_and_process_term_to_token(index, args.term)
            if token:
                bm25idf = index.get_bm25_idf(token[0])
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            token = load_index_and_process_term_to_token(index, args.term)
            if token:
                bm25tf = index.get_bm25_tf(args.id, token[0], args.k1, args.b)
                print(f"BM25 TF score of '{args.term}' in document {args.id}: {bm25tf:.2f}")

        case "bm25search":
            try:
                index.load()
            except Exception as e:
                print(e)
                sys.exit(1)
            print(f"Searching for: {args.query}")
            result = index.bm25_search(args.query, args.limit)
            for doc_id, score in result.items():
                print(f"({doc_id}) {index.docmap[doc_id]['title']} - Score: {score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()