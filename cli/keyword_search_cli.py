import argparse, sys
from process_text import process_text, remove_stop_words_and_stem
from utils import Movie
from inverted_index import InvertedIndex

def has_matching_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for q_token in query_tokens:
        for t_token in title_tokens:
            if q_token in t_token:
                return True
    return False

def main() -> None:
    #movies = load_movies()
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build the inverted index")

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
            search_result: list[Movie] = []
            for token in input:
                movie_ids = index.get_document(token)
                for movie_id in movie_ids:
                    movie = index.docmap[movie_id]
                    if movie not in search_result:
                        search_result.append(movie)
                    if len(search_result) >= 5:
                        break
                if len(search_result) >= 5:
                    break
            for result in search_result:
                print(f"{result['title'], result['id']}")
                
            #for movie in movies["movies"]:
            #    movie_tokens = remove_stop_words_and_stem(process_text(movie["title"]), processed_stop_words)
            #    if has_matching_tokens(input, movie_tokens):
            #        search_result.append(movie)
            #for i, movie in enumerate(search_result[:5], start=1):
            #    print(f"{i}. {movie["title"]}")
        case "build":
            index.build_command()
            

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()