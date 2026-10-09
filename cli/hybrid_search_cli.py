import argparse
from lib.hybrid_search import normalize_scores, weighted_search_command, rrf_search_command
from lib.query_enhancement import correct_spelling, rewrite, expand
from lib.reranking import rerank_individual, rerank_batch, rerank_cross_encode, validate_rankings, RerankedRRF
from lib.utils import format_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize = subparsers.add_parser("normalize", help="normalize scores")
    normalize.add_argument("scores", nargs="*", type=float)
    weighted = subparsers.add_parser("weighted-search", help="Hybrid search with customizable alpha")
    weighted.add_argument("query", type=str)
    weighted.add_argument("--alpha", type=float, default=0.5)
    weighted.add_argument("--limit", type=int, default=5)
    rrf = subparsers.add_parser("rrf-search", help="RRF search with customizable k")
    rrf.add_argument("query", type=str)
    rrf.add_argument("--k", type=int, default=60)
    rrf.add_argument("--limit", type=int, default=5)
    rrf.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Reranking method")
    rrf.add_argument("--debug", action="store_true")
    rrf.add_argument("--evaluate", action="store_true")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"{score:.4f}")

        case "weighted-search":
            result = weighted_search_command(args.query, args.alpha, args.limit)
            for i in range(args.limit):
                movie = result[i]
                print(f"{i +1}. {movie['title']}")
                print(f"Hybrid Score: {movie['hybrid']:.3f}")
                print(f"BM25: {movie['bm25']:.3f}, Semantic: {movie['semantic']:.3f}")
                print(f"{movie['description']}")

        case "rrf-search":
            debug = args.debug
            query = args.query
            if debug:
                print(f"Original query: {query}")
            match args.enhance:
                case "spell":
                    input = query
                    query = correct_spelling(query)
                    print(f"Enhanced query ({args.enhance}) : '{input}' -> '{query}'")
                case "rewrite":
                    input = query
                    query = rewrite(query)
                    print(f"Enhanced query ({args.enhance}) : '{input}' -> '{query}'")
                case "expand":
                    input = query
                    query = expand(query)
                    print(f"Enhanced query ({args.enhance}) : '{input}' -> '{query}'")
                case _:
                    pass
            result = rrf_search_command(query, args.k, args.limit, debug)
            if debug:
                original_list = [r['title'] for r in result[:args.limit * 5]]
                print(f"RRF raw results: '{", ".join(original_list)}'")
            match args.rerank_method:
                case "individual":
                    trun_results = result[:args.limit * 5]
                    r_results: list[RerankedRRF] = []
                    for t_result in trun_results:
                        score = rerank_individual(query, t_result["title"], t_result["description"])
                        if score is None:
                            for _ in range(2):
                                score = rerank_individual(query, t_result["title"], t_result["description"])
                                if score is not None:
                                    break
                        r_result: RerankedRRF = {**t_result, "rerank": score}
                        r_results.append(r_result)
                    result = sorted(r_results, key= lambda r:r["rerank"] if r["rerank"] is not None else -1.0, reverse=True)

                case "batch":
                    trun_results = result[:args.limit * 5]
                    r_results = []
                    scores = rerank_batch(query, trun_results)
                    if scores is None:
                        for _ in range(4):
                            score = rerank_batch(query, trun_results)
                            if score is not None:
                                break
                    assert scores is not None
                    for rank, id in enumerate(scores[:args.limit]):
                        for entry in trun_results:
                            if id == entry["id"]:
                                r_result:RerankedRRF = {**entry, "rerank": rank}
                                r_results.append(r_result)
                    result = r_results
                case "cross_encoder":
                    trun_results = result[:args.limit * 5]
                    ce_result = rerank_cross_encode(trun_results, query)
                    r_results = []
                    for i in range(len(ce_result)):
                        score = float(ce_result[i])
                        r_result:RerankedRRF = {**trun_results[i], "rerank": score}
                        r_results.append(r_result)
                    result = sorted(r_results, key= lambda d:d["rerank"] if d["rerank"] is not None else 0, reverse=True)

                case _:
                    pass
            if debug:
                rerank_list = [r["title"] for r in result[:args.limit]]
                print(f"Reranked RRF results: '{", ".join(rerank_list)}'")
            print(f"searching for {query}")
            formatted_results:list[str] = format_results(result[:args.limit])
            for i in range(args.limit):
                movie = result[i]
                print(f"{i +1}. {movie['title']}")
                if movie.get("rerank") is not None:
                    match args.rerank_method:
                        case "individual":
                            print(f"Re-rank Score: {movie['rerank']:.3f}/10")
                        case "batch":
                            print(f"Re-rank Rank: {movie['rerank']}")
                        case "cross_encoder":
                            print(f"Cross Encoder Score: {movie['rerank']:.3f}")
                        case _:
                            pass
                print(f"RRF Score: {movie['rrf']:.3f}")
                print(f"BM25: {movie['bm25']}, Semantic: {movie['semantic']}")
                print(f"{movie['description']}")
            if args.evaluate:
                print("LLM evaluation of relevance to query on scale of 0 to 3:")
                scores = validate_rankings(formatted_results, query)
                if scores is None:
                    for _ in range(4):
                        score = validate_rankings(formatted_results, query)
                        if score is not None:
                            break
                assert scores is not None
                for rank, score in enumerate(scores, start=1):
                    print(f"{rank}. {result[rank - 1]['title']}: {score}/3")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()