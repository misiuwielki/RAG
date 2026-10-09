import argparse, json
from typing import TypedDict
from lib.hybrid_search import rrf_search_command

class GoldenSub(TypedDict):
    query: str
    relevant_docs: list[str]

class Golden(TypedDict):
    test_cases: list[GoldenSub]

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
         "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    
    args = parser.parse_args()
    limit = args.limit
    with open("data/golden_dataset.json", "r") as f:
        golden:Golden = json.load(f)
    print(f"k={limit}")
    print()
    for case in golden["test_cases"]:
        result = rrf_search_command(case["query"], limit=limit)
        result = result[:limit]
        result_titles = [r["title"] for r in result]
        p_counter = 0
        for wanted in case["relevant_docs"]:
            if wanted in result_titles:
                p_counter += 1
        precision = p_counter/limit
        recall = p_counter/len(case['relevant_docs'])
        print(f"- Query: {case['query'].strip()}")
        print(f"    - Precision@{limit}: {precision:.4f}")
        print(f"    - Recall@{limit}: {recall:.4f}")
        print(f"    - F1 Score: {(2 * (precision * recall) / (precision + recall)):.4f}")
        print(f"    - Retrieved: {", ".join(result_titles)}")
        print(f"    - Relevant: {", ".join(case['relevant_docs'])}")
        print()


     # run evaluation logic here

if __name__ == "__main__":
     main()