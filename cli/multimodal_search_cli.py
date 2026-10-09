import argparse
from lib.image_search import verify_embedding, image_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    ver_im = subparsers.add_parser("verify_image_embedding", help="Print info about image embedding")
    ver_im.add_argument("image_path", type=str)
    s_im = subparsers.add_parser("image_search")
    s_im.add_argument("image_path", type=str)

    args = parser.parse_args()

    match args.command:
            case "verify_image_embedding":
                verify_embedding(args.image_path)

            case "image_search":
                result = image_search_command(args.image_path)
                for rank, movie in enumerate(result, start=1):
                    print(f"{rank}. {movie['title']} (similarity: {movie['score']:.3f})")
                    print(f"{movie["description"][:100]}...")
    
            case _:
                pass

if __name__ == "__main__":
    main()