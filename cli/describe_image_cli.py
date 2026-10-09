import argparse, mimetypes, base64
from lib.image_search import query_with_image, verify_embedding

def main() -> None:
    parser = argparse.ArgumentParser(description="Image description CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    parser.add_argument("--image", type=str, help="path to image file")
    parser.add_argument("--query", type=str, help="search query")

    ver_im = subparsers.add_parser("verify_image_embedding", help="Print info about image embedding")
    ver_im.add_argument("image_path", type=str)
    

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_embedding(args.image_path)
        case _:
            pass

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpg"
    with open(args.image, 'rb') as f:
        image = f.read()
    data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
    query = args.query.strip()
    response = query_with_image(query, data_url)
    assert response is not None
    print(f"Rewritten query: {response}")
    

if __name__ == "__main__":
    main()