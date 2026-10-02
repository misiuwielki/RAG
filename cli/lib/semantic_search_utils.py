import numpy as np, string, re

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product/(norm1 * norm2)

def chunk_command(text: str, chunk_size: int, overlap:int=0) -> list[str]:
    parts = text.split()
    result:list[str] = []
    times = len(parts) // chunk_size
    for i in range(times):
        first_slice = 0
        if i > 0:
            first_slice = i * chunk_size - overlap
        chunk_part = parts[first_slice: ((i + 1) * chunk_size)]
        chnk = " ".join(chunk_part)
        result.append(chnk)
    last_chunk = " ".join(parts[(times * chunk_size) - overlap:])
    if last_chunk.strip() != "":
        result.append(last_chunk)
    return result

def semantic_chunk_command(text: str, chunk_size: int, overlap:int=0) -> list[str]:
    redacted_text = text.strip()
    if redacted_text == "":
        return []
    sentences: list[str] = re.split(r"(?<=[.!?])\s+", redacted_text)
    if len(sentences) == 1 and not sentences[0].endswith(string.punctuation):
        return sentences
    sentences = [s.strip() for s in sentences if s.strip() != ""]
    result:list[str] = []
    i = 0
    while i < len(sentences):
        sub_sentences = sentences[i : i + chunk_size]
        chunk = " ".join(sub_sentences)
        if chunk != "":
            result.append(" ".join(sub_sentences))
        if i + chunk_size < len(sentences):
            i += chunk_size - overlap
        else:
            i += chunk_size
    return result