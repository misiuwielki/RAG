import string
from nltk.stem import PorterStemmer

def process_text(input:str) -> list[str]:
    lower = input.lower()
    translation_table = str.maketrans({punctuation: "" for punctuation in string.punctuation})
    no_punctuation = lower.translate(translation_table)
    tokenised = no_punctuation.split()
    return tokenised

def remove_stop_words_and_stem(input: list[str], stop_words: list[str]) -> list[str]:
    result: list[str] = []
    stemmer = PorterStemmer()
    for word in input:
        if word not in stop_words:
            stem = stemmer.stem(word=word)
            result.append(stem)
    return result

def load_stop_words():
    stop_words = open("data/stopwords.txt").read().splitlines()
    processed_stop_words: list[str] = []
    for word in stop_words:
        processed_stop_words.append(process_text(word)[0])
    return processed_stop_words

def process_single_term(term: str, stop_words: list[str]):
    result = remove_stop_words_and_stem(process_text(term), stop_words)
    if len(result) != 1:
        raise Exception(f"expected to return 1 token, got {len(result)}")
    return result
