import re
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def sentence_tokenize(text: str) -> List[str]:
    """
    Lightweight sentence splitter. Splits on ., !, ? followed by whitespace/newline.
    Keeps sentences with at least 2 non-space chars.
    """
    if not text:
        return []
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    # Filter very short parts (e.g., empty or single char)
    sentences = [s.strip() for s in parts if len(s.strip()) >= 2]
    return sentences


def compute_sentence_similarity(
    sentences_a: List[str], sentences_b: List[str]
) -> Tuple[np.ndarray, float]:
    """
    Build a TF-IDF over both sets of sentences and compute cosine similarity matrix
    between sentences in A (rows) and sentences in B (cols). Returns (matrix, score).

    Score is symmetric average of row-wise and column-wise maxima:
      0.5 * (mean(max_row) + mean(max_col))
    """
    a = [s for s in sentences_a if s]
    b = [s for s in sentences_b if s]

    if len(a) == 0 or len(b) == 0:
        # No cross similarity possible
        sim = np.zeros((len(a), len(b)), dtype=np.float32)
        return sim, 0.0

    corpus = a + b
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",  # words only
        ngram_range=(1, 2),            # unigrams + bigrams for better lexical coverage
        min_df=1
    )
    X = vectorizer.fit_transform(corpus)
    Xa = X[: len(a), :]
    Xb = X[len(a) :, :]

    sim = cosine_similarity(Xa, Xb)  # shape (len(a), len(b))

    # Symmetric score: average of row maxes and column maxes
    row_max = sim.max(axis=1) if sim.size else np.array([0.0])
    col_max = sim.max(axis=0) if sim.size else np.array([0.0])
    score = 0.5 * (row_max.mean() + col_max.mean())
    return sim.astype(np.float32), float(score)


def compute_similarity_for_texts(text_a: str, text_b: str) -> Dict[str, Any]:
    sentences_a = sentence_tokenize(text_a)
    sentences_b = sentence_tokenize(text_b)
    sim, score = compute_sentence_similarity(sentences_a, sentences_b)
    return {
        "sentencesA": sentences_a,
        "sentencesB": sentences_b,
        "matrix": sim.tolist(),
        "score": score,
    }


def compute_similarity_for_files(path_a: str, path_b: str, encoding: str = "utf-8") -> Dict[str, Any]:
    with open(path_a, "r", encoding=encoding, errors="ignore") as fa:
        text_a = fa.read()
    with open(path_b, "r", encoding=encoding, errors="ignore") as fb:
        text_b = fb.read()
    return compute_similarity_for_texts(text_a, text_b)


if __name__ == "__main__":
    result = compute_similarity_for_files('D:\\coding\\AEye\\backend\\uploads\\a1.txt', 'D:\\coding\\AEye\\backend\\uploads\\a18.txt')
    print(f"Sentences A: {len(result['sentencesA'])}, B: {len(result['sentencesB'])}")
    print(f"Lexical similarity score: {result['score']:.4f}")