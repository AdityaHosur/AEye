import os
import sys
import re
import string
from typing import List, Tuple, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# NLTK setup
import nltk
from nltk.corpus import stopwords

def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # stopwords
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    # POS tagger (name depends on nltk version)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        except LookupError:
            try:
                nltk.download("averaged_perceptron_tagger", quiet=True)
            except Exception:
                try:
                    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
                except Exception:
                    pass

_ensure_nltk()

# Optional: readability
try:
    from textstat import flesch_reading_ease
    _HAS_TEXTSTAT = True
except Exception:
    _HAS_TEXTSTAT = False


_FN_WORDS = {
    # A small set of common function words; ratio gives authorial style signal
    "the","a","an","and","or","but","if","then","else","when","while","because",
    "to","of","in","on","for","with","as","by","from","at","about","into","over",
    "after","before","between","without","within",
    "is","are","was","were","be","been","being","do","does","did","have","has","had",
    "that","which","who","whom","whose",
    "this","these","those","it","its","i","you","he","she","we","they","them","me","us",
    "my","your","his","her","our","their",
    "not","no","nor","so","too","very","just"
}

_STOP = set(stopwords.words("english"))

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")  # simple word pattern


def _tokenize_sentences(text: str) -> List[str]:
    return nltk.sent_tokenize(text)


def _tokenize_words(text: str) -> List[str]:
    # Robust word tokenizer with fallback to regex
    try:
        toks = nltk.word_tokenize(text)
        return [t for t in toks if any(ch.isalpha() for ch in t)]
    except Exception:
        return _WORD_RE.findall(text)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _readability(text: str) -> float:
    if not _HAS_TEXTSTAT:
        return 0.0
    try:
        return float(flesch_reading_ease(text))
    except Exception:
        return 0.0


def stylometric_features(text: str) -> Tuple[np.ndarray, List[str]]:
    words = _tokenize_words(text)
    sents = _tokenize_sentences(text)

    num_words = len(words)
    num_sents = len(sents)
    num_chars = len(text)

    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    word_lens = [len(w) for w in alpha_words]
    avg_word_len = float(np.mean(word_lens)) if word_lens else 0.0

    avg_sent_len = _safe_div(num_words, num_sents)
    ttr = _safe_div(len(set(w.lower() for w in words)), num_words)

    stop_ratio = _safe_div(sum(1 for w in words if w.lower() in _STOP), num_words)
    punct_ratio = _safe_div(sum(1 for ch in text if ch in string.punctuation), num_chars)
    digit_ratio = _safe_div(sum(1 for ch in text if ch.isdigit()), num_chars)
    upper_ratio = _safe_div(sum(1 for ch in text if ch.isupper()), num_chars)

    # Hapax legomena ratio
    if num_words:
        from collections import Counter
        counts = Counter(w.lower() for w in words)
        hapax_ratio = _safe_div(sum(1 for w,c in counts.items() if c == 1), num_words)
    else:
        hapax_ratio = 0.0

    # Word length distribution
    short_ratio = _safe_div(sum(1 for w in alpha_words if len(w) <= 3), len(alpha_words))
    medium_ratio = _safe_div(sum(1 for w in alpha_words if 4 <= len(w) <= 6), len(alpha_words))
    long_ratio = _safe_div(sum(1 for w in alpha_words if len(w) >= 7), len(alpha_words))

    # Function word usage ratio
    fn_ratio = _safe_div(sum(1 for w in words if w.lower() in _FN_WORDS), num_words)

    # POS tag distribution (coarse buckets)
    try:
        pos_tags = nltk.pos_tag(words)
    except Exception:
        pos_tags = []

    def pos_ratio(prefixes: Tuple[str, ...]) -> float:
        if not pos_tags:
            return 0.0
        cnt = sum(1 for _, tag in pos_tags if any(tag.startswith(p) for p in prefixes))
        return _safe_div(cnt, len(pos_tags))

    noun_ratio = pos_ratio(("NN",))
    verb_ratio = pos_ratio(("VB",))
    adj_ratio = pos_ratio(("JJ",))
    adv_ratio = pos_ratio(("RB",))
    pron_ratio = pos_ratio(("PR",))

    readability = _readability(text)

    feats = np.array([
        avg_word_len,
        avg_sent_len,
        ttr,
        stop_ratio,
        punct_ratio,
        digit_ratio,
        upper_ratio,
        hapax_ratio,
        short_ratio,
        medium_ratio,
        long_ratio,
        fn_ratio,
        noun_ratio,
        verb_ratio,
        adj_ratio,
        adv_ratio,
        pron_ratio,
        readability,
    ], dtype=float)

    names = [
        "avg_word_len",
        "avg_sent_len",
        "type_token_ratio",
        "stopword_ratio",
        "punct_ratio",
        "digit_ratio",
        "uppercase_ratio",
        "hapax_ratio",
        "short_word_ratio",
        "medium_word_ratio",
        "long_word_ratio",
        "function_word_ratio",
        "noun_ratio",
        "verb_ratio",
        "adj_ratio",
        "adv_ratio",
        "pron_ratio",
        "flesch_reading_ease",
    ]
    return feats, names


def normalize_features(f1: np.ndarray, f2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple min-max normalization across the two documents to equalize scales.
    """
    mn = np.minimum(f1, f2)
    mx = np.maximum(f1, f2)
    rng = mx - mn
    rng[rng == 0] = 1.0
    nf1 = (f1 - mn) / rng
    nf2 = (f2 - mn) / rng
    return nf1, nf2


def compare_stylometry(f1: np.ndarray, f2: np.ndarray) -> Dict[str, float]:
    """
    Returns:
      - cosine_sim: similarity in [0,1] (1 = identical)
      - euclidean: Euclidean distance on normalized features
      - l1: L1 (Manhattan) distance on normalized features
      - copying_likelihood: heuristic in [0,1]
    """
    nf1, nf2 = normalize_features(f1, f2)

    # Cosine similarity on raw features (shift to [0,1])
    cos = float(cosine_similarity([f1], [f2])[0][0])
    cos = max(-1.0, min(1.0, cos))
    cos01 = 0.5 * (cos + 1.0)

    # Distances on normalized features
    eu = float(np.linalg.norm(nf1 - nf2))
    l1 = float(np.abs(nf1 - nf2).sum())

    # Heuristic copying likelihood: higher with greater cosine and smaller distances
    # Scale distances to [0,1] via exp decay
    eu_term = float(np.exp(-eu))            # smaller eu -> closer to 1
    l1_term = float(np.exp(-l1 / len(f1)))  # avg per-feature deviation
    copying_likelihood = float(np.clip(0.5 * cos01 + 0.25 * eu_term + 0.25 * l1_term, 0.0, 1.0))

    return {
        "cosine_sim": cos01,
        "euclidean": eu,
        "l1": l1,
        "copying_likelihood": copying_likelihood,
    }


def _fmt_row(k: str, v1: float, v2: float) -> str:
    return f"{k:24s} | A: {v1:8.4f} | B: {v2:8.4f} | Δ: {v2 - v1:+8.4f}"


if __name__ == "__main__":
    file_a, file_b = "D:\\coding\\AEye\\backend\\uploads\\a1.txt", "D:\\coding\\AEye\\backend\\uploads\\a19.txt"
    if not (os.path.isfile(file_a) and os.path.isfile(file_b)):
        print("Both inputs must be valid files.")
        sys.exit(1)

    with open(file_a, "r", encoding="utf-8", errors="ignore") as fa:
        text_a = fa.read()
    with open(file_b, "r", encoding="utf-8", errors="ignore") as fb:
        text_b = fb.read()

    f1, names = stylometric_features(text_a)
    f2, _ = stylometric_features(text_b)

    print(f"Features extracted: {len(names)}")
    print("-" * 64)
    for i, name in enumerate(names):
        print(_fmt_row(name, f1[i], f2[i]))
    print("-" * 64)

    metrics = compare_stylometry(f1, f2)
    print(f"Cosine similarity (stylometric): {metrics['cosine_sim']:.4f}  [0=orthogonal, 1=identical]")
    print(f"Euclidean distance (norm’d):     {metrics['euclidean']:.4f}  [smaller is closer]")
    print(f"L1 distance (norm’d):            {metrics['l1']:.4f}  [smaller is closer]")
    print(f"Copying likelihood (heuristic):  {metrics['copying_likelihood']:.4f}  [0..1]")

    # Simple textual verdict (tune thresholds as needed)
    verdict = (
        "High stylometric similarity (possible copying)"
        if metrics["copying_likelihood"] >= 0.75
        else "Moderate stylometric similarity"
        if metrics["copying_likelihood"] >= 0.55
        else "Low stylometric similarity"
    )
    print(f"Verdict: {verdict}")