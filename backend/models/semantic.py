import re
import os
import sys
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Optional heavy deps
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


def sentence_tokenize(text: str) -> List[str]:
    """
    Lightweight sentence splitter. Splits on ., !, ? followed by whitespace/newline.
    Keeps sentences with at least 2 non-space chars.
    """
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in parts if len(s.strip()) >= 2]
    return sentences


def compute_sentence_similarity(
    sentences_a: List[str], sentences_b: List[str]
) -> Tuple[np.ndarray, float]:
    """
    Lexical similarity (TF-IDF). Returns (matrix, score), where matrix is |A|x|B|.
    Score = 0.5 * (mean(row-max) + mean(col-max)).
    """
    a = [s for s in sentences_a if s]
    b = [s for s in sentences_b if s]

    if len(a) == 0 or len(b) == 0:
        sim = np.zeros((len(a), len(b)), dtype=np.float32)
        return sim, 0.0

    corpus = a + b
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=1,
    )
    X = vectorizer.fit_transform(corpus)
    Xa = X[: len(a), :]
    Xb = X[len(a) :, :]

    sim = cosine_similarity(Xa, Xb)  # shape (len(a), len(b))

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


# -------- Semantic similarity (embeddings) --------

_ST_MODEL: Optional[SentenceTransformer] = None

def _get_st_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    global _ST_MODEL
    if _ST_MODEL is None:
        if not _HAS_ST:
            raise RuntimeError(
                "sentence-transformers is not installed. Install with:\n  pip install sentence-transformers"
            )
        _ST_MODEL = SentenceTransformer(model_name)
    return _ST_MODEL


def embed_sentences(sentences: List[str]) -> np.ndarray:
    if len(sentences) == 0:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_st_model()
    emb = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    emb = emb.astype(np.float32, copy=False)
    return emb


def compute_semantic_sentence_similarity(
    sentences_a: List[str], sentences_b: List[str]
) -> Tuple[np.ndarray, float]:
    """
    Semantic similarity using sentence-transformers embeddings.
    Returns (matrix, score), matrix is |A|x|B| cosine similarity.
    """
    a = [s for s in sentences_a if s]
    b = [s for s in sentences_b if s]
    if len(a) == 0 or len(b) == 0:
        sim = np.zeros((len(a), len(b)), dtype=np.float32)
        return sim, 0.0

    Ea = embed_sentences(a)
    Eb = embed_sentences(b)

    # cosine similarity on normalized embeddings = dot product
    sim = Ea @ Eb.T  # shape (len(a), len(b))

    row_max = sim.max(axis=1) if sim.size else np.array([0.0], dtype=np.float32)
    col_max = sim.max(axis=0) if sim.size else np.array([0.0], dtype=np.float32)
    score = 0.5 * (row_max.mean() + col_max.mean())
    return sim.astype(np.float32), float(score)


def compute_semantic_similarity_for_texts(text_a: str, text_b: str) -> Dict[str, Any]:
    sentences_a = sentence_tokenize(text_a)
    sentences_b = sentence_tokenize(text_b)
    sim, score = compute_semantic_sentence_similarity(sentences_a, sentences_b)
    return {
        "sentencesA": sentences_a,
        "sentencesB": sentences_b,
        "matrix": sim.tolist(),
        "score": score,
    }


def compute_semantic_similarity_for_files(path_a: str, path_b: str, encoding: str = "utf-8") -> Dict[str, Any]:
    with open(path_a, "r", encoding=encoding, errors="ignore") as fa:
        text_a = fa.read()
    with open(path_b, "r", encoding=encoding, errors="ignore") as fb:
        text_b = fb.read()
    return compute_semantic_similarity_for_texts(text_a, text_b)


# -------- Clustering + Visualization --------

def choose_k_via_silhouette(embeddings: np.ndarray, k_min: int = 2, k_max: int = 8) -> int:
    n = embeddings.shape[0]
    if n <= 2:
        return 1
    k_max = min(k_max, n - 1)
    best_k, best_score = None, -1.0
    for k in range(k_min, max(k_min + 1, k_max + 1)):
        try:
            km = KMeans(n_clusters=k, n_init='auto', random_state=42)
            labels = km.fit_predict(embeddings)
            if len(set(labels)) == 1:
                continue
            score = silhouette_score(embeddings, labels, metric="cosine")
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue
    if best_k is None:
        best_k = 1
    return best_k


def cluster_sentences(embeddings: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, int]:
    if embeddings.shape[0] == 0:
        return np.array([], dtype=int), 0
    if k is None:
        k = choose_k_via_silhouette(embeddings, 2, 8)
    if k <= 1:
        return np.zeros(embeddings.shape[0], dtype=int), 1
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = km.fit_predict(embeddings)
    return labels.astype(int), k


def _shorten(text: str, max_len: int = 32) -> str:
    t = text.strip().replace("\n", " ")
    return (t[: max_len - 1] + "…") if len(t) > max_len else t


def plot_cross_similarity_heatmap(matrix: np.ndarray, sentences_a: List[str], sentences_b: List[str], title: str):
    if not _HAS_MPL:
        print("[plot] matplotlib not installed. pip install matplotlib")
        return
    fig_w = min(12, max(6, matrix.shape[1] * 0.25))
    fig_h = min(12, max(4, matrix.shape[0] * 0.25))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Doc B sentences")
    ax.set_ylabel("Doc A sentences")
    # Tick labels: keep it readable
    max_ticks = 30
    def idxs(n): 
        return list(range(n)) if n <= max_ticks else list(range(0, n, max(1, n // max_ticks)))
    ax.set_xticks(idxs(len(sentences_b)))
    ax.set_yticks(idxs(len(sentences_a)))
    ax.set_xticklabels([_shorten(sentences_b[i], 24) for i in ax.get_xticks()], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([_shorten(sentences_a[i], 24) for i in ax.get_yticks()], fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Cosine similarity")
    fig.tight_layout()
    plt.show()


def plot_2d_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    doc_ids: np.ndarray,
    sentences: List[str],
    title: str,
):
    if not _HAS_MPL:
        print("[plot] matplotlib not installed. pip install matplotlib")
        return
    if embeddings.shape[0] == 0:
        return
    # Reduce to 2D
    n_components = 2 if embeddings.shape[0] > 2 else 1
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(embeddings)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros_like(coords)])

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = mpl.cm.get_cmap("tab10", max(1, labels.max() + 1 if labels.size else 1))
    markers = {0: "o", 1: "s"}

    for cl in sorted(set(labels.tolist() if labels.size else [0])):
        mask = labels == cl if labels.size else np.ones(embeddings.shape[0], dtype=bool)
        for doc in [0, 1]:
            sub = mask & (doc_ids == doc)
            if not np.any(sub):
                continue
            ax.scatter(
                coords[sub, 0],
                coords[sub, 1],
                s=60,
                c=[cmap(cl)],
                marker=markers.get(doc, "o"),
                alpha=0.8,
                label=f"Cluster {cl} • Doc {'A' if doc==0 else 'B'}",
                edgecolor="white",
                linewidths=0.5,
            )

    # Annotate a few points
    show_n = min(30, coords.shape[0])
    step = max(1, coords.shape[0] // show_n)
    for i in range(0, coords.shape[0], step):
        ax.annotate(_shorten(sentences[i], 28), (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.legend(loc="best", fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_a, file_b = 'D:\\coding\\AEye\\backend\\uploads\\a1.txt', 'D:\\coding\\AEye\\backend\\uploads\\a2.txt'
    if not (os.path.isfile(file_a) and os.path.isfile(file_b)):
        print("Both inputs must be valid files.")
        sys.exit(1)

    with open(file_a, "r", encoding="utf-8", errors="ignore") as fa:
        text_a = fa.read()
    with open(file_b, "r", encoding="utf-8", errors="ignore") as fb:
        text_b = fb.read()

    # Tokenize
    sents_a = sentence_tokenize(text_a)
    sents_b = sentence_tokenize(text_b)
    print(f"Sentences: A={len(sents_a)}, B={len(sents_b)}")

    # Lexical similarity (TF-IDF)
    lex_mat, lex_score = compute_sentence_similarity(sents_a, sents_b)
    print(f"Lexical similarity score: {lex_score:.4f}")

    # Semantic similarity (embeddings)
    if not _HAS_ST:
        print("\nSemantic similarity requires sentence-transformers. Install:")
        print("  pip install sentence-transformers")
        sys.exit(0)

    sem_mat, sem_score = compute_semantic_sentence_similarity(sents_a, sents_b)
    print(f"Semantic similarity score: {sem_score:.4f}")

    # Build combined embeddings for clustering and plotting
    all_sents = sents_a + sents_b
    if len(all_sents) == 0:
        print("No sentences to cluster.")
        sys.exit(0)
    all_emb = embed_sentences(all_sents)
    doc_ids = np.array([0] * len(sents_a) + [1] * len(sents_b), dtype=int)

    labels, k = cluster_sentences(all_emb, None)
    print(f"Clustering: K={k}, labels in [0..{labels.max() if labels.size else 0}]")

    # Plots
    if _HAS_MPL:
        plot_cross_similarity_heatmap(
            sem_mat, sents_a, sents_b, title="Cross-document sentence similarity (Semantic)"
        )
        plot_2d_clusters(
            all_emb, labels, doc_ids, all_sents, title="Sentence clusters (semantic embeddings + PCA)"
        )
    else:
        print("\nGraphs require matplotlib. Install:")
        print("  pip install matplotlib")