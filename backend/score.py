import os
import json
import csv
import math
import warnings
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# Optional plotting deps (static)
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# Optional interactive plotting deps
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# Optional convex hull (to draw cluster hulls)
try:
    from scipy.spatial import ConvexHull
    _HAS_HULL = True
except Exception:
    _HAS_HULL = False

# For 2D layout
try:
    from sklearn.manifold import SpectralEmbedding, TSNE
    _HAS_SE = True
except Exception:
    _HAS_SE = False

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

# Local models
from models.lexical import sentence_tokenize as lex_sent_tokenize, compute_sentence_similarity as lex_pair_sim
from models.semantic import sentence_tokenize as sem_sent_tokenize, embed_sentences
from models.style import stylometric_features, compare_stylometry


def list_text_files(dir_path: str) -> List[str]:
    exts = {".txt", ".md", ".text", ".log"}
    files = []
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
            files.append(p)
    return sorted(files, key=lambda x: x.lower())


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def safe_mean_max_score(sim: np.ndarray) -> float:
    if sim.size == 0:
        return 0.0
    row_max = sim.max(axis=1)
    col_max = sim.max(axis=0)
    return float(0.5 * (row_max.mean() + col_max.mean()))


def combine_scores(scores: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, float]]:
    base_w = {"semantic": 0.5, "lexical": 0.35, "stylometric": 0.15}
    avail = {k: v for k, v in scores.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    if not avail:
        return 0.0, {}
    w_sum = sum(base_w[k] for k in avail)
    weights = {k: base_w[k] / w_sum for k in avail}
    combined = sum(weights[k] * float(avail[k]) for k in avail)
    return float(combined), weights


def layout_2d_from_similarity(sim: np.ndarray, method: str = "spectral") -> np.ndarray:
    """
    Compute 2D layout from similarity matrix.
    method: 'spectral', 'tsne', or 'pca'
    """
    n = sim.shape[0]
    if n == 1:
        return np.zeros((1, 2), dtype=float)
    
    A = np.array(sim, dtype=float)
    A[~np.isfinite(A)] = 0.0
    A = np.clip(A, 0.0, 1.0)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    
    if method == "spectral" and _HAS_SE:
        try:
            emb = SpectralEmbedding(n_components=2, affinity="precomputed", random_state=42).fit_transform(A)
            return emb.astype(float)
        except Exception:
            pass
    
    if method == "tsne" and _HAS_SE:
        try:
            D = 1.0 - A
            np.fill_diagonal(D, 0.0)
            emb = TSNE(n_components=2, metric="precomputed", random_state=42, perplexity=min(30, n-1)).fit_transform(D)
            return emb.astype(float)
        except Exception:
            pass
    
    # Fallback: PCA on distance
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(D)
    return coords.astype(float)


def cluster_from_similarity(sim: np.ndarray, k_min: int = 2, k_max: int = 10) -> Tuple[np.ndarray, int]:
    """
    Cluster docs from combined similarity so copied docs group together.
    Picks k via best silhouette on precomputed distance (1 - sim).
    """
    n = sim.shape[0]
    if n <= 2:
        return np.zeros(n, dtype=int), 1
    
    A = np.array(sim, dtype=float)
    A[~np.isfinite(A)] = 0.0
    A = np.clip(A, 0.0, 1.0)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0)
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)

    k_min = max(2, k_min)
    k_max = max(k_min, min(k_max, n - 1))

    best_labels = None
    best_score = -1.0
    best_k = 1

    # Try SpectralClustering across k
    for k in range(k_min, k_max + 1):
        try:
            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=42,
            )
            labels = sc.fit_predict(A)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(D, labels, metric="precomputed")
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k
        except Exception:
            continue

    # Fallback: AgglomerativeClustering
    if best_labels is None:
        for k in range(k_min, k_max + 1):
            try:
                try:
                    agg = AgglomerativeClustering(
                        n_clusters=k, affinity="precomputed", linkage="average"
                    )
                except TypeError:
                    agg = AgglomerativeClustering(
                        n_clusters=k, metric="precomputed", linkage="average"
                    )
                labels = agg.fit_predict(D)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(D, labels, metric="precomputed")
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_k = k
            except Exception:
                continue

    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
        best_k = 1

    return best_labels.astype(int), int(best_k)


# ---------- Interactive Plotly Visualizations ----------

def _prim_mst_edges(sim: np.ndarray) -> List[Tuple[int, int]]:
    """Build MST on distance (1 - similarity); return list of (i,j)"""
    n = sim.shape[0]
    if n <= 1:
        return []
    A = np.array(sim, dtype=float)
    A[~np.isfinite(A)] = 0.0
    A = np.clip(A, 0.0, 1.0)
    D = 1.0 - 0.5 * (A + A.T)
    np.fill_diagonal(D, np.inf)

    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    edges = []
    min_edge = D[0, :].copy()
    parent = np.zeros(n, dtype=int)
    parent[:] = 0

    for _ in range(n - 1):
        v = np.argmin(min_edge + in_mst * np.inf)
        if not np.isfinite(min_edge[v]):
            break
        edges.append((parent[v], v))
        in_mst[v] = True
        for u in range(n):
            if not in_mst[u] and D[v, u] < min_edge[u]:
                min_edge[u] = D[v, u]
                parent[u] = v
    return edges


def _select_clean_edges(sim: np.ndarray, threshold: float, top_k: int = 3, add_mst: bool = True) -> List[Tuple[int, int, float]]:
    """Select edges for a clean graph: MST + top-k strong edges per node"""
    n = sim.shape[0]
    A = np.array(sim, dtype=float)
    A[~np.isfinite(A)] = 0.0
    A = np.clip(A, 0.0, 1.0)

    chosen = set()
    
    # Add MST edges to ensure connectivity
    if add_mst:
        for i, j in _prim_mst_edges(A):
            if i != j:
                chosen.add(tuple(sorted((i, j))))

    # Add top-k strong edges per node beyond threshold
    for i in range(n):
        cand = [(A[i, j], j) for j in range(n) if j != i and A[i, j] >= threshold]
        cand.sort(reverse=True)
        for _, j in cand[:top_k]:
            chosen.add(tuple(sorted((i, j))))

    edges = [(i, j, float(A[i, j])) for (i, j) in sorted(chosen)]
    return edges


def _convex_hull_shape(xs: np.ndarray, ys: np.ndarray, color: str, opacity: float = 0.15) -> Optional[dict]:
    """Create convex hull shape for cluster visualization"""
    if not _HAS_HULL:
        return None
    pts = np.c_[xs, ys]
    if pts.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(pts)
        verts = pts[hull.vertices]
        # Add padding to hull
        center = verts.mean(axis=0)
        verts = center + 1.15 * (verts - center)
        path = "M " + " L ".join(f"{x},{y}" for x, y in verts) + " Z"
        return dict(
            type="path",
            path=path,
            fillcolor=color,
            line=dict(color=color, width=2),
            opacity=opacity,
            layer="below",
        )
    except Exception:
        return None


def create_interactive_similarity_graph(
    sim: np.ndarray, 
    labels: List[str], 
    clusters: np.ndarray, 
    threshold: float, 
    out_html: str, 
    top_k_edges: int = 3,
    layout_method: str = "spectral"
):
    """
    Create a beautiful, interactive similarity graph using Plotly.
    Shows clusters with hulls, clean edges, and rich hover information.
    """
    if not _HAS_PLOTLY:
        print("[plotly] Not installed. Install with: pip install plotly")
        return
    
    n = sim.shape[0]
    if n == 0:
        return

    # Compute 2D layout
    coords = layout_2d_from_similarity(sim, method=layout_method)
    xs, ys = coords[:, 0], coords[:, 1]
    names = [os.path.basename(x) for x in labels]

    # Normalize coordinates for better visualization (NumPy 2.0 compatible)
    x_range = np.ptp(xs) if np.ptp(xs) > 0 else 1.0
    y_range = np.ptp(ys) if np.ptp(ys) > 0 else 1.0
    xs = (xs - xs.min()) / max(1e-8, x_range)
    ys = (ys - ys.min()) / max(1e-8, y_range)
    
    # Scale to reasonable range
    xs = xs * 100
    ys = ys * 100

    # Select clean edges
    edges = _select_clean_edges(sim, threshold=threshold, top_k=top_k_edges, add_mst=True)

    # Cluster info
    cluster_ids = clusters.astype(int)
    num_clusters = int(cluster_ids.max() + 1) if cluster_ids.size else 1
    
    # Enhanced color palette
    colors_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    node_colors = [colors_palette[c % len(colors_palette)] for c in cluster_ids]

    # Compute node metrics
    A = np.array(sim, dtype=float)
    A[~np.isfinite(A)] = 0.0
    deg = (A >= threshold).sum(axis=1) - 1
    deg = np.clip(deg, 0, None)

    # Create figure
    fig = go.Figure()

    # Add cluster hulls
    if _HAS_HULL and num_clusters > 1:
        shapes = []
        for c in range(num_clusters):
            mask = (cluster_ids == c)
            if mask.sum() >= 3:
                col = colors_palette[c % len(colors_palette)]
                shape = _convex_hull_shape(xs[mask], ys[mask], col, opacity=0.15)
                if shape:
                    shapes.append(shape)
        if shapes:
            fig.update_layout(shapes=shapes)

    # Add edges with varying opacity based on similarity
    for i, j, w in edges:
        # Normalize similarity for visual styling (handle edges below threshold from MST)
        w_normalized = max(0.0, min(1.0, w))  # Ensure w is in [0, 1]
        
        # Calculate alpha and width based on normalized similarity
        if w >= threshold:
            # Above threshold: scale between threshold and 1.0
            t = (w - threshold) / max(0.01, 1.0 - threshold)
            alpha = 0.3 + 0.5 * t
            width = 1.0 + 2.5 * t
        else:
            # Below threshold (MST edges): use minimum styling
            alpha = 0.15
            width = 0.5
        
        alpha = float(np.clip(alpha, 0.1, 0.9))
        width = float(np.clip(width, 0.3, 5.0))
        
        edge_trace = go.Scatter(
            x=[xs[i], xs[j], None],
            y=[ys[i], ys[j], None],
            mode='lines',
            line=dict(
                color=f'rgba(150, 150, 180, {alpha})',
                width=width
            ),
            hoverinfo='text',
            text=f'{names[i]} ↔ {names[j]}<br>Similarity: {w:.3f}',
            showlegend=False,
            name='edge'
        )
        fig.add_trace(edge_trace)

    # Prepare hover text with rich information
    hover_texts = []
    for i in range(n):
        # Get top similar documents
        similarities = [(A[i, j], names[j]) for j in range(n) if j != i]
        similarities.sort(reverse=True)
        top_similar = similarities[:5]
        
        similar_text = '<br>'.join([f'  • {nm}: {sim_val:.3f}' for sim_val, nm in top_similar])
        
        hover_text = (
            f"<b>{names[i]}</b><br>"
            f"<b>Cluster:</b> {int(cluster_ids[i])}<br>"
            f"<b>Connections (≥{threshold}):</b> {int(deg[i])}<br>"
            f"<br><b>Top Similar Documents:</b><br>{similar_text}"
        )
        hover_texts.append(hover_text)

    # Add nodes with size based on degree
    node_sizes = 12 + 8 * np.sqrt(deg)
    
    node_trace = go.Scatter(
        x=xs,
        y=ys,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color='white', width=2),
            opacity=0.9
        ),
        text=names,
        textposition="top center",
        textfont=dict(size=9, color='white'),
        hovertext=hover_texts,
        hoverinfo='text',
        showlegend=False,
        name='documents'
    )
    fig.add_trace(node_trace)

    # Add legend for clusters
    for c in range(num_clusters):
        cluster_size = (cluster_ids == c).sum()
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=12,
                color=colors_palette[c % len(colors_palette)],
                line=dict(color='white', width=2)
            ),
            name=f'Cluster {c} ({cluster_size} docs)',
            showlegend=True
        ))

    # Update layout with professional styling
    fig.update_layout(
        title=dict(
            text=f'📊 Document Similarity Network<br><sub>Threshold: {threshold:.2f} | Clusters: {num_clusters} | Documents: {n}</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#e8eaf6')
        ),
        template='plotly_dark',
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        width=1400,
        height=900,
        xaxis=dict(
            visible=False,
            range=[xs.min() - 5, xs.max() + 5]
        ),
        yaxis=dict(
            visible=False,
            range=[ys.min() - 5, ys.max() + 5]
        ),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            title=dict(text='<b>Clusters</b>', font=dict(size=14)),
            bgcolor='rgba(20, 25, 45, 0.8)',
            bordercolor='#3f51b5',
            borderwidth=2,
            font=dict(size=11)
        ),
        margin=dict(l=20, r=20, t=80, b=20),
    )

    # Add annotations
    fig.add_annotation(
        text=f"💡 Hover over nodes for details | Click legend to toggle clusters",
        xref="paper", yref="paper",
        x=0.5, y=-0.02,
        showarrow=False,
        font=dict(size=11, color='#9fa8da'),
        xanchor='center'
    )

    # Save to HTML
    fig.write_html(
        out_html,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'similarity_graph',
                'height': 1200,
                'width': 1600,
                'scale': 2
            }
        },
        include_plotlyjs='cdn'
    )
    print(f"✅ Interactive similarity graph saved: {out_html}")


def create_interactive_heatmap(matrix: np.ndarray, labels: List[str], title: str, out_html: str):
    """Create interactive heatmap with Plotly"""
    if not _HAS_PLOTLY:
        print("[plotly] Not installed. Skipping interactive heatmap:", title)
        return
    
    z = np.array(matrix, dtype=float)
    z[~np.isfinite(z)] = np.nan
    z = np.clip(z, 0.0, 1.0)
    
    # Upper triangle mask
    n = z.shape[0]
    mask = np.triu(np.ones_like(z, dtype=bool), k=1)
    z_plot = np.where(mask, z, np.nan)
    
    labels_short = [os.path.basename(x) for x in labels]
    
    fig = px.imshow(
        z_plot,
        x=labels_short,
        y=labels_short,
        color_continuous_scale='Viridis',
        zmin=0.0, 
        zmax=1.0,
        title=title,
        origin='upper',
        aspect='equal',
    )
    
    fig.update_layout(
        template='plotly_dark',
        width=min(1600, max(800, 50 * n)),
        height=min(1600, max(800, 50 * n)),
        margin=dict(l=80, r=40, t=80, b=80),
        coloraxis_colorbar=dict(
            title="Similarity",
            tickformat='.2f'
        ),
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
    )
    
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"✅ Interactive heatmap saved: {out_html}")


def main():
    parser = argparse.ArgumentParser(description="Compute pairwise copy-detection scores and create visualizations.")
    parser.add_argument("uploads", nargs="?", default=os.path.join(os.path.dirname(__file__), "uploads"),
                        help="Path to folder containing documents (default: backend/uploads).")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Combined score threshold to flag as likely copied (default: 0.70).")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "outputs"),
                        help="Output directory for results (default: backend/outputs).")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Disable semantic scoring (skip sentence-transformers).")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive HTML plots.")
    parser.add_argument("--layout", default="spectral", choices=["spectral", "tsne", "pca"],
                        help="Layout algorithm for similarity graph (default: spectral).")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top edges per node to display (default: 3).")
    args = parser.parse_args()

    uploads_dir = args.uploads
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    files = list_text_files(uploads_dir)
    if len(files) < 2:
        print(f"⚠️  No pairs to score. Found {len(files)} file(s) in {uploads_dir}.")
        return

    print(f"📁 Processing {len(files)} documents...")

    # Per-doc caches
    doc_text: Dict[str, str] = {}
    doc_sents: Dict[str, List[str]] = {}
    doc_sem_emb: Dict[str, Optional[np.ndarray]] = {}
    doc_style_feat: Dict[str, Optional[np.ndarray]] = {}

    # Prepare docs
    for fp in files:
        txt = read_file(fp)
        doc_text[fp] = txt
        sents = lex_sent_tokenize(txt)
        doc_sents[fp] = sents

        try:
            feats, _names = stylometric_features(txt)
            doc_style_feat[fp] = feats
        except Exception as e:
            print(f"⚠️  [stylometric] {os.path.basename(fp)}: {e}")
            doc_style_feat[fp] = None

    # Semantic embeddings
    semantic_enabled = not args.no_semantic
    if semantic_enabled:
        print("🔍 Computing semantic embeddings...")
        for fp in files:
            try:
                sents = doc_sents[fp]
                emb = embed_sentences(sents)
                doc_sem_emb[fp] = emb
            except Exception as e:
                print(f"⚠️  [semantic] Disabled: {e}")
                semantic_enabled = False
                doc_sem_emb = {k: None for k in files}
                break
    else:
        doc_sem_emb = {k: None for k in files}

    results: List[Dict[str, object]] = []
    n = len(files)
    names = [os.path.basename(p) for p in files]

    # Initialize matrices
    M_lex = np.full((n, n), np.nan, dtype=float)
    M_sem = np.full((n, n), np.nan, dtype=float)
    M_sty = np.full((n, n), np.nan, dtype=float)
    M_cmb = np.full((n, n), np.nan, dtype=float)

    print("⚙️  Computing pairwise similarities...")
    # Pairwise scoring
    for i in range(n):
        M_lex[i, i] = 1.0
        M_sem[i, i] = 1.0
        M_sty[i, i] = 1.0
        
        for j in range(i + 1, n):
            A, B = files[i], files[j]

            # Lexical
            lex_score: Optional[float] = None
            try:
                _, lex_score = lex_pair_sim(doc_sents[A], doc_sents[B])
                if lex_score is not None:
                    M_lex[i, j] = M_lex[j, i] = float(lex_score)
            except Exception as e:
                print(f"⚠️  [lexical] {names[i]} vs {names[j]}: {e}")

            # Semantic
            sem_score: Optional[float] = None
            if semantic_enabled and doc_sem_emb.get(A) is not None and doc_sem_emb.get(B) is not None:
                try:
                    sim_sem = doc_sem_emb[A] @ doc_sem_emb[B].T
                    sem_score = safe_mean_max_score(sim_sem)
                    if sem_score is not None:
                        M_sem[i, j] = M_sem[j, i] = float(sem_score)
                except Exception as e:
                    print(f"⚠️  [semantic] {names[i]} vs {names[j]}: {e}")

            # Stylometric
            sty_score: Optional[float] = None
            if doc_style_feat[A] is not None and doc_style_feat[B] is not None:
                try:
                    metrics = compare_stylometry(doc_style_feat[A], doc_style_feat[B])
                    sty_score = float(metrics.get("copying_likelihood", 0.0))
                    if sty_score is not None:
                        M_sty[i, j] = M_sty[j, i] = float(sty_score)
                except Exception as e:
                    print(f"⚠️  [style] {names[i]} vs {names[j]}: {e}")

            # Combined
            combined, _ = combine_scores({
                "lexical": lex_score,
                "semantic": sem_score,
                "stylometric": sty_score,
            })
            M_cmb[i, j] = M_cmb[j, i] = float(combined)

            decision = (
                "copied" if combined >= args.threshold
                else "suspect" if combined >= (args.threshold - 0.1)
                else "distinct"
            )
            
            results.append({
                "file_a": names[i],
                "file_b": names[j],
                "lexical": None if lex_score is None else round(lex_score, 4),
                "semantic": None if sem_score is None else round(sem_score, 4),
                "stylometric": None if sty_score is None else round(sty_score, 4),
                "combined": round(combined, 4),
                "decision": decision,
                "len_sent_a": len(doc_sents[A]),
                "len_sent_b": len(doc_sents[B]),
            })

    np.fill_diagonal(M_cmb, 1.0)

    # Save CSV/JSON
    csv_path = os.path.join(outdir, "pair_scores.csv")
    json_path = os.path.join(outdir, "pair_scores.json")
    
    fieldnames = ["file_a", "file_b", "lexical", "semantic", "stylometric", "combined", "decision", "len_sent_a", "len_sent_b"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    flagged = [r for r in results if r["decision"] == "copied"]
    print(f"\n📊 Results Summary:")
    print(f"   Total pairs: {len(results)}")
    print(f"   Flagged as copied: {len(flagged)} (threshold={args.threshold})")
    
    top = sorted(results, key=lambda r: r["combined"], reverse=True)[:5]
    print(f"\n🔝 Top 5 similar pairs:")
    for r in top:
        print(f"   {r['file_a']} ↔ {r['file_b']}")
        print(f"      Combined: {r['combined']:.3f} (lex={r['lexical']}, sem={r['semantic']}, sty={r['stylometric']})")

    print(f"\n💾 Saved: {csv_path}")
    print(f"💾 Saved: {json_path}")

    # Interactive visualizations
    if not args.no_interactive:
        if not _HAS_PLOTLY:
            print("\n⚠️  plotly not installed. Install with: pip install plotly")
        else:
            print("\n🎨 Creating interactive visualizations...")
            
            # Heatmaps
            create_interactive_heatmap(M_lex, files, "Lexical Similarity (Upper Triangle)", 
                                     os.path.join(outdir, "heatmap_lexical.html"))
            if np.isfinite(M_sem).any():
                create_interactive_heatmap(M_sem, files, "Semantic Similarity (Upper Triangle)", 
                                         os.path.join(outdir, "heatmap_semantic.html"))
            create_interactive_heatmap(M_sty, files, "Stylometric Similarity (Upper Triangle)", 
                                     os.path.join(outdir, "heatmap_stylometric.html"))
            create_interactive_heatmap(M_cmb, files, "Combined Similarity (Upper Triangle)", 
                                     os.path.join(outdir, "heatmap_combined.html"))

            # Clustering
            clusters, k = cluster_from_similarity(M_cmb, k_min=2, k_max=min(10, n-1))
            print(f"🎯 Detected {k} clusters")

            # Similarity graph
            create_interactive_similarity_graph(
                M_cmb, files, clusters, 
                threshold=args.threshold,
                out_html=os.path.join(outdir, "similarity_graph.html"),
                top_k_edges=args.top_k,
                layout_method=args.layout
            )

    print("\n✨ Done!")


if __name__ == "__main__":
    main()