import os
import json
import csv
import math
import warnings
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx

warnings.filterwarnings("ignore", category=UserWarning)

# Optional interactive plotting deps
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# For 2D layout fallbacks
try:
    from sklearn.manifold import SpectralEmbedding, TSNE
    _HAS_SE = True
except Exception:
    _HAS_SE = False

from sklearn.decomposition import PCA

# Local models
from models.lexical import sentence_tokenize as lex_sent_tokenize, compute_sentence_similarity as lex_pair_sim
from models.semantic import embed_sentences
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
            emb = TSNE(n_components=2, metric="precomputed", random_state=42, perplexity=min(30, n - 1)).fit_transform(D)
            return emb.astype(float)
        except Exception:
            pass

    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(D)
    return coords.astype(float)


def cluster_from_similarity(
    sim: np.ndarray,
    threshold: float = 0.70,
    k_min: int = 2,
    k_max: int = 10,
) -> Tuple[np.ndarray, int]:
    n = sim.shape[0]
    if n <= 1:
        return np.full(n, -1, dtype=int), 0

    A = np.array(sim, dtype=float)
    A[~np.isfinite(A)] = 0.0
    A = np.clip(A, 0.0, 1.0)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] >= threshold:
                G.add_edge(i, j, weight=float(A[i, j]))

    components = list(nx.connected_components(G))
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    for comp in components:
        comp_list = list(comp)
        if len(comp_list) >= 2:
            for node in comp_list:
                labels[node] = cluster_id
            cluster_id += 1

    return labels.astype(int), int(cluster_id)


def _collision_avoidance(xs: np.ndarray, ys: np.ndarray, min_dist: float = 30.0, passes: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    n = len(xs)
    for _ in range(passes):
        moved = False
        for a in range(n):
            for b in range(a + 1, n):
                dx = xs[a] - xs[b]
                dy = ys[a] - ys[b]
                dist = math.hypot(dx, dy)
                if dist < min_dist and dist > 1e-6:
                    push = (min_dist - dist) / 2.0
                    ux = dx / dist
                    uy = dy / dist
                    xs[a] += ux * push
                    ys[a] += uy * push
                    xs[b] -= ux * push
                    ys[b] -= uy * push
                    moved = True
        if not moved:
            break
    return xs, ys


def _edge_color_for_similarity(s: float) -> str:
    """
    Color buckets:
    - Red: Highly similar (0.70–1.00) → possible copying
    - Orange: Moderately similar (0.50–0.70)
    """
    if s >= 0.70:
        return '#ef4444'  # red
    if s >= 0.50:
        return '#f59e0b'  # orange
    # No color returned for low similarity; we won't draw those edges.
    return ''


def create_document_similarity_network(
    sim: np.ndarray,
    labels: List[str],
    threshold: float,
    out_html: str,
):
    """
    Document Similarity Graph (Network View)

    - Nodes: circular, labeled with document name. Size optionally encodes degree.
    - Edge length inversely proportional to similarity using spring_layout:
        desired_length_ij ≈ 1 / max(sim_ij, 1e-6)
      We set layout k ≈ median(desired_length) and weight = similarity to pull close pairs together.
    - Edge colors: red (0.70–1.00), orange (0.50–0.70). Low or no similarity edges are not drawn.
    - Edge width constant; proximity encodes similarity.
    - Hover node: high-similarity neighbors (≥ threshold) and all scores.
    - Hover edge: "A–B: 0.82".
    - Edges ≥ threshold get a red glow underlay.
    - Zoom/pan enabled; suitable for dashboards.
    """
    if not _HAS_PLOTLY:
        print("[plotly] Not installed. Install with: pip install plotly")
        return

    n = sim.shape[0]
    if n == 0:
        return

    # Normalize and symmetrize
    S = np.array(sim, dtype=float)
    S[~np.isfinite(S)] = 0.0
    S = np.clip(S, 0.0, 1.0)
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 0.0)

    names = [os.path.basename(x) for x in labels]

    # Spring layout graph: include all pairs but weight by similarity
    G = nx.Graph()
    G.add_nodes_from(range(n))
    desired_lengths: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(S[i, j])
            G.add_edge(i, j, weight=max(s, 1e-6))
            desired_lengths.append(1.0 / max(s, 1e-6))

    # Layout with k ~ median desired length
    if G.number_of_edges() > 0:
        k_val = float(np.median(desired_lengths)) if desired_lengths else 1.0
        pos = nx.spring_layout(G, weight='weight', k=k_val, seed=42, iterations=250)
        xs = np.array([pos[i][0] for i in range(n)], dtype=float)
        ys = np.array([pos[i][1] for i in range(n)], dtype=float)
    else:
        coords = layout_2d_from_similarity(S, method="pca")
        xs, ys = coords[:, 0], coords[:, 1]

    # Normalize and avoid collisions
    xs = (xs - xs.mean()) / max(xs.std(), 1e-6) * 320.0
    ys = (ys - ys.mean()) / max(ys.std(), 1e-6) * 320.0
    xs, ys = _collision_avoidance(xs, ys, min_dist=34.0, passes=240)

    # Degree above threshold (number of connections with sim ≥ threshold)
    deg = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and S[i, j] >= threshold:
                deg[i] += 1

    # Node hover: high-similarity names and all scores
    node_hover = []
    for i in range(n):
        high = [(names[j], S[i, j]) for j in range(n) if j != i and S[i, j] >= threshold]
        high.sort(key=lambda x: x[1], reverse=True)
        high_html = "<br>".join([f"{nm}: {sc:.3f}" for nm, sc in high]) if high else "None"

        all_scores = [(names[j], S[i, j]) for j in range(n) if j != i]
        all_scores.sort(key=lambda x: x[1], reverse=True)
        all_html = "<br>".join([f"{nm}: {sc:.3f}" for nm, sc in all_scores])

        node_hover.append(
            f"<b>{names[i]}</b><br>"
            f"Connections ≥ {threshold:.2f}: {deg[i]}<br><br>"
            f"<b>High similarity (≥ {threshold:.2f}):</b><br>{high_html}<br><br>"
            f"<b>All similarity scores:</b><br>{all_html}"
        )

    # Build edge traces: only draw edges for s ≥ 0.50 (orange) or s ≥ 0.70 (red)
    edge_x_red, edge_y_red, edge_text_red = [], [], []
    edge_x_orange, edge_y_orange, edge_text_orange = [], [], []
    glow_x, glow_y = [], []

    for i in range(n):
        for j in range(i + 1, n):
            s = float(S[i, j])
            if s < 0.50:
                continue  # do not draw low or no similarity edges
            color = _edge_color_for_similarity(s)
            text = f"{names[i]}–{names[j]}: {s:.3f}"
            if color == '#ef4444':
                edge_x_red.extend([xs[i], xs[j], None])
                edge_y_red.extend([ys[i], ys[j], None])
                edge_text_red.append(text)
            elif color == '#f59e0b':
                edge_x_orange.extend([xs[i], xs[j], None])
                edge_y_orange.extend([ys[i], ys[j], None])
                edge_text_orange.append(text)

            if s >= threshold:
                glow_x.extend([xs[i], xs[j], None])
                glow_y.extend([ys[i], ys[j], None])

    fig = go.Figure()

    # Red glow for edges above threshold
    if glow_x:
        fig.add_trace(go.Scatter(
            x=glow_x, y=glow_y, mode='lines',
            line=dict(width=16, color='rgba(239,68,68,0.20)'),
            hoverinfo='skip',
            showlegend=False
        ))

    # Bucket traces (no grey bucket)
    if edge_x_red:
        fig.add_trace(go.Scatter(
            x=edge_x_red, y=edge_y_red, mode='lines',
            line=dict(width=2.0, color='#ef4444'),
            hoverinfo='text', text=edge_text_red, name='High (0.70–1.00)'
        ))
    if edge_x_orange:
        fig.add_trace(go.Scatter(
            x=edge_x_orange, y=edge_y_orange, mode='lines',
            line=dict(width=2.0, color='#f59e0b'),
            hoverinfo='text', text=edge_text_orange, name='Moderate (0.50–0.70)'
        ))

    # Nodes: circular, show doc name and optionally degree ≥ threshold
    node_labels = [f"{names[i]} ({deg[i]})" if deg[i] > 0 else f"{names[i]}" for i in range(n)]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='markers+text',
        marker=dict(
            size=[16 + 6 * math.sqrt(max(0, d)) for d in deg],
            color='#3b82f6',
            line=dict(color='white', width=2),
            symbol='circle',
            opacity=0.95
        ),
        text=node_labels,
        textposition="bottom center",
        textfont=dict(size=12, color='white'),
        hovertext=node_hover,
        hoverinfo='text',
        name='Documents'
    ))

    fig.update_layout(
        title=dict(
            text=f"Document Similarity Network<br>"
                 f"<sub>Distance encodes similarity (shorter = more similar). Red glow for edges ≥ {threshold:.2f}.</sub>",
            x=0.5, xanchor='center', font=dict(size=20, color='#e8eaf6')
        ),
        template='plotly_dark',
        plot_bgcolor='#0a0e27', paper_bgcolor='#0a0e27',
        width=1600, height=1000,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            title=dict(text='<b>Edge similarity buckets</b>', font=dict(size=13)),
            bgcolor='rgba(20, 25, 45, 0.92)',
            bordercolor='#3f51b5', borderwidth=2, font=dict(size=11),
            x=1.02, y=1, xanchor='left', yanchor='top'
        ),
        margin=dict(l=20, r=280, t=100, b=20),
        annotations=[
            dict(
                text="Scroll to zoom, drag to pan. Click legend to focus/show/hide buckets.",
                showarrow=False, xref="paper", yref="paper", x=0.02, y=0.98,
                xanchor='left', yanchor='top', font=dict(size=12, color='#9fa8da')
            )
        ]
    )

    fig.write_html(
        out_html,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {'format': 'png', 'filename': 'doc_similarity_network', 'height': 1400, 'width': 2000, 'scale': 2}
        },
        include_plotlyjs='cdn'
    )
    print(f"✅ Network map saved: {out_html}")


# Compatibility wrapper for main.py
def create_interactive_similarity_graph(
    sim: np.ndarray,
    labels: List[str],
    clusters: np.ndarray,
    threshold: float,
    out_html: str,
    top_k_edges: int = 3,
    layout_method: str = "spring"
):
    return create_document_similarity_network(
        sim=sim,
        labels=labels,
        threshold=threshold,
        out_html=out_html,
    )


def create_interactive_heatmap(matrix: np.ndarray, labels: List[str], title: str, out_html: str):
    if not _HAS_PLOTLY:
        print("[plotly] Not installed. Skipping interactive heatmap:", title)
        return

    z = np.array(matrix, dtype=float)
    z[~np.isfinite(z)] = np.nan
    z = np.clip(z, 0.0, 1.0)

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
        coloraxis_colorbar=dict(title="Similarity", tickformat='.2f'),
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
    parser.add_argument("--layout", default="spring", choices=["spring", "kamada_kawai", "spectral", "tsne", "pca"],
                        help="Layout algorithm for similarity graph (default: spring).")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Reserved (not used in this view).")
    args = parser.parse_args()

    uploads_dir = args.uploads
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    files = list_text_files(uploads_dir)
    if len(files) < 2:
        print(f"⚠️  No pairs to score. Found {len(files)} file(s) in {uploads_dir}.")
        return

    print(f"📁 Processing {len(files)} documents...")

    doc_sents: Dict[str, List[str]] = {}
    doc_sem_emb: Dict[str, Optional[np.ndarray]] = {}
    doc_style_feat: Dict[str, Optional[np.ndarray]] = {}

    for fp in files:
        txt = read_file(fp)
        sents = lex_sent_tokenize(txt)
        doc_sents[fp] = sents
        try:
            feats, _names = stylometric_features(txt)
            doc_style_feat[fp] = feats
        except Exception as e:
            print(f"⚠️  [stylometric] {os.path.basename(fp)}: {e}")
            doc_style_feat[fp] = None

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

    M_lex = np.full((n, n), np.nan, dtype=float)
    M_sem = np.full((n, n), np.nan, dtype=float)
    M_sty = np.full((n, n), np.nan, dtype=float)
    M_cmb = np.full((n, n), np.nan, dtype=float)

    print("⚙️  Computing pairwise similarities...")
    for i in range(n):
        M_lex[i, i] = 1.0
        M_sem[i, i] = 1.0
        M_sty[i, i] = 1.0
        for j in range(i + 1, n):
            A_fp, B_fp = files[i], files[j]

            # Lexical
            lex_score: Optional[float] = None
            try:
                _, lex_score = lex_pair_sim(doc_sents[A_fp], doc_sents[B_fp])
                if lex_score is not None:
                    M_lex[i, j] = M_lex[j, i] = float(lex_score)
            except Exception as e:
                print(f"⚠️  [lexical] {names[i]} vs {names[j]}: {e}")

            # Semantic
            sem_score: Optional[float] = None
            if semantic_enabled and doc_sem_emb.get(A_fp) is not None and doc_sem_emb.get(B_fp) is not None:
                try:
                    sim_sem = doc_sem_emb[A_fp] @ doc_sem_emb[B_fp].T
                    sem_score = safe_mean_max_score(sim_sem)
                    if sem_score is not None:
                        M_sem[i, j] = M_sem[j, i] = float(sem_score)
                except Exception as e:
                    print(f"⚠️  [semantic] {names[i]} vs {names[j]}: {e}")

            # Stylometric
            sty_score: Optional[float] = None
            if doc_style_feat[A_fp] is not None and doc_style_feat[B_fp] is not None:
                try:
                    metrics = compare_stylometry(doc_style_feat[A_fp], doc_style_feat[B_fp])
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
                "len_sent_a": len(doc_sents[A_fp]),
                "len_sent_b": len(doc_sents[B_fp]),
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

    if not args.no_interactive:
        print("\nℹ️ Interactive plots disabled.")
        return

    if not _HAS_PLOTLY:
        print("\n⚠️  plotly not installed. Install with: pip install plotly")
        return

    print("\n🎨 Creating interactive visualizations...")
    if np.isfinite(M_lex).any():
        create_interactive_heatmap(M_lex, files, "Lexical Similarity (Upper Triangle)",
                                   os.path.join(outdir, "heatmap_lexical.html"))
    if np.isfinite(M_sem).any():
        create_interactive_heatmap(M_sem, files, "Semantic Similarity (Upper Triangle)",
                                   os.path.join(outdir, "heatmap_semantic.html"))
    if np.isfinite(M_sty).any():
        create_interactive_heatmap(M_sty, files, "Stylometric Similarity (Upper Triangle)",
                                   os.path.join(outdir, "heatmap_stylometric.html"))
    create_interactive_heatmap(M_cmb, files, "Combined Similarity (Upper Triangle)",
                               os.path.join(outdir, "heatmap_combined.html"))

    create_document_similarity_network(
        M_cmb, files,
        threshold=args.threshold,
        out_html=os.path.join(outdir, "network_similarity.html"),
    )

    print("\n✨ Done!")


if __name__ == "__main__":
    main()