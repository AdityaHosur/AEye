import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import numpy as np
import json

# OCR
from models.ocr import (
    extract_text_from_handwritten_image,
    is_handwritten_image,
    is_trocr_available,
    get_available_engines
)

# Scoring utilities
from score import (
    list_text_files,
    read_file,
    safe_mean_max_score,
    combine_scores,
    cluster_from_similarity,
    create_interactive_similarity_graph,
    create_interactive_heatmap,
)
from models.lexical import sentence_tokenize as lex_sent_tokenize, compute_sentence_similarity as lex_pair_sim
from models.semantic import embed_sentences
from models.style import stylometric_features, compare_stylometry

app = FastAPI(title="AEye - Document Similarity Detection API with OCR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_DIR = BASE_DIR / "images"
for d in (UPLOAD_DIR, OUTPUT_DIR, IMAGE_DIR):
    d.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

doc_cache: Dict[str, Any] = {}


class ComparePairRequest(BaseModel):
    file_a: str
    file_b: str


@app.get("/")
async def root():
    return {
        "message": "AEye API is running with OCR support",
        "version": "2.1.0",
        "features": {
            "text_upload": True,
            "image_ocr": is_trocr_available(),
            "ocr_engines": get_available_engines()
        }
    }


@app.get("/ocr-status")
async def ocr_status():
    return {
        "available": is_trocr_available(),
        "engines": get_available_engines(),
        "supported_formats": ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    }


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="Please upload at least 1 document")
    uploaded, skipped = [], []
    for file in files:
        if not file.filename.endswith(('.txt', '.md', '.text', '.log')):
            skipped.append({"filename": file.filename, "reason": "Unsupported file type"})
            continue
        fp = UPLOAD_DIR / file.filename
        content = await file.read()
        with open(fp, "wb") as f:
            f.write(content)
        uploaded.append({
            "filename": file.filename,
            "size": len(content),
            "size_kb": round(len(content) / 1024, 2),
            "path": str(fp),
            "type": "text"
        })
    if not uploaded:
        raise HTTPException(status_code=400, detail="No valid text files uploaded")
    return {"message": f"Uploaded {len(uploaded)} file(s)", "files": uploaded, "skipped": skipped or None}


@app.post("/upload-images")
async def upload_images(files: List[UploadFile] = File(...), split_lines: bool = Form(True)):
    if not is_trocr_available():
        raise HTTPException(status_code=503, detail="OCR engine unavailable – install dependencies")
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="Please upload at least 1 image")

    processed, errors = [], []
    for file in files:
        try:
            if not is_handwritten_image(file.filename):
                errors.append({"filename": file.filename, "error": "Unsupported image type"})
                continue
            img_path = IMAGE_DIR / file.filename
            content = await file.read()
            with open(img_path, "wb") as f:
                f.write(content)
            text_filename = Path(file.filename).stem + ".txt"
            text_path = UPLOAD_DIR / text_filename
            result = extract_text_from_handwritten_image(
                str(img_path),
                output_path=str(text_path),
                split_lines=split_lines,
                save_preprocessed=True
            )
            processed.append({
                "original_filename": file.filename,
                "text_filename": text_filename,
                "image_size_kb": round(len(content) / 1024, 2),
                "extracted_text_preview": (result['text'][:200] + "...") if len(result['text']) > 200 else result['text'],
                "confidence": result['confidence'],
                "statistics": result['statistics'],
                "engine": result['engine'],
                "success": True
            })
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
            if img_path.exists():
                img_path.unlink()
    if not processed:
        raise HTTPException(status_code=400, detail=f"OCR failed for all images: {errors}")
    return {
        "message": f"Processed {len(processed)} image(s)",
        "processed": processed,
        "errors": errors or None,
        "success_count": len(processed),
        "error_count": len(errors)
    }


@app.get("/files")
async def get_files():
    files = list_text_files(str(UPLOAD_DIR))
    info = []
    for fp in files:
        p = Path(fp)
        info.append({"filename": p.name, "path": fp, "size_kb": round(p.stat().st_size / 1024, 2), "type": "text"})
    return {"count": len(info), "files": info}


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    fp = UPLOAD_DIR / filename
    if not fp.exists():
        raise HTTPException(status_code=404, detail="File not found")
    fp.unlink()
    global doc_cache
    if doc_cache and filename in [os.path.basename(f) for f in doc_cache.get("files", [])]:
        doc_cache = {}
    return {"success": True, "message": f"Deleted {filename}"}


@app.delete("/files")
async def clear_files():
    deleted_text = sum(1 for f in UPLOAD_DIR.glob("*") if f.is_file() and not f.unlink())
    deleted_images = sum(1 for f in IMAGE_DIR.glob("*") if f.is_file() and not f.unlink())
    global doc_cache
    doc_cache = {}
    return {
        "success": True,
        "message": f"Cleared {deleted_text} text and {deleted_images} image files",
        "text_files_deleted": deleted_text,
        "images_deleted": deleted_images
    }


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    stop = {
        'the','and','a','an','of','to','in','is','it','that','for','on','as','with','this','by','be','are','or','from',
        'at','was','were','has','have','had','but','not','can','could','should','would','will','do','does','did','we',
        'you','he','she','they','them','their','our','my','your','i'
    }
    return [t for t in tokens if t not in stop and len(t) > 2]


def _describe_pair(lex: Optional[float], sem: Optional[float], combined: Optional[float], threshold: float) -> str:
    parts = []
    if lex is not None:
        parts.append(f"Lexical {lex:.3f}")
    else:
        parts.append("Lexical N/A")
    if sem is not None:
        parts.append(f"Semantic {sem:.3f}")
    else:
        parts.append("Semantic N/A")
    if combined is not None:
        if combined >= threshold:
            verdict = "High copying likelihood"
        elif combined >= (threshold - 0.1):
            verdict = "Borderline overlap"
        else:
            verdict = "Distinct"
        parts.append(f"Combined {combined:.3f} → {verdict}")
    else:
        parts.append("Combined N/A")
    return " | ".join(parts)


def _highlight_sentences_and_words(
    text_a: str,
    sents_a: List[str],
    emb_a: Optional[np.ndarray],
    text_b: str,
    sents_b: List[str],
    emb_b: Optional[np.ndarray],
    sem_threshold: float = 0.80,
    top_words: int = 20
) -> Tuple[str, str, Dict[str, Any]]:
    high_pairs: List[Tuple[int, int, float]] = []
    if emb_a is not None and emb_b is not None:
        try:
            def norm(m: np.ndarray) -> np.ndarray:
                d = np.linalg.norm(m, axis=1, keepdims=True) + 1e-8
                return m / d
            A = norm(emb_a)
            B = norm(emb_b)
            sim = A @ B.T
            for i in range(sim.shape[0]):
                j = int(np.argmax(sim[i]))
                sc = float(sim[i, j])
                if sc >= sem_threshold:
                    high_pairs.append((i, j, sc))
        except Exception:
            pass

    toks_a = _tokenize(text_a)
    toks_b = _tokenize(text_b)
    fa, fb = {}, {}
    for t in toks_a:
        fa[t] = fa.get(t, 0) + 1
    for t in toks_b:
        fb[t] = fb.get(t, 0) + 1
    common = [(w, fa[w] + fb[w]) for w in (set(fa) & set(fb))]
    common.sort(key=lambda x: x[1], reverse=True)
    highlight_words = set([w for w, _ in common[:top_words]])

    a_high = {i for i, _, _ in high_pairs}
    b_high = {j for _, j, _ in high_pairs}

    def render(sents: List[str], idxs: set) -> str:
        def span_words(s: str) -> str:
            return re.sub(
                r"[A-Za-z]+",
                lambda m: f'<span class="hl-word">{m.group(0)}</span>' if m.group(0).lower() in highlight_words else m.group(0),
                s
            )
        blocks = []
        for i, s in enumerate(sents):
            sp = span_words(s)
            if i in idxs:
                blocks.append(f'<p class="hl-sent">{sp}</p>')
            else:
                blocks.append(f'<p>{sp}</p>')
        return "\n".join(blocks)

    html_a = render(sents_a, a_high)
    html_b = render(sents_b, b_high)
    meta = {
        "high_sentence_pairs": [
            {
                "a_index": i,
                "b_index": j,
                "score": round(sc, 3),
                "a_text": sents_a[i],
                "b_text": sents_b[j]
            } for (i, j, sc) in high_pairs
        ],
        "highlight_words": sorted(highlight_words)
    }
    return html_a, html_b, meta


@app.post("/detect-similarity")
async def detect_similarity(threshold: float = 0.70):
    files = list_text_files(str(UPLOAD_DIR))
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 files to compare")

    doc_text: Dict[str, str] = {}
    doc_sents: Dict[str, List[str]] = {}
    doc_sem_emb: Dict[str, Optional[np.ndarray]] = {}
    doc_style_feat: Dict[str, Optional[np.ndarray]] = {}

    for fp in files:
        txt = read_file(fp)
        doc_text[fp] = txt
        sents = lex_sent_tokenize(txt)
        doc_sents[fp] = sents
        try:
            feats, _ = stylometric_features(txt)
            doc_style_feat[fp] = feats
        except Exception:
            doc_style_feat[fp] = None

    semantic_enabled = True
    for fp in files:
        try:
            emb = embed_sentences(doc_sents[fp])
            doc_sem_emb[fp] = emb
        except Exception:
            semantic_enabled = False
            doc_sem_emb = {k: None for k in files}
            break

    n = len(files)
    names = [os.path.basename(p) for p in files]
    M_lex = np.full((n, n), np.nan)
    M_sem = np.full((n, n), np.nan)
    M_sty = np.full((n, n), np.nan)
    M_cmb = np.full((n, n), np.nan)

    results = []
    for i in range(n):
        M_lex[i, i] = M_sem[i, i] = M_sty[i, i] = 1.0
        for j in range(i + 1, n):
            A, B = files[i], files[j]
            lex_score = None
            try:
                _, lex_score = lex_pair_sim(doc_sents[A], doc_sents[B])
                if lex_score is not None:
                    M_lex[i, j] = M_lex[j, i] = lex_score
            except Exception:
                pass
            sem_score = None
            if semantic_enabled and doc_sem_emb[A] is not None and doc_sem_emb[B] is not None:
                try:
                    sim_sem = doc_sem_emb[A] @ doc_sem_emb[B].T
                    sem_score = safe_mean_max_score(sim_sem)
                    if sem_score is not None:
                        M_sem[i, j] = M_sem[j, i] = sem_score
                except Exception:
                    pass
            sty_score = None
            if doc_style_feat[A] is not None and doc_style_feat[B] is not None:
                try:
                    metrics = compare_stylometry(doc_style_feat[A], doc_style_feat[B])
                    sty_score = float(metrics.get("copying_likelihood", 0.0))
                    if sty_score is not None:
                        M_sty[i, j] = M_sty[j, i] = sty_score
                except Exception:
                    pass
            combined, _ = combine_scores({"lexical": lex_score, "semantic": sem_score, "stylometric": sty_score})
            M_cmb[i, j] = M_cmb[j, i] = combined
            decision = "copied" if combined >= threshold else "suspect" if combined >= (threshold - 0.1) else "distinct"
            results.append({
                "file_a": names[i],
                "file_b": names[j],
                "lexical": None if lex_score is None else round(lex_score, 4),
                "semantic": None if sem_score is None else round(sem_score, 4),
                "stylometric": None if sty_score is None else round(sty_score, 4),
                "combined": round(combined, 4),
                "decision": decision
            })

    np.fill_diagonal(M_cmb, 1.0)

    with open(OUTPUT_DIR / "pair_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    create_interactive_heatmap(M_lex, files, "Lexical Similarity", str(OUTPUT_DIR / "heatmap_lexical.html"))
    if np.isfinite(M_sem).any():
        create_interactive_heatmap(M_sem, files, "Semantic Similarity", str(OUTPUT_DIR / "heatmap_semantic.html"))
    create_interactive_heatmap(M_sty, files, "Stylometric Similarity", str(OUTPUT_DIR / "heatmap_stylometric.html"))
    create_interactive_heatmap(M_cmb, files, "Combined Similarity", str(OUTPUT_DIR / "heatmap_combined.html"))

    clusters, k = cluster_from_similarity(M_cmb, k_min=2, k_max=min(10, n - 1))
    cluster_map: Dict[int, List[str]] = {}
    for idx, cid in enumerate(clusters.tolist()):
        if cid >= 0:
            cluster_map.setdefault(cid, []).append(names[idx])
    clusters_detail = [{"id": cid, "documents": sorted(docs)} for cid, docs in sorted(cluster_map.items())]

    create_interactive_similarity_graph(
        M_cmb, files, clusters,
        threshold=threshold,
        out_html=str(OUTPUT_DIR / "similarity_graph.html"),
        top_k_edges=3,
        layout_method="spring"
    )

    flagged = [r for r in results if r["decision"] == "copied"]

    global doc_cache
    doc_cache = {
        "files": files,
        "names": names,
        "doc_text": doc_text,
        "doc_sents": doc_sents,
        "doc_sem_emb": doc_sem_emb,
        "matrices": {
            "lexical": M_lex,
            "semantic": M_sem,
            "stylometric": M_sty,
            "combined": M_cmb
        },
        "threshold": threshold,
        "clusters_detail": clusters_detail
    }

    return {
        "success": True,
        "summary": {
            "total_documents": n,
            "total_pairs": len(results),
            "flagged_copied": len(flagged),
            "threshold": threshold,
            "clusters": k
        },
        "clusters_detail": clusters_detail,
        "results": results,
        "visualizations": {
            "heatmap_lexical": "/outputs/heatmap_lexical.html",
            "heatmap_semantic": "/outputs/heatmap_semantic.html" if np.isfinite(M_sem).any() else None,
            "heatmap_stylometric": "/outputs/heatmap_stylometric.html",
            "heatmap_combined": "/outputs/heatmap_combined.html",
            "similarity_graph": "/outputs/similarity_graph.html"
        }
    }


@app.post("/compare-pair")
async def compare_pair(request: ComparePairRequest):
    if not doc_cache:
        raise HTTPException(status_code=400, detail="Run detection first")
    file_a, file_b = request.file_a, request.file_b
    names = doc_cache["names"]
    try:
        ia = names.index(file_a)
        ib = names.index(file_b)
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found")

    M = doc_cache["matrices"]
    lex = float(M["lexical"][ia, ib]); sem = float(M["semantic"][ia, ib])
    sty = float(M["stylometric"][ia, ib]); cmb = float(M["combined"][ia, ib])

    lex = None if np.isnan(lex) else round(lex, 4)
    sem = None if np.isnan(sem) else round(sem, 4)
    sty = None if np.isnan(sty) else round(sty, 4)
    cmb = None if np.isnan(cmb) else round(cmb, 4)

    threshold = doc_cache.get("threshold", 0.70)
    description = _describe_pair(lex, sem, cmb, threshold)

    if cmb is None or cmb < 0.6:
        verdict, confidence = "distinct", "high"
    elif cmb < 0.7:
        verdict, confidence = "suspect", "medium"
    else:
        verdict, confidence = "copied", "high" if cmb >= 0.85 else "medium"

    return {
        "file_a": file_a,
        "file_b": file_b,
        "scores": {"lexical": lex, "semantic": sem, "stylometric": sty, "combined": cmb},
        "verdict": verdict,
        "confidence": confidence,
        "description": description,
        "details": {
            "sentences_a": len(doc_cache["doc_sents"][doc_cache["files"][ia]]),
            "sentences_b": len(doc_cache["doc_sents"][doc_cache["files"][ib]])
        }
    }


@app.post("/compare-pair-detailed")
async def compare_pair_detailed(request: ComparePairRequest):
    if not doc_cache:
        raise HTTPException(status_code=400, detail="Run detection first")
    file_a, file_b = request.file_a, request.file_b
    names = doc_cache["names"]; files = doc_cache["files"]
    try:
        ia = names.index(file_a); ib = names.index(file_b)
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found")

    M = doc_cache["matrices"]
    lex = float(M["lexical"][ia, ib]); sem = float(M["semantic"][ia, ib])
    sty = float(M["stylometric"][ia, ib]); cmb = float(M["combined"][ia, ib])
    lex = None if np.isnan(lex) else round(lex, 4)
    sem = None if np.isnan(sem) else round(sem, 4)
    sty = None if np.isnan(sty) else round(sty, 4)
    cmb = None if np.isnan(cmb) else round(cmb, 4)

    threshold = doc_cache.get("threshold", 0.70)
    description = _describe_pair(lex, sem, cmb, threshold)

    text_a = doc_cache["doc_text"][files[ia]]
    text_b = doc_cache["doc_text"][files[ib]]
    sents_a = doc_cache["doc_sents"][files[ia]]
    sents_b = doc_cache["doc_sents"][files[ib]]
    emb_a = doc_cache["doc_sem_emb"][files[ia]]
    emb_b = doc_cache["doc_sem_emb"][files[ib]]

    html_a, html_b, meta = _highlight_sentences_and_words(
        text_a, sents_a, emb_a,
        text_b, sents_b, emb_b,
        sem_threshold=0.80,
        top_words=20
    )

    if cmb is None or cmb < 0.6:
        verdict, confidence = "distinct", "high"
    elif cmb < 0.7:
        verdict, confidence = "suspect", "medium"
    else:
        verdict, confidence = "copied", "high" if cmb >= 0.85 else "medium"

    return {
        "file_a": file_a,
        "file_b": file_b,
        "scores": {"lexical": lex, "semantic": sem, "stylometric": sty, "combined": cmb},
        "verdict": verdict,
        "confidence": confidence,
        "description": description,
        "details": {
            "sentences_a": len(sents_a),
            "sentences_b": len(sents_b)
        },
        "highlights": {
            "html_a": f'<div class="doc-highlight">{html_a}</div>',
            "html_b": f'<div class="doc-highlight">{html_b}</div>',
            "meta": meta
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)