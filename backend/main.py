import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import scoring logic
from score import (
    list_text_files,
    read_file,
    safe_mean_max_score,
    combine_scores,
    layout_2d_from_similarity,
    cluster_from_similarity,
    create_interactive_similarity_graph,
    create_interactive_heatmap,
)
from models.lexical import sentence_tokenize as lex_sent_tokenize, compute_sentence_similarity as lex_pair_sim
from models.semantic import sentence_tokenize as sem_sent_tokenize, embed_sentences
from models.style import stylometric_features, compare_stylometry

import numpy as np
import json

app = FastAPI(title="AEye - Document Similarity Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount outputs folder for serving HTML files
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Global cache for processed documents
doc_cache: Dict[str, Any] = {}


# Pydantic models for request validation
class ComparePairRequest(BaseModel):
    file_a: str
    file_b: str


@app.get("/")
async def root():
    return {"message": "AEye API is running", "version": "1.0.0"}


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload multiple documents"""
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Please upload at least 2 documents")
    
    uploaded = []
    for file in files:
        if not file.filename.endswith(('.txt', '.md', '.text', '.log')):
            continue
        
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        uploaded.append({
            "filename": file.filename,
            "size": len(content),
            "path": str(file_path)
        })
    
    return {
        "message": f"Successfully uploaded {len(uploaded)} files",
        "files": uploaded
    }


@app.get("/files")
async def get_files():
    """Get list of uploaded files"""
    files = list_text_files(str(UPLOAD_DIR))
    return {
        "count": len(files),
        "files": [{"filename": os.path.basename(f), "path": f} for f in files]
    }


@app.delete("/files")
async def clear_files():
    """Clear all uploaded files"""
    for file in UPLOAD_DIR.glob("*"):
        if file.is_file():
            file.unlink()
    
    # Clear cache
    global doc_cache
    doc_cache = {}
    
    return {"message": "All files cleared"}


@app.post("/detect-similarity")
async def detect_similarity(threshold: float = 0.70):
    """
    Run similarity detection on all uploaded files.
    Returns results and generates visualizations.
    """
    files = list_text_files(str(UPLOAD_DIR))
    
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 files to compare")
    
    print(f"🔍 Processing {len(files)} documents...")
    
    # Process documents
    doc_text = {}
    doc_sents = {}
    doc_sem_emb = {}
    doc_style_feat = {}
    
    for fp in files:
        txt = read_file(fp)
        doc_text[fp] = txt
        sents = lex_sent_tokenize(txt)
        doc_sents[fp] = sents
        
        # Stylometric features
        try:
            feats, _ = stylometric_features(txt)
            doc_style_feat[fp] = feats
        except Exception as e:
            print(f"⚠️  [stylometric] {os.path.basename(fp)}: {e}")
            doc_style_feat[fp] = None
    
    # Semantic embeddings
    print("🔍 Computing semantic embeddings...")
    semantic_enabled = True
    for fp in files:
        try:
            sents = doc_sents[fp]
            emb = embed_sentences(sents)
            doc_sem_emb[fp] = emb
        except Exception as e:
            print(f"⚠️  [semantic] Error: {e}")
            semantic_enabled = False
            doc_sem_emb = {k: None for k in files}
            break
    
    # Initialize matrices
    n = len(files)
    names = [os.path.basename(p) for p in files]
    M_lex = np.full((n, n), np.nan, dtype=float)
    M_sem = np.full((n, n), np.nan, dtype=float)
    M_sty = np.full((n, n), np.nan, dtype=float)
    M_cmb = np.full((n, n), np.nan, dtype=float)
    
    results = []
    
    print("⚙️  Computing pairwise similarities...")
    for i in range(n):
        M_lex[i, i] = 1.0
        M_sem[i, i] = 1.0
        M_sty[i, i] = 1.0
        
        for j in range(i + 1, n):
            A, B = files[i], files[j]
            
            # Lexical
            lex_score = None
            try:
                _, lex_score = lex_pair_sim(doc_sents[A], doc_sents[B])
                if lex_score is not None:
                    M_lex[i, j] = M_lex[j, i] = float(lex_score)
            except Exception as e:
                print(f"⚠️  [lexical] {names[i]} vs {names[j]}: {e}")
            
            # Semantic
            sem_score = None
            if semantic_enabled and doc_sem_emb.get(A) is not None and doc_sem_emb.get(B) is not None:
                try:
                    sim_sem = doc_sem_emb[A] @ doc_sem_emb[B].T
                    sem_score = safe_mean_max_score(sim_sem)
                    if sem_score is not None:
                        M_sem[i, j] = M_sem[j, i] = float(sem_score)
                except Exception:
                    pass
            
            # Stylometric
            sty_score = None
            if doc_style_feat[A] is not None and doc_style_feat[B] is not None:
                try:
                    metrics = compare_stylometry(doc_style_feat[A], doc_style_feat[B])
                    sty_score = float(metrics.get("copying_likelihood", 0.0))
                    if sty_score is not None:
                        M_sty[i, j] = M_sty[j, i] = float(sty_score)
                except Exception:
                    pass
            
            # Combined
            combined, weights = combine_scores({
                "lexical": lex_score,
                "semantic": sem_score,
                "stylometric": sty_score,
            })
            M_cmb[i, j] = M_cmb[j, i] = float(combined)
            
            decision = (
                "copied" if combined >= threshold
                else "suspect" if combined >= (threshold - 0.1)
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
            })
    
    np.fill_diagonal(M_cmb, 1.0)
    
    # Save results
    json_path = OUTPUT_DIR / "pair_scores.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Generate visualizations
    print("🎨 Creating interactive visualizations...")
    
    create_interactive_heatmap(M_lex, files, "Lexical Similarity", 
                             str(OUTPUT_DIR / "heatmap_lexical.html"))
    if np.isfinite(M_sem).any():
        create_interactive_heatmap(M_sem, files, "Semantic Similarity", 
                                 str(OUTPUT_DIR / "heatmap_semantic.html"))
    create_interactive_heatmap(M_sty, files, "Stylometric Similarity", 
                             str(OUTPUT_DIR / "heatmap_stylometric.html"))
    create_interactive_heatmap(M_cmb, files, "Combined Similarity", 
                             str(OUTPUT_DIR / "heatmap_combined.html"))
    
    # Clustering and graph
    clusters, k = cluster_from_similarity(M_cmb, k_min=2, k_max=min(10, n-1))
    print(f"🎯 Detected {k} clusters")
    
    create_interactive_similarity_graph(
        M_cmb, files, clusters, 
        threshold=threshold,
        out_html=str(OUTPUT_DIR / "similarity_graph.html"),
        top_k_edges=3,
        layout_method="spectral"
    )
    
    # Store in cache for quick pairwise lookups
    global doc_cache
    doc_cache = {
        "files": files,
        "names": names,
        "matrices": {
            "lexical": M_lex,
            "semantic": M_sem,
            "stylometric": M_sty,
            "combined": M_cmb,
        },
        "doc_sents": doc_sents,
        "doc_sem_emb": doc_sem_emb,
        "doc_style_feat": doc_style_feat,
    }
    
    # Summary
    flagged = [r for r in results if r["decision"] == "copied"]
    
    return {
        "success": True,
        "summary": {
            "total_documents": n,
            "total_pairs": len(results),
            "flagged_copied": len(flagged),
            "threshold": threshold,
            "clusters": k,
        },
        "results": results,
        "visualizations": {
            "heatmap_lexical": "/outputs/heatmap_lexical.html",
            "heatmap_semantic": "/outputs/heatmap_semantic.html" if np.isfinite(M_sem).any() else None,
            "heatmap_stylometric": "/outputs/heatmap_stylometric.html",
            "heatmap_combined": "/outputs/heatmap_combined.html",
            "similarity_graph": "/outputs/similarity_graph.html",
        }
    }


@app.post("/compare-pair")
async def compare_pair(request: ComparePairRequest):
    """
    Get detailed similarity scores for a specific pair of documents.
    """
    file_a = request.file_a
    file_b = request.file_b
    
    if not doc_cache:
        raise HTTPException(status_code=400, detail="Please run similarity detection first")
    
    files = doc_cache["files"]
    names = doc_cache["names"]
    
    # Find indices
    try:
        idx_a = names.index(file_a)
        idx_b = names.index(file_b)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"One or both files not found. Available: {names}")
    
    matrices = doc_cache["matrices"]
    
    lex_score = float(matrices["lexical"][idx_a, idx_b])
    sem_score = float(matrices["semantic"][idx_a, idx_b])
    sty_score = float(matrices["stylometric"][idx_a, idx_b])
    cmb_score = float(matrices["combined"][idx_a, idx_b])
    
    # Convert NaN to None
    lex_score = None if np.isnan(lex_score) else round(lex_score, 4)
    sem_score = None if np.isnan(sem_score) else round(sem_score, 4)
    sty_score = None if np.isnan(sty_score) else round(sty_score, 4)
    cmb_score = None if np.isnan(cmb_score) else round(cmb_score, 4)
    
    # Determine verdict
    if cmb_score is None or cmb_score < 0.6:
        verdict = "distinct"
        confidence = "high"
    elif cmb_score < 0.7:
        verdict = "suspect"
        confidence = "medium"
    else:
        verdict = "copied"
        confidence = "high" if cmb_score >= 0.85 else "medium"
    
    return {
        "file_a": file_a,
        "file_b": file_b,
        "scores": {
            "lexical": lex_score,
            "semantic": sem_score,
            "stylometric": sty_score,
            "combined": cmb_score,
        },
        "verdict": verdict,
        "confidence": confidence,
        "details": {
            "sentences_a": len(doc_cache["doc_sents"][files[idx_a]]),
            "sentences_b": len(doc_cache["doc_sents"][files[idx_b]]),
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)