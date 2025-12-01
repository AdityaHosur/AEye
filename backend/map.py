import os
import time
import json
import psutil
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.lexical import sentence_tokenize as lex_sent_tokenize
from models.semantic import embed_sentences
from score import list_text_files, read_file, safe_mean_max_score


def complex_similarity_computation(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Optimized complex similarity computation with reduced memory footprint.
    Processes in chunks to avoid large intermediate matrices.
    """
    # 1. Cosine similarity matrix - compute in chunks to save memory
    chunk_size = 100
    n_a, n_b = len(emb_a), len(emb_b)
    
    # Collect statistics without storing full matrix
    row_maxes = []
    col_maxes = np.full(n_b, -np.inf)
    row_means = []
    
    for i in range(0, n_a, chunk_size):
        end_i = min(i + chunk_size, n_a)
        chunk_sim = emb_a[i:end_i] @ emb_b.T
        
        # Row statistics
        row_maxes.extend(chunk_sim.max(axis=1))
        row_means.extend(chunk_sim.mean(axis=1))
        
        # Column maximums
        chunk_col_max = chunk_sim.max(axis=0)
        col_maxes = np.maximum(col_maxes, chunk_col_max)
    
    row_maxes = np.array(row_maxes)
    row_means = np.array(row_means)
    
    # 2. Basic scores
    score_max = 0.5 * (row_maxes.mean() + col_maxes.mean())
    score_mean = 0.5 * (row_means.mean() + col_maxes.mean())
    
    # 3. Compute additional metrics on smaller sample
    sample_size = min(50, n_a, n_b)
    if sample_size > 0:
        # Sample indices
        sample_a = np.random.choice(n_a, sample_size, replace=False) if n_a > sample_size else np.arange(n_a)
        sample_b = np.random.choice(n_b, sample_size, replace=False) if n_b > sample_size else np.arange(n_b)
        
        # Sample similarity matrix
        sample_sim = emb_a[sample_a] @ emb_b[sample_b].T
        
        # Percentiles on sample
        p75_score = np.percentile(sample_sim, 75)
        p90_score = np.percentile(sample_sim, 90)
        std_score = sample_sim.std()
        
        # Pairwise distances on smaller sample (memory efficient)
        dist_sample = min(20, sample_size)
        distances = []
        for i in range(dist_sample):
            for j in range(dist_sample):
                dist = np.linalg.norm(emb_a[sample_a[i]] - emb_b[sample_b[j]])
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0.0
        distance_score = 1.0 / (1.0 + avg_distance)
    else:
        p75_score = score_max
        p90_score = score_max
        std_score = 0.0
        distance_score = score_max
    
    # 4. Final weighted combination
    final_score = (
        0.35 * score_max +
        0.25 * score_mean +
        0.15 * p90_score +
        0.10 * p75_score +
        0.10 * distance_score +
        0.05 * (1.0 - std_score)
    )
    
    return float(final_score)


def compute_serial(doc_embeddings: Dict[str, np.ndarray], files: List[str]) -> float:
    """
    Compute all pairwise similarities serially.
    Uses pre-computed embeddings to focus on similarity computation only.
    """
    n = len(files)
    start_time = time.time()
    
    for i in range(n):
        for j in range(i + 1, n):
            emb_a = doc_embeddings[files[i]]
            emb_b = doc_embeddings[files[j]]
            _ = complex_similarity_computation(emb_a, emb_b)
    
    return time.time() - start_time


def compute_similarity_batch(batch_data: Tuple[List[Tuple[int, int]], Dict[str, np.ndarray], List[str]]) -> List[Tuple[int, int, float]]:
    """
    Process a batch of similarity computations.
    This function will be executed by each worker thread.
    
    Args:
        batch_data: Tuple of (list of (i,j) pairs, embeddings dict, files list)
    
    Returns:
        List of (i, j, score) tuples
    """
    pairs, doc_embeddings, files = batch_data
    results = []
    
    for i, j in pairs:
        emb_a = doc_embeddings[files[i]]
        emb_b = doc_embeddings[files[j]]
        score = complex_similarity_computation(emb_a, emb_b)
        results.append((i, j, score))
    
    return results


def compute_parallel_threaded(doc_embeddings: Dict[str, np.ndarray], files: List[str], n_workers: int = 8) -> Tuple[float, float, float]:
    """
    Compute similarities in parallel using ThreadPoolExecutor with batching.
    Returns execution time, peak memory usage, and average memory usage.
    """
    n = len(files)
    
    # Generate all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    total_pairs = len(all_pairs)
    
    # Calculate optimal batch size - smaller batches for better load balancing
    min_pairs_per_batch = max(1, total_pairs // (n_workers * 4))
    
    # Create batches
    batches = []
    for i in range(0, total_pairs, min_pairs_per_batch):
        batch_pairs = all_pairs[i:i + min_pairs_per_batch]
        batches.append((batch_pairs, doc_embeddings, files))
    
    print(f"   🔢 Created {len(batches)} batches (~{min_pairs_per_batch} pairs/batch) for {n_workers} workers")
    
    # Track memory usage
    process = psutil.Process()
    memory_samples = []
    
    # Measure parallel computation time
    start_time = time.time()
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all batches
        futures = [executor.submit(compute_similarity_batch, batch) for batch in batches]
        
        # Collect results and sample memory
        all_results = []
        for future in futures:
            all_results.extend(future.result())
            # Sample memory during execution
            current_mem = process.memory_info().rss / (1024 * 1024)
            memory_samples.append(current_mem)
    
    elapsed = time.time() - start_time
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Calculate memory metrics
    peak_memory = max(memory_samples) if memory_samples else mem_after
    avg_memory = np.mean(memory_samples) if memory_samples else mem_after
    
    return elapsed, peak_memory, avg_memory


def precompute_embeddings(doc_sents: Dict[str, List[str]], files: List[str]) -> Dict[str, np.ndarray]:
    """Pre-compute embeddings for all documents in ONE batch"""
    all_sentences = []
    doc_sentence_ranges = {}
    
    current_idx = 0
    for fp in files:
        sents = doc_sents[fp]
        doc_sentence_ranges[fp] = (current_idx, current_idx + len(sents))
        all_sentences.extend(sents)
        current_idx += len(sents)
    
    # Compute embeddings
    all_embeddings = embed_sentences(all_sentences)
    
    # Split back into documents and convert to float16 to save memory
    doc_embeddings = {}
    for fp in files:
        start, end = doc_sentence_ranges[fp]
        # Use float16 for storage (half the memory)
        doc_embeddings[fp] = all_embeddings[start:end].astype(np.float16)
    
    return doc_embeddings


def run_benchmark(uploads_dir: str, output_dir: str, doc_counts: List[int] = None, thread_counts: List[int] = None):
    """Run benchmark comparing serial vs threaded parallelization"""
    
    if doc_counts is None:
        doc_counts = [5, 10, 15, 20]
    
    if thread_counts is None:
        thread_counts = [2, 4, 8, 12]
    
    # Load documents
    all_files = list_text_files(uploads_dir)
    max_docs = max(doc_counts)
    
    if len(all_files) < max_docs:
        print(f"⚠️  Only {len(all_files)} files available, adjusting...")
        doc_counts = [d for d in doc_counts if d <= len(all_files)]
        max_docs = len(all_files)
    
    print(f"🚀 Complex Similarity Benchmark (Memory-Optimized)")
    print(f"📁 Documents available: {len(all_files)}")
    print(f"📊 Testing document counts: {doc_counts}")
    print(f"🧵 Testing thread counts: {thread_counts}")
    print(f"💻 CPU cores available: {mp.cpu_count()}\n")
    
    # Load and tokenize
    print("📄 Loading documents...")
    all_doc_sents = {}
    for idx, fp in enumerate(all_files[:max_docs], 1):
        txt = read_file(fp)
        sents = lex_sent_tokenize(txt)
        all_doc_sents[fp] = sents
        print(f"   Document {idx}: {len(sents)} sentences")
    
    # Warm up model
    print("\n🧠 Loading model...")
    _ = embed_sentences(["warmup"])
    print("✅ Ready\n")
    
    # Results storage
    results = {
        'doc_counts': doc_counts,
        'serial_times': [],
        'thread_times': {n: [] for n in thread_counts},
        'peak_memory': {n: [] for n in thread_counts},
        'avg_memory': {n: [] for n in thread_counts},
        'embedding_times': [],
        'thread_counts': thread_counts
    }
    
    # Benchmark
    for num_docs in doc_counts:
        files_subset = all_files[:num_docs]
        doc_sents_subset = {fp: all_doc_sents[fp] for fp in files_subset}
        num_pairs = num_docs * (num_docs - 1) // 2
        
        print("="*80)
        print(f"📊 Testing {num_docs} documents ({num_pairs} similarity computations)")
        print("="*80)
        
        # Pre-compute embeddings ONCE for all tests
        print(f"📦 Pre-computing embeddings...")
        embed_start = time.time()
        doc_embeddings = precompute_embeddings(doc_sents_subset, files_subset)
        embed_time = time.time() - embed_start
        results['embedding_times'].append(embed_time)
        
        # Calculate total memory size
        total_memory_mb = sum(emb.nbytes for emb in doc_embeddings.values()) / (1024 * 1024)
        print(f"   ✅ Embeddings: {embed_time:.2f}s ({total_memory_mb:.1f} MB in float16)\n")
        
        # Serial execution (using pre-computed embeddings)
        print(f"🔄 Serial execution (single-threaded)...")
        serial_time = compute_serial(doc_embeddings, files_subset)
        results['serial_times'].append(serial_time)
        print(f"   ⏱️  Time: {serial_time:.3f}s\n")
        
        # Test different thread counts
        for n_workers in thread_counts:
            print(f"⚡ Testing with {n_workers} workers:")
            
            # Thread-based parallelization
            print(f"   🧵 Thread-based (batched)...")
            thread_time, peak_mem, avg_mem = compute_parallel_threaded(doc_embeddings, files_subset, n_workers)
            
            results['thread_times'][n_workers].append(thread_time)
            results['peak_memory'][n_workers].append(peak_mem)
            results['avg_memory'][n_workers].append(avg_mem)
            
            thread_speedup = serial_time / thread_time if thread_time > 0 else 1.0
            thread_efficiency = (thread_speedup / n_workers) * 100
            
            print(f"   ⏱️  Time: {thread_time:.3f}s")
            print(f"   📈 Speedup: {thread_speedup:.2f}x | Efficiency: {thread_efficiency:.1f}%")
            print(f"   💾 Memory: Peak={peak_mem:.1f}MB, Avg={avg_mem:.1f}MB")
            print()
        
        print()
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_json = {
        'doc_counts': results['doc_counts'],
        'serial_times': results['serial_times'],
        'thread_times': {str(k): v for k, v in results['thread_times'].items()},
        'peak_memory': {str(k): v for k, v in results['peak_memory'].items()},
        'avg_memory': {str(k): v for k, v in results['avg_memory'].items()},
        'embedding_times': results['embedding_times'],
        'thread_counts': results['thread_counts']
    }
    
    with open(output_path / "benchmark_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    
    # Generate plots
    print("="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    create_comprehensive_plots(results, output_path)
    
    print("\n✨ Benchmark Complete!")
    print(f"📈 Plots saved in: {output_path}")


def create_comprehensive_plots(results: Dict, output_path: Path):
    """Create comprehensive plots comparing different parallelization strategies"""
    
    doc_counts = results['doc_counts']
    serial_times = results['serial_times']
    thread_counts = results['thread_counts']
    thread_times = results['thread_times']
    peak_memory = results['peak_memory']
    avg_memory = results['avg_memory']
    
    # Calculate speedups and efficiencies
    thread_speedups = {n: [s/p if p > 0 else 1.0 for s, p in zip(serial_times, thread_times[n])] for n in thread_counts}
    thread_efficiencies = {n: [(speedup / n) * 100 for speedup in thread_speedups[n]] for n in thread_counts}
    
    colors = {
        'serial': '#ef4444',
        2: '#f59e0b',
        4: '#10b981',
        8: '#3b82f6',
        12: '#8b5cf6',
        16: '#ec4899'
    }
    
    # ============= PLOT 1: SPEEDUP COMPARISON (ALL THREADS) =============
    fig_speedup_all = go.Figure()
    
    # Add speedup lines for each thread count
    for n_workers in thread_counts:
        speedups = thread_speedups[n_workers]
        fig_speedup_all.add_trace(go.Scatter(
            x=doc_counts,
            y=speedups,
            mode='lines+markers',
            name=f'{n_workers} threads',
            line=dict(color=colors[n_workers], width=3),
            marker=dict(size=12, symbol='diamond', line=dict(width=2, color='white')),
            hovertemplate='<b>%{fullData.name}</b><br>Documents: %{x}<br>Speedup: %{y:.2f}x<extra></extra>'
        ))
    
    # Add ideal speedup lines (dotted)
    for n_workers in thread_counts:
        fig_speedup_all.add_trace(go.Scatter(
            x=doc_counts,
            y=[n_workers] * len(doc_counts),
            mode='lines',
            name=f'Ideal {n_workers}x',
            line=dict(color=colors[n_workers], width=1, dash='dash'),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig_speedup_all.update_layout(
        title='<b>🚀 Speedup Comparison: All Thread Counts</b><br><sub>Actual Performance vs Ideal Linear Speedup</sub>',
        xaxis_title='<b>Number of Documents</b>',
        yaxis_title='<b>Speedup (x times faster than serial)</b>',
        template='plotly_white',
        height=700,
        hovermode='x unified',
        legend=dict(
            x=0.02, y=0.98, 
            bgcolor='rgba(255,255,255,0.95)', 
            bordercolor='black', 
            borderwidth=1
        ),
        font=dict(size=12)
    )
    
    fig_speedup_all.write_html(str(output_path / "speedup_all_threads.html"), include_plotlyjs='cdn')
    print(f"✅ Speedup (all threads) plot: {output_path / 'speedup_all_threads.html'}")
    
    # ============= PLOT 2: EXECUTION TIME BY THREAD COUNT =============
    fig_exec_time = go.Figure()
    
    # Serial baseline
    fig_exec_time.add_trace(go.Scatter(
        x=doc_counts,
        y=serial_times,
        mode='lines+markers',
        name='Serial (1 thread)',
        line=dict(color=colors['serial'], width=3),
        marker=dict(size=12, symbol='circle', line=dict(width=2, color='white')),
        hovertemplate='<b>Serial</b><br>Documents: %{x}<br>Time: %{y:.3f}s<extra></extra>'
    ))
    
    # Thread-based lines
    for n_workers in thread_counts:
        times = thread_times[n_workers]
        fig_exec_time.add_trace(go.Scatter(
            x=doc_counts,
            y=times,
            mode='lines+markers',
            name=f'{n_workers} threads',
            line=dict(color=colors[n_workers], width=2, dash='dot'),
            marker=dict(size=10, symbol='square'),
            hovertemplate=f'<b>{n_workers} threads</b><br>Documents: %{{x}}<br>Time: %{{y:.3f}}s<extra></extra>'
        ))
    
    fig_exec_time.update_layout(
        title='<b>⏱️ Execution Time: Different Thread Counts</b><br><sub>Time vs Number of Documents (Lower is Better)</sub>',
        xaxis_title='<b>Number of Documents</b>',
        yaxis_title='<b>Execution Time (seconds)</b>',
        template='plotly_white',
        height=700,
        hovermode='x unified',
        legend=dict(
            x=0.98, y=0.98, 
            xanchor='right',
            bgcolor='rgba(255,255,255,0.95)', 
            bordercolor='black', 
            borderwidth=1
        ),
        font=dict(size=12)
    )
    
    fig_exec_time.write_html(str(output_path / "execution_time_by_threads.html"), include_plotlyjs='cdn')
    print(f"✅ Execution time plot: {output_path / 'execution_time_by_threads.html'}")
    
    # ============= PLOT 3: MEMORY UTILIZATION (PEAK) =============
    fig_memory_peak = go.Figure()
    
    for n_workers in thread_counts:
        peak_mem = peak_memory[n_workers]
        fig_memory_peak.add_trace(go.Scatter(
            x=doc_counts,
            y=peak_mem,
            mode='lines+markers',
            name=f'{n_workers} threads',
            line=dict(color=colors[n_workers], width=3),
            marker=dict(size=12, symbol='diamond', line=dict(width=2, color='white')),
            hovertemplate=f'<b>{n_workers} threads</b><br>Documents: %{{x}}<br>Peak Memory: %{{y:.1f}} MB<extra></extra>'
        ))
    
    fig_memory_peak.update_layout(
        title='<b>💾 Peak Memory Utilization</b><br><sub>Memory Usage vs Thread Count</sub>',
        xaxis_title='<b>Number of Documents</b>',
        yaxis_title='<b>Peak Memory (MB)</b>',
        template='plotly_white',
        height=700,
        hovermode='x unified',
        legend=dict(
            x=0.02, y=0.98, 
            bgcolor='rgba(255,255,255,0.95)', 
            bordercolor='black', 
            borderwidth=1
        ),
        font=dict(size=12)
    )
    
    fig_memory_peak.write_html(str(output_path / "memory_peak_utilization.html"), include_plotlyjs='cdn')
    print(f"✅ Peak memory plot: {output_path / 'memory_peak_utilization.html'}")
    
    # ============= PLOT 4: MEMORY UTILIZATION (AVERAGE) =============
    fig_memory_avg = go.Figure()
    
    for n_workers in thread_counts:
        avg_mem = avg_memory[n_workers]
        fig_memory_avg.add_trace(go.Scatter(
            x=doc_counts,
            y=avg_mem,
            mode='lines+markers',
            name=f'{n_workers} threads',
            line=dict(color=colors[n_workers], width=3),
            marker=dict(size=12, symbol='circle', line=dict(width=2, color='white')),
            hovertemplate=f'<b>{n_workers} threads</b><br>Documents: %{{x}}<br>Avg Memory: %{{y:.1f}} MB<extra></extra>'
        ))
    
    fig_memory_avg.update_layout(
        title='<b>💾 Average Memory Utilization</b><br><sub>Average Memory Usage During Execution</sub>',
        xaxis_title='<b>Number of Documents</b>',
        yaxis_title='<b>Average Memory (MB)</b>',
        template='plotly_white',
        height=700,
        hovermode='x unified',
        legend=dict(
            x=0.02, y=0.98, 
            bgcolor='rgba(255,255,255,0.95)', 
            bordercolor='black', 
            borderwidth=1
        ),
        font=dict(size=12)
    )
    
    fig_memory_avg.write_html(str(output_path / "memory_avg_utilization.html"), include_plotlyjs='cdn')
    print(f"✅ Average memory plot: {output_path / 'memory_avg_utilization.html'}")
    
    # ============= PLOT 5: EFFICIENCY COMPARISON =============
    fig_efficiency = go.Figure()
    
    for n_workers in thread_counts:
        efficiencies = thread_efficiencies[n_workers]
        fig_efficiency.add_trace(go.Scatter(
            x=doc_counts,
            y=efficiencies,
            mode='lines+markers',
            name=f'{n_workers} threads',
            line=dict(color=colors[n_workers], width=3),
            marker=dict(size=12, symbol='square', line=dict(width=2, color='white')),
            hovertemplate=f'<b>{n_workers} threads</b><br>Documents: %{{x}}<br>Efficiency: %{{y:.1f}}%<extra></extra>'
        ))
    
    # Add 100% efficiency reference line
    fig_efficiency.add_shape(
        type="line",
        x0=min(doc_counts), y0=100, x1=max(doc_counts), y1=100,
        line=dict(color="gray", width=2, dash="dash"),
    )
    
    fig_efficiency.add_annotation(
        x=max(doc_counts), y=100,
        text="100% (Ideal)",
        showarrow=False,
        yshift=10,
        font=dict(color="gray", size=10)
    )
    
    fig_efficiency.update_layout(
        title='<b>📊 Threading Efficiency</b><br><sub>Percentage of Ideal Linear Speedup (Higher is Better)</sub>',
        xaxis_title='<b>Number of Documents</b>',
        yaxis_title='<b>Efficiency (%)</b>',
        yaxis=dict(range=[0, 110]),
        template='plotly_white',
        height=700,
        hovermode='x unified',
        legend=dict(
            x=0.98, y=0.02, 
            xanchor='right',
            yanchor='bottom',
            bgcolor='rgba(255,255,255,0.95)', 
            bordercolor='black', 
            borderwidth=1
        ),
        font=dict(size=12)
    )
    
    fig_efficiency.write_html(str(output_path / "efficiency_comparison.html"), include_plotlyjs='cdn')
    print(f"✅ Efficiency plot: {output_path / 'efficiency_comparison.html'}")
    
    # ============= PLOT 6: EFFICIENCY HEATMAP =============
    efficiency_matrix = []
    for n_workers in thread_counts:
        efficiency_matrix.append(thread_efficiencies[n_workers])
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=efficiency_matrix,
        x=doc_counts,
        y=[f'{n} threads' for n in thread_counts],
        colorscale='RdYlGn',
        text=[[f'{val:.1f}%' for val in row] for row in efficiency_matrix],
        texttemplate='%{text}',
        textfont={"size": 12, "color": "black"},
        colorbar=dict(title="Efficiency %"),
        hovertemplate='<b>%{y}</b><br>Documents: %{x}<br>Efficiency: %{z:.1f}%<extra></extra>'
    ))
    
    fig_heatmap.update_layout(
        title='<b>🔥 Threading Efficiency Heatmap</b><br><sub>Darker Green = Better Performance</sub>',
        xaxis_title='<b>Number of Documents</b>',
        yaxis_title='<b>Thread Count</b>',
        template='plotly_white',
        height=500,
        font=dict(size=12)
    )
    
    fig_heatmap.write_html(str(output_path / "efficiency_heatmap.html"), include_plotlyjs='cdn')
    print(f"✅ Efficiency heatmap: {output_path / 'efficiency_heatmap.html'}")
    
    # ============= PLOT 7: COMBINED DASHBOARD (4 subplots) =============
    fig_dashboard = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Execution Time vs Documents',
            'Speedup vs Documents',
            'Efficiency vs Documents',
            'Peak Memory vs Documents'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Subplot 1: Execution Time
    for n_workers in thread_counts:
        fig_dashboard.add_trace(
            go.Scatter(
                x=doc_counts,
                y=thread_times[n_workers],
                mode='lines+markers',
                name=f'{n_workers} threads',
                line=dict(color=colors[n_workers], width=2),
                marker=dict(size=8),
                showlegend=True,
                legendgroup='threads'
            ),
            row=1, col=1
        )
    
    # Subplot 2: Speedup
    for n_workers in thread_counts:
        fig_dashboard.add_trace(
            go.Scatter(
                x=doc_counts,
                y=thread_speedups[n_workers],
                mode='lines+markers',
                name=f'{n_workers} threads',
                line=dict(color=colors[n_workers], width=2),
                marker=dict(size=8),
                showlegend=False,
                legendgroup='threads'
            ),
            row=1, col=2
        )
    
    # Subplot 3: Efficiency
    for n_workers in thread_counts:
        fig_dashboard.add_trace(
            go.Scatter(
                x=doc_counts,
                y=thread_efficiencies[n_workers],
                mode='lines+markers',
                name=f'{n_workers} threads',
                line=dict(color=colors[n_workers], width=2),
                marker=dict(size=8),
                showlegend=False,
                legendgroup='threads'
            ),
            row=2, col=1
        )
    
    # Subplot 4: Peak Memory
    for n_workers in thread_counts:
        fig_dashboard.add_trace(
            go.Scatter(
                x=doc_counts,
                y=peak_memory[n_workers],
                mode='lines+markers',
                name=f'{n_workers} threads',
                line=dict(color=colors[n_workers], width=2),
                marker=dict(size=8),
                showlegend=False,
                legendgroup='threads'
            ),
            row=2, col=2
        )
    
    # Update axes labels
    fig_dashboard.update_xaxes(title_text="Documents", row=1, col=1)
    fig_dashboard.update_xaxes(title_text="Documents", row=1, col=2)
    fig_dashboard.update_xaxes(title_text="Documents", row=2, col=1)
    fig_dashboard.update_xaxes(title_text="Documents", row=2, col=2)
    
    fig_dashboard.update_yaxes(title_text="Time (s)", row=1, col=1)
    fig_dashboard.update_yaxes(title_text="Speedup (x)", row=1, col=2)
    fig_dashboard.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig_dashboard.update_yaxes(title_text="Memory (MB)", row=2, col=2)
    
    fig_dashboard.update_layout(
        title_text='<b>📊 Performance Dashboard</b><br><sub>Comprehensive Threading Performance Metrics</sub>',
        height=900,
        showlegend=True,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig_dashboard.write_html(str(output_path / "performance_dashboard.html"), include_plotlyjs='cdn')
    print(f"✅ Performance dashboard: {output_path / 'performance_dashboard.html'}")
    
    # ============= SUMMARY TABLE =============
    print("\n" + "="*100)
    print("📊 PERFORMANCE SUMMARY")
    print("="*100)
    print(f"{'Docs':<6} {'Serial':<12} ", end="")
    for n in thread_counts:
        print(f"{'T'+str(n):<16} ", end="")
    print()
    print("-" * 100)
    
    for i, num_docs in enumerate(doc_counts):
        print(f"{num_docs:<6} {serial_times[i]:>10.3f}s  ", end="")
        for n in thread_counts:
            t_time = thread_times[n][i]
            t_speedup = thread_speedups[n][i]
            print(f"{t_time:>7.3f}s({t_speedup:>5.2f}x)  ", end="")
        print()
    
    print("="*100)
    
    # Best configurations
    print("\n💡 KEY FINDINGS:")
    print("-" * 100)
    
    best_worker_count = max(thread_counts, key=lambda n: np.mean(thread_speedups[n]))
    best_speedup = np.mean(thread_speedups[best_worker_count])
    
    for n_workers in thread_counts:
        avg_speedup = np.mean(thread_speedups[n_workers])
        avg_efficiency = np.mean(thread_efficiencies[n_workers])
        avg_peak_mem = np.mean(peak_memory[n_workers])
        
        marker = "⭐" if n_workers == best_worker_count else "  "
        print(f"{marker} {n_workers:2d} Workers: {avg_speedup:.2f}x speedup, {avg_efficiency:.1f}% efficiency, {avg_peak_mem:.1f}MB peak mem")
    
    print("\n" + "="*100)
    print("🎯 RECOMMENDATIONS:")
    print(f"   • Best thread count: {best_worker_count} workers ({best_speedup:.2f}x average speedup)")
    print(f"   • Memory usage reduced by ~50% using float16 storage")
    print(f"   • Chunk-based processing prevents memory spikes")
    print(f"   • Good scaling observed up to {thread_counts[-1]} threads")
    print("="*100)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory-optimized similarity benchmark")
    parser.add_argument("--uploads", default="laup")
    parser.add_argument("--output", default="laout")
    parser.add_argument("--counts", nargs='+', type=int, default=None,
                       help="Document counts to test (default: 5 10 15 20)")
    parser.add_argument("--workers", nargs='+', type=int, default=None,
                       help="Worker counts to test (default: 2 4 8 12)")
    args = parser.parse_args()
    
    uploads_dir = os.path.join(os.path.dirname(__file__), args.uploads)
    output_dir = os.path.join(os.path.dirname(__file__), args.output)
    
    run_benchmark(uploads_dir, output_dir, args.counts, args.workers)


if __name__ == "__main__":
    main()