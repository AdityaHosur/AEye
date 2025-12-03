import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import {
  uploadFiles,
  uploadImages,
  getOcrStatus,
  getFiles,
  clearFiles,
  detectSimilarity,
  comparePair,
  comparePairDetailed,
  getVisualizationUrl,
} from './api';

function App() {
  const [files, setFiles] = useState([]);
  const [imageFiles, setImageFiles] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [selectedFileA, setSelectedFileA] = useState('');
  const [selectedFileB, setSelectedFileB] = useState('');
  const [comparisonResult, setComparisonResult] = useState(null);
  const [threshold, setThreshold] = useState(0.70);
  const [activeViz, setActiveViz] = useState('similarity_graph');
  const [ocrAvailable, setOcrAvailable] = useState(false);
  const [uploadMode, setUploadMode] = useState('text'); // 'text' or 'image'
  
  const fileInputRef = useRef(null);
  const imageInputRef = useRef(null);

  // Check OCR availability on mount
  useEffect(() => {
    const checkOcr = async () => {
      try {
        const status = await getOcrStatus();
        setOcrAvailable(status.available);
      } catch (error) {
        console.error('Failed to check OCR status:', error);
      }
    };
    checkOcr();
  }, []);

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
  };

  const handleImageSelect = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setImageFiles(selectedFiles);
  };

  const handleUpload = async () => {
    if (files.length < 2) {
      alert('Please select at least 2 files');
      return;
    }

    setLoading(true);
    try {
      const result = await uploadFiles(files);
      alert(result.message);
      
      const fileList = await getFiles();
      setUploadedFiles(fileList.files);
      setFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      alert('Upload failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = async () => {
    if (imageFiles.length < 1) {
      alert('Please select at least 1 image');
      return;
    }

    setLoading(true);
    try {
      const result = await uploadImages(imageFiles, true);
      
      let message = `✅ Successfully processed ${result.success_count} image(s)`;
      if (result.error_count > 0) {
        message += `\n⚠️ Failed: ${result.error_count} image(s)`;
      }
      
      if (result.processed && result.processed.length > 0) {
        const preview = result.processed[0];
        message += `\n\n📝 Sample extracted text:\n"${preview.extracted_text_preview}"`;
        message += `\n\n📊 Confidence: ${preview.confidence}%`;
        message += `\n📄 Words: ${preview.statistics.words}`;
      }
      
      alert(message);
      
      const fileList = await getFiles();
      setUploadedFiles(fileList.files);
      setImageFiles([]);
      if (imageInputRef.current) {
        imageInputRef.current.value = '';
      }
    } catch (error) {
      alert('Image upload failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClearFiles = async () => {
    if (!window.confirm('Clear all uploaded files?')) return;
    
    setLoading(true);
    try {
      await clearFiles();
      setUploadedFiles([]);
      setDetectionResult(null);
      setComparisonResult(null);
      setSelectedFileA('');
      setSelectedFileB('');
      alert('All files cleared');
    } catch (error) {
      alert('Clear failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDetectSimilarity = async () => {
    if (uploadedFiles.length < 2) {
      alert('Please upload at least 2 files first');
      return;
    }

    setLoading(true);
    setDetectionResult(null);
    try {
      const result = await detectSimilarity(threshold);
      setDetectionResult(result);
      setActiveViz('similarity_graph');
    } catch (error) {
      alert('Detection failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleComparePair = async () => {
    if (!selectedFileA || !selectedFileB) {
      alert('Please select two files to compare');
      return;
    }

    if (selectedFileA === selectedFileB) {
      alert('Please select different files');
      return;
    }

    setLoading(true);
    setComparisonResult(null);
    try {
      const result = await comparePairDetailed(selectedFileA, selectedFileB);
      setComparisonResult(result);
    } catch (error) {
      alert('Comparison failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

   const highlightContainerStyle = {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '1rem',
    marginTop: '1rem'
  };

  const getVerdictColor = (verdict) => {
    switch (verdict) {
      case 'copied':
        return '#ef4444';
      case 'suspect':
        return '#f59e0b';
      case 'distinct':
        return '#10b981';
      default:
        return '#6b7280';
    }
  };

  const getScoreColor = (score) => {
    if (score === null || score === undefined) return '#6b7280';
    if (score >= 0.7) return '#ef4444';
    if (score >= 0.6) return '#f59e0b';
    return '#10b981';
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🔍 AEye - Document Similarity Detection</h1>
        <p>Advanced Copy detection with OCR support for handwritten documents</p>
      </header>

      <div className="container">
        {/* Upload Section with Tabs */}
        <section className="card">
          <h2>📤 Upload Documents</h2>
          
          {/* Upload Mode Toggle */}
          <div className="upload-mode-toggle">
            <button
              className={`mode-btn ${uploadMode === 'text' ? 'active' : ''}`}
              onClick={() => setUploadMode('text')}
            >
              📄 Text Files
            </button>
            <button
              className={`mode-btn ${uploadMode === 'image' ? 'active' : ''}`}
              onClick={() => setUploadMode('image')}
              disabled={!ocrAvailable}
              title={!ocrAvailable ? 'OCR not available' : ''}
            >
              🖼️ Handwritten Images {!ocrAvailable && '(Unavailable)'}
            </button>
          </div>

          {/* Text Upload */}
          {uploadMode === 'text' && (
            <div className="upload-area">
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".txt,.md,.text,.log"
                onChange={handleFileSelect}
                className="file-input"
              />
              <div className="upload-info">
                <p>{files.length > 0 ? `${files.length} file(s) selected` : 'No files selected'}</p>
                {files.length > 0 && (
                  <ul className="file-list">
                    {files.map((file, idx) => (
                      <li key={idx}>📄 {file.name} <span className="file-size">({(file.size / 1024).toFixed(2)} KB)</span></li>
                    ))}
                  </ul>
                )}
              </div>
              <button 
                onClick={handleUpload} 
                disabled={files.length < 2 || loading}
                className="btn btn-primary"
              >
                {loading ? '⏳ Uploading...' : '📤 Upload Text Files'}
              </button>
            </div>
          )}

          {/* Image Upload */}
          {uploadMode === 'image' && (
            <div className="upload-area">
              <input
                ref={imageInputRef}
                type="file"
                multiple
                accept=".jpg,.jpeg,.png,.bmp,.tiff,.gif"
                onChange={handleImageSelect}
                className="file-input"
              />
              <div className="upload-info">
                <p>{imageFiles.length > 0 ? `${imageFiles.length} image(s) selected` : 'No images selected'}</p>
                {imageFiles.length > 0 && (
                  <ul className="file-list">
                    {imageFiles.map((file, idx) => (
                      <li key={idx}>🖼️ {file.name} <span className="file-size">({(file.size / 1024).toFixed(2)} KB)</span></li>
                    ))}
                  </ul>
                )}
                <div className="ocr-info">
                  <p>ℹ️ Images will be processed with TrOCR to extract handwritten text</p>
                  <p>✓ Line detection enabled for better accuracy</p>
                  <p>✓ Supports: .jpg, .png, .bmp, .tiff, .gif</p>
                </div>
              </div>
              <button 
                onClick={handleImageUpload} 
                disabled={imageFiles.length < 1 || loading}
                className="btn btn-primary"
              >
                {loading ? '⏳ Processing OCR...' : '🤖 Extract Text from Images'}
              </button>
            </div>
          )}

          {/* Clear Files Button */}
          {uploadedFiles.length > 0 && (
            <div className="button-group" style={{marginTop: '1rem'}}>
              <button 
                onClick={handleClearFiles} 
                disabled={loading}
                className="btn btn-secondary"
              >
                🗑️ Clear All Files
              </button>
            </div>
          )}

          {/* Uploaded Files List */}
          {uploadedFiles.length > 0 && (
            <div className="uploaded-files">
              <h3>📁 Uploaded Files ({uploadedFiles.length})</h3>
              <div className="file-grid">
                {uploadedFiles.map((file, idx) => (
                  <div key={idx} className="file-item">
                    📄 {file.filename} <span className="file-size">({file.size_kb} KB)</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* Detection Section */}
        {uploadedFiles.length >= 2 && (
          <section className="card">
            <h2>🔬 Similarity Detection</h2>
            <div className="detection-controls">
              <div className="threshold-control">
                <label>
                  Threshold: <strong>{threshold.toFixed(2)}</strong>
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="0.95"
                  step="0.05"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="threshold-slider"
                />
              </div>
              <button
                onClick={handleDetectSimilarity}
                disabled={loading}
                className="btn btn-detect"
              >
                {loading ? '⏳ Analyzing...' : '🔍 Detect Similarity'}
              </button>
            </div>

            {detectionResult && (
              <div className="detection-summary">
                <h3>📊 Detection Summary</h3>
                <div className="summary-grid">
                  <div className="summary-item">
                    <span className="summary-label">Total Documents</span>
                    <span className="summary-value">{detectionResult.summary.total_documents}</span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Total Pairs</span>
                    <span className="summary-value">{detectionResult.summary.total_pairs}</span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Flagged as Copied</span>
                    <span className="summary-value copied">{detectionResult.summary.flagged_copied}</span>
                  </div>
                  <div className="summary-item">
                    <span className="summary-label">Clusters Detected</span>
                    <span className="summary-value">{detectionResult.summary.clusters}</span>
                  </div>
                </div>
              </div>
            )}
          </section>
        )}

        {/* Visualizations */}
        {detectionResult && (
          <section className="card visualizations">
            <h2>📈 Interactive Visualizations</h2>
            <div className="viz-tabs">
              <button
                className={`viz-tab ${activeViz === 'similarity_graph' ? 'active' : ''}`}
                onClick={() => setActiveViz('similarity_graph')}
              >
                🕸️ Similarity Network
              </button>
              <button
                className={`viz-tab ${activeViz === 'heatmap_combined' ? 'active' : ''}`}
                onClick={() => setActiveViz('heatmap_combined')}
              >
                🔥 Combined Heatmap
              </button>
              <button
                className={`viz-tab ${activeViz === 'heatmap_lexical' ? 'active' : ''}`}
                onClick={() => setActiveViz('heatmap_lexical')}
              >
                📝 Lexical Heatmap
              </button>
              {detectionResult.visualizations.heatmap_semantic && (
                <button
                  className={`viz-tab ${activeViz === 'heatmap_semantic' ? 'active' : ''}`}
                  onClick={() => setActiveViz('heatmap_semantic')}
                >
                  🧠 Semantic Heatmap
                </button>
              )}
              <button
                className={`viz-tab ${activeViz === 'heatmap_stylometric' ? 'active' : ''}`}
                onClick={() => setActiveViz('heatmap_stylometric')}
              >
                ✍️ Stylometric Heatmap
              </button>
            </div>

            <div className="viz-content">
              <iframe
                src={getVisualizationUrl(detectionResult.visualizations[activeViz])}
                title={activeViz}
                className="viz-iframe"
              />
            </div>
          </section>
        )}
        {detectionResult && detectionResult.clusters_detail && detectionResult.clusters_detail.length > 0 && (
          <section className="card">
            <h2>🧩 Detected Clusters</h2>
            <div className="clusters-list">
              {detectionResult.clusters_detail.map((cluster) => (
                <div key={cluster.id} className="cluster-item">
                  <h3>Cluster #{cluster.id}</h3>
                  <ul className="cluster-docs">
                    {cluster.documents.map((doc) => (
                      <li key={doc}>📄 {doc}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        )}
        {/* Pairwise Comparison */}
        {detectionResult && (
          <section className="card">
            <h2>⚖️ Pairwise Comparison</h2>
            <div className="comparison-controls">
              <div className="dropdown-group">
                <label>Document A:</label>
                <select
                  value={selectedFileA}
                  onChange={(e) => setSelectedFileA(e.target.value)}
                  className="file-select"
                >
                  <option value="">Select a document</option>
                  {uploadedFiles.map((file, idx) => (
                    <option key={idx} value={file.filename}>
                      {file.filename}
                    </option>
                  ))}
                </select>
              </div>

              <div className="comparison-vs">VS</div>

              <div className="dropdown-group">
                <label>Document B:</label>
                <select
                  value={selectedFileB}
                  onChange={(e) => setSelectedFileB(e.target.value)}
                  className="file-select"
                >
                  <option value="">Select a document</option>
                  {uploadedFiles.map((file, idx) => (
                    <option key={idx} value={file.filename}>
                      {file.filename}
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleComparePair}
                disabled={!selectedFileA || !selectedFileB || loading}
                className="btn btn-primary"
              >
                {loading ? '⏳ Comparing...' : '🔍 Compare'}
              </button>
            </div>

            {comparisonResult && (
              <div className="comparison-result">
                <div className="verdict-banner" style={{ backgroundColor: getVerdictColor(comparisonResult.verdict) }}>
                  <h3>
                    {comparisonResult.verdict === 'copied' && '⚠️ COPIED'}
                    {comparisonResult.verdict === 'suspect' && '🔍 SUSPECT'}
                    {comparisonResult.verdict === 'distinct' && '✅ DISTINCT'}
                  </h3>
                  <p>Confidence: <strong>{comparisonResult.confidence.toUpperCase()}</strong></p>
                </div>
                <div className="pair-description">
                  <p>{comparisonResult.description}</p>
                </div>
                <div className="scores-grid">
                  <div className="score-card">
                    <h4>📝 Lexical</h4>
                    <div 
                      className="score-value"
                      style={{ color: getScoreColor(comparisonResult.scores.lexical) }}
                    >
                      {comparisonResult.scores.lexical !== null 
                        ? comparisonResult.scores.lexical.toFixed(3)
                        : 'N/A'}
                    </div>
                    <div className="score-label">TF-IDF Similarity</div>
                  </div>

                  <div className="score-card">
                    <h4>🧠 Semantic</h4>
                    <div 
                      className="score-value"
                      style={{ color: getScoreColor(comparisonResult.scores.semantic) }}
                    >
                      {comparisonResult.scores.semantic !== null 
                        ? comparisonResult.scores.semantic.toFixed(3)
                        : 'N/A'}
                    </div>
                    <div className="score-label">Embedding Similarity</div>
                  </div>

                  <div className="score-card">
                    <h4>✍️ Stylometric</h4>
                    <div 
                      className="score-value"
                      style={{ color: getScoreColor(comparisonResult.scores.stylometric) }}
                    >
                      {comparisonResult.scores.stylometric !== null 
                        ? comparisonResult.scores.stylometric.toFixed(3)
                        : 'N/A'}
                    </div>
                    <div className="score-label">Writing Style</div>
                  </div>

                  <div className="score-card combined">
                    <h4>🎯 Combined</h4>
                    <div 
                      className="score-value large"
                      style={{ color: 'white' }}
                    >
                      {comparisonResult.scores.combined !== null 
                        ? comparisonResult.scores.combined.toFixed(3)
                        : 'N/A'}
                    </div>
                    <div className="score-label">Weighted Average</div>
                  </div>
                </div>

                <div className="comparison-details">
                  <p>📄 Document A: <strong>{comparisonResult.details.sentences_a}</strong> sentences</p>
                  <p>📄 Document B: <strong>{comparisonResult.details.sentences_b}</strong> sentences</p>
                </div>
                {comparisonResult.highlights && (
                  <>
                    <h3>🔦 Similarity Highlights</h3>
                    <p className="highlight-legend">
                      Sentences matched semantically (≥ 0.80) are highlighted in blue. Shared high-impact words are highlighted in yellow.
                    </p>
                    <div style={highlightContainerStyle}>
                      <div className="highlight-pane">
                        <h4>Document A: {comparisonResult.file_a}</h4>
                        <div
                          className="highlight-html"
                          dangerouslySetInnerHTML={{ __html: comparisonResult.highlights.html_a }}
                        />
                      </div>
                      <div className="highlight-pane">
                        <h4>Document B: {comparisonResult.file_b}</h4>
                        <div
                          className="highlight-html"
                          dangerouslySetInnerHTML={{ __html: comparisonResult.highlights.html_b }}
                        />
                      </div>
                    </div>
                    {comparisonResult.highlights.meta &&
                      comparisonResult.highlights.meta.high_sentence_pairs &&
                      comparisonResult.highlights.meta.high_sentence_pairs.length > 0 && (
                        <div className="matched-sentences">
                          <h4>Matched Sentence Pairs (Top)</h4>
                          <ul>
                            {comparisonResult.highlights.meta.high_sentence_pairs.slice(0, 10).map((p, idx) => (
                              <li key={idx}>
                                <strong>A[{p.a_index}]</strong>: {p.a_text}
                                <br />
                                <strong>B[{p.b_index}]</strong>: {p.b_text}
                                <br />
                                <span className="pair-score">Sim: {p.score.toFixed(3)}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )
                    }
                  </>
                )}
              </div>
            )}
              
          </section>
        )}
      </div>
    </div>
  );
}

export default App;