import React, { useState, useRef } from 'react';
import './App.css';
import {
  uploadFiles,
  getFiles,
  clearFiles,
  detectSimilarity,
  comparePair,
  getVisualizationUrl,
} from './api';

function App() {
  const [files, setFiles] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [selectedFileA, setSelectedFileA] = useState('');
  const [selectedFileB, setSelectedFileB] = useState('');
  const [comparisonResult, setComparisonResult] = useState(null);
  const [threshold, setThreshold] = useState(0.70);
  const [activeViz, setActiveViz] = useState('similarity_graph');
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
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
      
      // Fetch updated file list
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
      const result = await comparePair(selectedFileA, selectedFileB);
      setComparisonResult(result);
    } catch (error) {
      alert('Comparison failed: ' + error.message);
    } finally {
      setLoading(false);
    }
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
        <p>Advanced plagiarism detection using lexical, semantic, and stylometric analysis</p>
      </header>

      <div className="container">
        {/* Upload Section */}
        <section className="card">
          <h2>📤 Upload Documents</h2>
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
                    <li key={idx}>{file.name} ({(file.size / 1024).toFixed(2)} KB)</li>
                  ))}
                </ul>
              )}
            </div>
            <div className="button-group">
              <button 
                onClick={handleUpload} 
                disabled={files.length < 2 || loading}
                className="btn btn-primary"
              >
                {loading ? '⏳ Uploading...' : '📤 Upload Files'}
              </button>
              {uploadedFiles.length > 0 && (
                <button 
                  onClick={handleClearFiles} 
                  disabled={loading}
                  className="btn btn-secondary"
                >
                  🗑️ Clear All
                </button>
              )}
            </div>
          </div>

          {uploadedFiles.length > 0 && (
            <div className="uploaded-files">
              <h3>📁 Uploaded Files ({uploadedFiles.length})</h3>
              <div className="file-grid">
                {uploadedFiles.map((file, idx) => (
                  <div key={idx} className="file-item">
                    📄 {file.filename}
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
                      style={{ color: getScoreColor(comparisonResult.scores.combined) }}
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
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
}

export default App;