const API_BASE_URL = 'http://localhost:8000';

export const uploadFiles = async (files) => {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));
  const r = await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: formData });
  if (!r.ok) throw new Error((await r.json()).detail || 'Upload failed');
  return r.json();
};

export const uploadImages = async (files, splitLines = true) => {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));
  formData.append('split_lines', String(splitLines));
  const r = await fetch(`${API_BASE_URL}/upload-images`, { method: 'POST', body: formData });
  if (!r.ok) throw new Error((await r.json()).detail || 'Image upload failed');
  return r.json();
};

export const getOcrStatus = async () => {
  const r = await fetch(`${API_BASE_URL}/ocr-status`);
  if (!r.ok) throw new Error('Failed OCR status');
  return r.json();
};

export const getFiles = async () => {
  const r = await fetch(`${API_BASE_URL}/files`);
  if (!r.ok) throw new Error('File list failed');
  return r.json();
};

export const clearFiles = async () => {
  const r = await fetch(`${API_BASE_URL}/files`, { method: 'DELETE' });
  if (!r.ok) throw new Error('Clear failed');
  return r.json();
};

export const detectSimilarity = async (threshold = 0.70) => {
  const r = await fetch(`${API_BASE_URL}/detect-similarity?threshold=${threshold}`, { method: 'POST' });
  if (!r.ok) throw new Error((await r.json()).detail || 'Detection failed');
  return r.json();
};

export async function comparePair(fileA, fileB) {
  const r = await fetch(`${API_BASE_URL}/compare-pair`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_a: fileA, file_b: fileB })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function comparePairDetailed(fileA, fileB) {
  const r = await fetch(`${API_BASE_URL}/compare-pair-detailed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_a: fileA, file_b: fileB })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function getVisualizationUrl(path) {
  return `${API_BASE_URL}${path}`;
}