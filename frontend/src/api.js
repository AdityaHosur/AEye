const API_BASE_URL = 'http://localhost:8000';

export const uploadFiles = async (files) => {
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Upload failed');
  }

  return response.json();
};

export const getFiles = async () => {
  const response = await fetch(`${API_BASE_URL}/files`);
  if (!response.ok) {
    throw new Error('Failed to fetch files');
  }
  return response.json();
};

export const clearFiles = async () => {
  const response = await fetch(`${API_BASE_URL}/files`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error('Failed to clear files');
  }
  return response.json();
};

export const detectSimilarity = async (threshold = 0.70) => {
  const response = await fetch(`${API_BASE_URL}/detect-similarity?threshold=${threshold}`, {
    method: 'POST',
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Detection failed');
  }

  return response.json();
};

export const comparePair = async (fileA, fileB) => {
  const response = await fetch(`${API_BASE_URL}/compare-pair`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_a: String(fileA),
      file_b: String(fileB),
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Comparison failed');
  }

  return response.json();
};

export const getVisualizationUrl = (path) => {
  return `${API_BASE_URL}${path}`;
};