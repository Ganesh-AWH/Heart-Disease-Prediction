const API_BASE = 'http://127.0.0.1:5000';  // ✅ Correct Flask backend port




export const api = {
  async uploadData(data) {
    const response = await fetch(`${API_BASE}/api/data/upload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data }),
    });
    return response.json();
  },

  async fetchData() {
    const response = await fetch(`${API_BASE}/api/data/fetch`);
    return response.json();
  },

  async getStatistics() {
    const response = await fetch(`${API_BASE}/api/dashboard`);
    if (!response.ok) {
      throw new Error('Failed to fetch dashboard statistics');
    }
    return response.json();
  },


  async trainSupervised() {
    const response = await fetch(`${API_BASE}/api/supervised/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    return response.json();
  },

  async trainUnsupervised(nClusters = 2) {
    const response = await fetch(`${API_BASE}/api/unsupervised/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n_clusters: nClusters }),
    });
    return response.json();
  },

  async predictSupervised(modelName, sample) {
    const response = await fetch(`${API_BASE}/api/supervised/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName, sample }),
    });
    return response.json();
  },

  async predictUnsupervised(modelName, sample) {
    const response = await fetch(`${API_BASE}/api/unsupervised/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName, sample }),
    });
    return response.json();
  },

  async getComparisonReport() {
    const response = await fetch(`${API_BASE}/api/comparison/report`);
    return response.json();
  },

  async getRankings(type = 'supervised') {
    const response = await fetch(`${API_BASE}/api/comparison/rankings?type=${type}`);
    return response.json();
  },

  async getFeatureImportance(modelName = 'random_forest') {
    const response = await fetch(`${API_BASE}/api/supervised/feature-importance?model=${modelName}`);
    return response.json();
  },

  async getTrainingHistory(type) {
    const url = type ? `${API_BASE}/api/training-history?type=${type}` : `${API_BASE}/api/training-history`;
    const response = await fetch(url);
    return response.json();
  },

  async getPredictionHistory(limit = 100) {
    const response = await fetch(`${API_BASE}/api/predictions/history?limit=${limit}`);
    return response.json();
  },

  async clearAllData() {
    const response = await fetch(`${API_BASE}/api/data/clear`, {
      method: 'DELETE',  // ✅ must be DELETE, not GET
    });
    return response.json();
  },


};
