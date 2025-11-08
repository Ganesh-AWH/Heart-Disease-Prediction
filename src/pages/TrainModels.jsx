import { useState } from 'react';
import { api } from '../lib/api';

export default function TrainModels() {
  const [supervisedLoading, setSupervisedLoading] = useState(false);
  const [unsupervisedLoading, setUnsupervisedLoading] = useState(false);
  const [supervisedResults, setSupervisedResults] = useState(null);
  const [unsupervisedResults, setUnsupervisedResults] = useState(null);
  const [nClusters, setNClusters] = useState(2);
  const [error, setError] = useState(null);

  const handleTrainSupervised = async () => {
    setSupervisedLoading(true);
    setError(null);

    try {
      const result = await api.trainSupervised();

      if (result.error) {
        throw new Error(result.error);
      }

      setSupervisedResults(result);
    } catch (err) {
      setError(err.message || 'Failed to train supervised models');
    } finally {
      setSupervisedLoading(false);
    }
  };

  const handleTrainUnsupervised = async () => {
    setUnsupervisedLoading(true);
    setError(null);

    try {
      const result = await api.trainUnsupervised(nClusters);

      if (result.error) {
        throw new Error(result.error);
      }

      setUnsupervisedResults(result);
    } catch (err) {
      setError(err.message || 'Failed to train unsupervised models');
    } finally {
      setUnsupervisedLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">Train Models</h1>
        <p className="page-subtitle">Train supervised and unsupervised learning models</p>
      </div>

      {error && (
        <div className="alert alert-error">{error}</div>
      )}

      <div className="grid-2">
        <div className="card">
          <h2 className="card-title">Supervised Learning</h2>
          <p style={{ marginBottom: '1rem', color: 'var(--gray-600)' }}>
            Train models for binary classification using labeled data
          </p>

          <div style={{ marginBottom: '1rem' }}>
            <strong>Models:</strong>
            <ul style={{ marginTop: '0.5rem', marginLeft: '1.5rem' }}>
              <li>Decision Tree</li>
              <li>Random Forest</li>
              <li>Support Vector Machine (SVM)</li>
              <li>Logistic Regression</li>
              <li>K-Nearest Neighbors (KNN)</li>
            </ul>
          </div>

          <button
            onClick={handleTrainSupervised}
            disabled={supervisedLoading}
            className="btn btn-primary"
          >
            {supervisedLoading ? 'Training...' : 'Train Supervised Models'}
          </button>

          {supervisedResults && (
            <div style={{ marginTop: '1.5rem' }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: 600, marginBottom: '1rem' }}>
                Training Results
              </h3>
              <div className="table-container">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Accuracy</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1-Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(supervisedResults.comparison).map(([model, metrics]) => (
                      <tr key={model}>
                        <td style={{ fontWeight: 600 }}>{model}</td>
                        <td>{(metrics.accuracy * 100).toFixed(2)}%</td>
                        <td>{(metrics.precision * 100).toFixed(2)}%</td>
                        <td>{(metrics.recall * 100).toFixed(2)}%</td>
                        <td>{(metrics.f1_score * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="alert alert-success" style={{ marginTop: '1rem' }}>
                <strong>Best Model:</strong> {supervisedResults.best_model.name}
                ({(supervisedResults.best_model.accuracy * 100).toFixed(2)}% accuracy)
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <h2 className="card-title">Unsupervised Learning</h2>
          <p style={{ marginBottom: '1rem', color: 'var(--gray-600)' }}>
            Discover patterns and cluster patients without labels
          </p>

          <div style={{ marginBottom: '1rem' }}>
            <strong>Models:</strong>
            <ul style={{ marginTop: '0.5rem', marginLeft: '1.5rem' }}>
              <li>K-Means Clustering</li>
              <li>DBSCAN</li>
              <li>Hierarchical Clustering</li>
              <li>Fuzzy C-Means</li>
              <li>Gaussian Mixture Models</li>
              <li>Autoencoders</li>
            </ul>
          </div>

          <div className="input-group">
            <label className="input-label">Number of Clusters</label>
            <input
              type="number"
              min="2"
              max="10"
              value={nClusters}
              onChange={(e) => setNClusters(Number(e.target.value))}
              className="input-field"
            />
          </div>

          <button
            onClick={handleTrainUnsupervised}
            disabled={unsupervisedLoading}
            className="btn btn-secondary"
          >
            {unsupervisedLoading ? 'Training...' : 'Train Unsupervised Models'}
          </button>

          {unsupervisedResults && (
            <div style={{ marginTop: '1.5rem' }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: 600, marginBottom: '1rem' }}>
                Training Results
              </h3>
              <div className="table-container">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Silhouette</th>
                      <th>Davies-Bouldin</th>
                      <th>Clusters</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(unsupervisedResults.comparison).map(([model, metrics]) => (
                      <tr key={model}>
                        <td style={{ fontWeight: 600 }}>{model}</td>
                        <td>{metrics.silhouette_score.toFixed(4)}</td>
                        <td>{metrics.davies_bouldin_score.toFixed(4)}</td>
                        <td>{metrics.n_clusters}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="alert alert-success" style={{ marginTop: '1rem' }}>
                <strong>Best Model:</strong> {unsupervisedResults.best_model.name}
                (Silhouette: {unsupervisedResults.best_model.silhouette_score.toFixed(4)})
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
