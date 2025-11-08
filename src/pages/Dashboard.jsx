import { useState, useEffect } from 'react';
import { api } from '../lib/api';
import { Link } from 'react-router-dom';

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const data = await api.getStatistics();
      setStats(data);
    } catch (error) {
      console.error('Error loading statistics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="container">
        <div className="loading">
          <div className="spinner"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">
          Comparative Analysis of Supervised and Unsupervised Learning for Heart Disease Prediction
        </p>
      </div>

      <div className="card-grid">
        <div className="stat-card">
          <div className="stat-label">Total Records</div>
          <div className="stat-value">{stats?.total_records || 0}</div>
        </div>
        <div className="stat-card success">
          <div className="stat-label">Disease Negative</div>
          <div className="stat-value">{stats?.disease_negative || 0}</div>
        </div>
        <div className="stat-card danger">
          <div className="stat-label">Disease Positive</div>
          <div className="stat-value">{stats?.disease_positive || 0}</div>
        </div>
        <div className="stat-card warning">
          <div className="stat-label">Total Predictions</div>
          <div className="stat-value">{stats?.total_predictions || 0}</div>
        </div>
      </div>

      <div className="card">
        <h2 className="card-title">About This Project</h2>
        <p style={{ marginBottom: '1rem' }}>
          This application demonstrates a comparative analysis of supervised and unsupervised machine learning
          algorithms for heart disease prediction. The system allows you to:
        </p>
        <ul style={{ marginLeft: '1.5rem', marginBottom: '1rem' }}>
          <li>Upload and manage heart disease datasets</li>
          <li>Train multiple supervised learning models (Decision Tree, Random Forest, SVM, Logistic Regression, KNN)</li>
          <li>Train unsupervised clustering models (K-Means, DBSCAN, Hierarchical, Fuzzy C-Means, GMM, Autoencoders)</li>
          <li>Make predictions on new patient data</li>
          <li>Compare model performance and view detailed analytics</li>
        </ul>
      </div>

      <div className="grid-2">
        <div className="card">
          <h2 className="card-title">Supervised Learning</h2>
          <p style={{ marginBottom: '1rem', color: 'var(--gray-600)' }}>
            Train models for binary classification (disease vs no disease) when labeled data is available.
          </p>
          <Link to="/train">
            <button className="btn btn-primary">Train Supervised Models</button>
          </Link>
        </div>

        <div className="card">
          <h2 className="card-title">Unsupervised Learning</h2>
          <p style={{ marginBottom: '1rem', color: 'var(--gray-600)' }}>
            Discover patterns and group patients into clusters for exploratory analysis.
          </p>
          <Link to="/train">
            <button className="btn btn-secondary">Train Unsupervised Models</button>
          </Link>
        </div>
      </div>

      <div className="card" style={{ marginTop: '2rem', textAlign: 'center' }}>
        <h2 className="card-title">Data Management</h2>
        <p style={{ color: 'var(--gray-600)', marginBottom: '1rem' }}>
          You can clear all uploaded data, predictions, and training records.
        </p>
        <button
          className="btn btn-danger"
          onClick={async () => {
            if (window.confirm('⚠️ Are you sure you want to delete all data? This action cannot be undone.')) {
              try {
                const res = await api.clearAllData();
                alert(res.message || 'All data cleared successfully!');
                setStats(null);
                setTimeout(() => loadStats(), 2000); // reload dashboard stats
              } catch (err) {
                console.error(err);
                alert('Failed to clear data.');
              }
            }
          }}
        >
          🧹 Clear All Data
        </button>
      </div>


      {stats && stats.total_records === 0 && (
        <div className="alert alert-error">
          <strong>No data available.</strong> Please upload your heart disease dataset to get started.
          <Link to="/upload" style={{ marginLeft: '1rem', textDecoration: 'underline' }}>
            Upload Data
          </Link>
        </div>
      )}
    </div>
  );
}
