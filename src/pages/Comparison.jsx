import FeatureImportanceChart from '../components/FeatureImportanceChart';
import { useState, useEffect } from 'react';
import { api } from '../lib/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

export default function Comparison() {
  // ✅ Hooks declared once at the top
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [bestModel, setBestModel] = useState('random_forest');

  // ✅ Fetch report + detect best model (single useEffect)
  useEffect(() => {
    async function loadReport() {
      try {
        const data = await api.getComparisonReport();
        if (data.error) throw new Error(data.error);

        setReport(data);

        // Automatically set best model if available
        let best = 'random_forest';
        if (data?.supervised_learning?.best_accuracy) {
          // Example: "RandomForest (Accuracy: 0.93)"
          best = data.supervised_learning.best_accuracy
            .split(' ')[0]
            .toLowerCase();
        } else if (data?.supervised?.ranking?.[0]?.model_name) {
          best = data.supervised.ranking[0].model_name;
        }
        setBestModel(best);
      } catch (err) {
        setError(err.message || 'Failed to load comparison report');
      } finally {
        setLoading(false);
      }
    }

    loadReport();
  }, []);

  // ✅ Prepare chart data safely
  const supervisedChartData =
    report?.supervised_learning?.comparison_table?.map((item) => ({
      name: item.Model,
      Accuracy: (item.Accuracy * 100).toFixed(2),
      'F1-Score': (item['F1-Score'] * 100).toFixed(2),
    })) || [];

  const unsupervisedChartData =
    report?.unsupervised_learning?.comparison_table?.map((item) => ({
      name: item.Model,
      'Silhouette Score': item['Silhouette Score'].toFixed(4),
    })) || [];

  // ✅ Loading state
  if (loading)
    return (
      <div className="container">
        <div className="loading">
          <div className="spinner"></div>
        </div>
      </div>
    );

  // ✅ Error state
  if (error)
    return (
      <div className="container">
        <div className="alert alert-error">{error}</div>
      </div>
    );

  // ✅ Main render
  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">Model Comparison</h1>
        <p className="page-subtitle">
          Comprehensive analysis and performance comparison
        </p>
      </div>

      {/* Grid: Supervised + Unsupervised */}
      <div
        className="grid-2"
        style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}
      >
        {report?.supervised_learning && (
          <div className="card">
            <h2 className="card-title">Supervised Learning Performance</h2>
            <table className="table" style={{ marginBottom: '1rem' }}>
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
                {report.supervised_learning.comparison_table.map((row) => (
                  <tr key={row.Model}>
                    <td style={{ fontWeight: 600 }}>{row.Model}</td>
                    <td>{(row.Accuracy * 100).toFixed(2)}%</td>
                    <td>{(row.Precision * 100).toFixed(2)}%</td>
                    <td>{(row.Recall * 100).toFixed(2)}%</td>
                    <td>{(row['F1-Score'] * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={supervisedChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="Accuracy" fill="#2563eb" />
                <Bar dataKey="F1-Score" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>

            <div
              style={{
                marginTop: '1rem',
                padding: '1rem',
                background: 'var(--gray-100)',
                borderRadius: '0.5rem',
              }}
            >
              <strong>Summary:</strong>
              <div>
                Average Accuracy:{' '}
                {(report.supervised_learning.summary.avg_accuracy * 100).toFixed(2)}%
              </div>
              <div>
                Average F1-Score:{' '}
                {(report.supervised_learning.summary.avg_f1 * 100).toFixed(2)}%
              </div>
              <div
                style={{
                  marginTop: '0.5rem',
                  color: 'var(--primary)',
                }}
              >
                <strong>Best Model:</strong> {report.supervised_learning.best_accuracy}
              </div>
            </div>

            {/* ✅ Feature Importance Chart */}
            <div className="card" style={{ marginTop: '2rem' }}>
              <h2 className="card-title">
                Feature Importance ({"random_forest"})
              </h2>
              <p style={{ marginBottom: '1rem', color: 'var(--gray-600)' }}>
                This chart highlights which features most influenced predictions made
                by the best-performing supervised model.
              </p>
              {/* <FeatureImportanceChart modelName={"random_forest"} /> */}
              {/* 🧠 Feature Importance for Any Model */}
                <div className="card" style={{ marginTop: '2rem' }}>
                  <h2 className="card-title">Feature Importance</h2>
                  <p style={{ marginBottom: '1rem', color: 'var(--gray-600)' }}>
                    Select a supervised model below to view its feature importance ranking.
                  </p>

                  {/* Dropdown for model selection */}
                  <select
                    value={bestModel}
                    onChange={(e) => setBestModel(e.target.value)}
                    className="select-field"
                    style={{ width: '250px', marginBottom: '1rem' }}
                  >
                    <option value="random_forest">Random Forest</option>
                    <option value="decision_tree">Decision Tree</option>
                    <option value="logistic_regression">Logistic Regression</option>
                    <option value="svm">Support Vector Machine</option>
                    <option value="knn">K-Nearest Neighbors</option>
                  </select>

                  {/* Chart for selected model */}
                  <FeatureImportanceChart modelName={bestModel} />
                </div>

            </div>
          </div>
        )}

        {/* ✅ Unsupervised Section */}
        {report?.unsupervised_learning && (
          <div className="card">
            <h2 className="card-title">Unsupervised Learning Performance</h2>
            <table className="table" style={{ marginBottom: '1rem' }}>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Silhouette Score</th>
                  <th>Davies-Bouldin</th>
                  <th>Calinski-Harabasz</th>
                  <th>Clusters</th>
                </tr>
              </thead>
              <tbody>
                {report.unsupervised_learning.comparison_table.map((row) => (
                  <tr key={row.Model}>
                    <td style={{ fontWeight: 600 }}>{row.Model}</td>
                    <td>{row['Silhouette Score'].toFixed(4)}</td>
                    <td>{row['Davies-Bouldin Index'].toFixed(4)}</td>
                    <td>{row['Calinski-Harabasz Score'].toFixed(2)}</td>
                    <td>{row['N Clusters']}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={unsupervisedChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="Silhouette Score" fill="#f59e0b" />
              </BarChart>
            </ResponsiveContainer>

            <div
              style={{
                marginTop: '1rem',
                padding: '1rem',
                background: 'var(--gray-100)',
                borderRadius: '0.5rem',
              }}
            >
              <strong>Summary:</strong>
              <div>
                Average Silhouette Score:{' '}
                {report.unsupervised_learning.summary.avg_silhouette.toFixed(4)}
              </div>
              <div
                style={{
                  marginTop: '0.5rem',
                  color: 'var(--secondary)',
                }}
              >
                <strong>Best Model:</strong>{' '}
                {report.unsupervised_learning.best_silhouette}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ✅ Recommendations Section */}
      {report?.recommendations && (
        <div className="card" style={{ marginTop: '2rem' }}>
          <h2 className="card-title">Recommendations</h2>
          <div
            className="grid-2"
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '1.5rem',
            }}
          >
            <div>
              <h3
                style={{ color: 'var(--primary)', fontWeight: 600 }}
              >
                When to Use Supervised Learning
              </h3>
              <ul style={{ marginLeft: '1.5rem' }}>
                {report.recommendations.use_cases.supervised.map((u, i) => (
                  <li key={i}>{u}</li>
                ))}
              </ul>
            </div>
            <div>
              <h3
                style={{ color: 'var(--secondary)', fontWeight: 600 }}
              >
                When to Use Unsupervised Learning
              </h3>
              <ul style={{ marginLeft: '1.5rem' }}>
                {report.recommendations.use_cases.unsupervised.map((u, i) => (
                  <li key={i}>{u}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
