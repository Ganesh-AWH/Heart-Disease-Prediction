import { useState } from 'react';
import { api } from '../lib/api';
import FeatureImportanceChart from '../components/FeatureImportanceChart';


const INITIAL_FORM = {
  age: 50,
  sex: 1,
  cp: 0,
  trestbps: 120,
  chol: 200,
  fbs: 0,
  restecg: 0,
  thalach: 150,
  exang: 0,
  oldpeak: 0,
  slope: 0,
  ca: 0,
  thal: 0,
};

export default function Prediction() {
  const [modelType, setModelType] = useState('supervised');
  const [modelName, setModelName] = useState('random_forest');
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const supervisedModels = [
    { value: 'decision_tree', label: 'Decision Tree' },
    { value: 'random_forest', label: 'Random Forest' },
    { value: 'svm', label: 'Support Vector Machine' },
    { value: 'logistic_regression', label: 'Logistic Regression' },
    { value: 'knn', label: 'K-Nearest Neighbors' },
  ];

  const unsupervisedModels = [
    { value: 'kmeans', label: 'K-Means' },
    { value: 'dbscan', label: 'DBSCAN' },
    { value: 'hierarchical', label: 'Hierarchical' },
    { value: 'fuzzy_cmeans', label: 'Fuzzy C-Means' },
    { value: 'gmm', label: 'Gaussian Mixture Model' },
    { value: 'autoencoder', label: 'Autoencoder' },
  ];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value) || 0
    }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let predictionResult;

      if (modelType === 'supervised') {
        predictionResult = await api.predictSupervised(modelName, formData);
      } else {
        predictionResult = await api.predictUnsupervised(modelName, formData);
      }

      if (predictionResult.error) {
        throw new Error(predictionResult.error);
      }

      setResult(predictionResult);
    } catch (err) {
      setError(err.message || 'Failed to make prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">Prediction</h1>
        <p className="page-subtitle">Make predictions on new patient data</p>
      </div>

      {error && (
        <div className="alert alert-error">{error}</div>
      )}

      <div className="grid-2">
        <div className="card">
          <h2 className="card-title">Model Selection</h2>

          <div className="input-group">
            <label className="input-label">Model Type</label>
            <select
              value={modelType}
              onChange={(e) => {
                setModelType(e.target.value);
                setModelName(e.target.value === 'supervised' ? 'random_forest' : 'kmeans');
                setResult(null);
              }}
              className="select-field"
            >
              <option value="supervised">Supervised Learning</option>
              <option value="unsupervised">Unsupervised Learning</option>
            </select>
          </div>

          <div className="input-group">
            <label className="input-label">Select Model</label>
            <select
              value={modelName}
              onChange={(e) => {
                setModelName(e.target.value);
                setResult(null);
              }}
              className="select-field"
            >
              {(modelType === 'supervised' ? supervisedModels : unsupervisedModels).map(model => (
                <option key={model.value} value={model.value}>
                  {model.label}
                </option>
              ))}
            </select>
          </div>

          {result && (
            <div style={{ marginTop: '1.5rem' }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: 600, marginBottom: '1rem' }}>
                Prediction Result
              </h3>

              {modelType === 'supervised' ? (
                <div>
                  <div className={`badge ${result.prediction.prediction === 1 ? 'badge-danger' : 'badge-success'}`}
                       style={{ fontSize: '1rem', padding: '0.5rem 1rem', marginBottom: '1rem' }}>
                    {result.prediction.risk_level}
                  </div>
                  <div style={{ marginTop: '1rem' }}>
                    <strong>Prediction:</strong> {result.prediction.prediction === 1 ? 'Disease' : 'No Disease'}
                  </div>
                  {result.prediction.probability && (
                    <div style={{ marginTop: '0.5rem' }}>
                      <strong>Confidence:</strong>
                      <div style={{ marginTop: '0.5rem' }}>
                        No Disease: {(result.prediction.probability[0] * 100).toFixed(2)}%
                      </div>
                      <div>
                        Disease: {(result.prediction.probability[1] * 100).toFixed(2)}%
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div>
                  <div className="badge badge-primary" style={{ fontSize: '1rem', padding: '0.5rem 1rem' }}>
                    {result.cluster_label}
                  </div>
                  <div style={{ marginTop: '1rem' }}>
                    <strong>Cluster Assignment:</strong> {result.cluster}
                  </div>
                </div>
              )}
            </div>
          )}
          {modelType === 'supervised' && (
            <FeatureImportanceChart modelName={modelName} />
          )}
        </div>


        <div className="card">
          <h2 className="card-title">Patient Data</h2>

          <form onSubmit={handlePredict}>
            <div className="grid-2">
              <div className="input-group">
                <label className="input-label">Age</label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>

              <div className="input-group">
                <label className="input-label">Sex (0=F, 1=M)</label>
                <select
                  name="sex"
                  value={formData.sex}
                  onChange={handleChange}
                  className="select-field"
                >
                  <option value={0}>Female</option>
                  <option value={1}>Male</option>
                </select>
              </div>

              <div className="input-group">
                <label className="input-label">Chest Pain Type (0-3)</label>
                <input
                  type="number"
                  name="cp"
                  min="0"
                  max="3"
                  value={formData.cp}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>

              <div className="input-group">
                <label className="input-label">Resting Blood Pressure</label>
                <input
                  type="number"
                  name="trestbps"
                  value={formData.trestbps}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>

              <div className="input-group">
                <label className="input-label">Cholesterol (mg/dl)</label>
                <input
                  type="number"
                  name="chol"
                  value={formData.chol}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>

              <div className="input-group">
                <label className="input-label">Fasting Blood Sugar {'>'} 120</label>
                <select
                  name="fbs"
                  value={formData.fbs}
                  onChange={handleChange}
                  className="select-field"
                >
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div className="input-group">
                <label className="input-label">Resting ECG (0-2)</label>
                <input
                  type="number"
                  name="restecg"
                  min="0"
                  max="2"
                  value={formData.restecg}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>

              <div className="input-group">
                <label className="input-label">Max Heart Rate</label>
                <input
                  type="number"
                  name="thalach"
                  value={formData.thalach}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>

              <div className="input-group">
                <label className="input-label">Exercise Angina</label>
                <select
                  name="exang"
                  value={formData.exang}
                  onChange={handleChange}
                  className="select-field"
                >
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div className="input-group">
                <label className="input-label">ST Depression</label>
                <input
                  type="number"
                  step="0.1"
                  name="oldpeak"
                  value={formData.oldpeak}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>

              <div className="input-group">
                <label className="input-label">Slope (0-2)</label>
                <input
                  type="number"
                  name="slope"
                  min="0"
                  max="2"
                  value={formData.slope}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>

              <div className="input-group">
                <label className="input-label">Major Vessels (0-4)</label>
                <input
                  type="number"
                  name="ca"
                  min="0"
                  max="4"
                  value={formData.ca}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>

              <div className="input-group">
                <label className="input-label">Thalassemia (0-3)</label>
                <input
                  type="number"
                  name="thal"
                  min="0"
                  max="3"
                  value={formData.thal}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary"
              style={{ marginTop: '1rem' }}
            >
              {loading ? 'Predicting...' : 'Make Prediction'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
