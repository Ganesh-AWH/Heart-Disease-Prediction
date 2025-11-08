import { useState } from 'react';
import { api } from '../lib/api';

export default function UploadData() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'text/csv') {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please select a valid CSV file');
      setFile(null);
    }
  };

  const parseCSV = (text) => {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());

    const data = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      if (values.length === headers.length) {
        const row = {};
        headers.forEach((header, index) => {
          const value = values[index].trim();
          row[header] = isNaN(value) ? value : Number(value);
        });
        data.push(row);
      }
    }
    return data;
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setMessage(null);
    setError(null);

    try {
      const text = await file.text();
      const parsedData = parseCSV(text);

      if (parsedData.length === 0) {
        throw new Error('No data found in CSV file');
      }

      const result = await api.uploadData(parsedData);

      if (result.error) {
        throw new Error(result.error);
      }

      setMessage(`Successfully uploaded ${result.records_inserted} records!`);
      setFile(null);
      document.getElementById('file-input').value = '';
    } catch (err) {
      setError(err.message || 'Failed to upload data');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="page-header">
        <h1 className="page-title">Upload Dataset</h1>
        <p className="page-subtitle">Upload your heart disease dataset in CSV format</p>
      </div>

      <div className="card">
        <h2 className="card-title">Dataset Requirements</h2>
        <p style={{ marginBottom: '1rem', color: 'var(--gray-600)' }}>
          Your CSV file should contain the following columns:
        </p>
        <div className="grid-3" style={{ marginBottom: '1.5rem' }}>
          <div>
            <strong>age</strong> - Age in years
          </div>
          <div>
            <strong>sex</strong> - Sex (0=female, 1=male)
          </div>
          <div>
            <strong>cp</strong> - Chest pain type (0-3)
          </div>
          <div>
            <strong>trestbps</strong> - Resting blood pressure
          </div>
          <div>
            <strong>chol</strong> - Serum cholesterol
          </div>
          <div>
            <strong>fbs</strong> - Fasting blood sugar
          </div>
          <div>
            <strong>restecg</strong> - Resting ECG (0-2)
          </div>
          <div>
            <strong>thalach</strong> - Max heart rate
          </div>
          <div>
            <strong>exang</strong> - Exercise angina (0-1)
          </div>
          <div>
            <strong>oldpeak</strong> - ST depression
          </div>
          <div>
            <strong>slope</strong> - Slope (0-2)
          </div>
          <div>
            <strong>ca</strong> - Major vessels (0-4)
          </div>
          <div>
            <strong>thal</strong> - Thalassemia (0-3)
          </div>
          <div>
            <strong>target</strong> - Disease (0=no, 1=yes)
          </div>
        </div>
      </div>

      <div className="card">
        <h2 className="card-title">Upload File</h2>

        {message && (
          <div className="alert alert-success">{message}</div>
        )}

        {error && (
          <div className="alert alert-error">{error}</div>
        )}

        <div className="input-group">
          <label className="input-label">Select CSV File</label>
          <input
            id="file-input"
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="input-field"
          />
        </div>

        {file && (
          <div style={{ marginBottom: '1rem', color: 'var(--gray-600)' }}>
            Selected file: <strong>{file.name}</strong> ({(file.size / 1024).toFixed(2)} KB)
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="btn btn-primary"
        >
          {loading ? 'Uploading...' : 'Upload Dataset'}
        </button>
      </div>

      <div className="card">
        <h2 className="card-title">Sample CSV Format</h2>
        <pre style={{
          background: 'var(--gray-100)',
          padding: '1rem',
          borderRadius: '0.5rem',
          overflow: 'auto',
          fontSize: '0.875rem'
        }}>
{`age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2,1
41,0,1,130,204,0,0,172,0,1.4,2,0,2,1`}
        </pre>
      </div>
    </div>
  );
}
