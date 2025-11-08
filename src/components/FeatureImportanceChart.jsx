import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  Title,
} from 'chart.js';

// ✅ Register required Chart.js modules
ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend, Title);

export default function FeatureImportanceChart({ modelName = 'random_forest' }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    async function fetchFeatureImportance() {
      try {
        const res = await api.getFeatureImportance(modelName);
        if (res.feature_importance) {
          setData(res.feature_importance);
        } else {
          setData(null);
        }
      } catch (err) {
        console.error('Error fetching feature importance:', err);
        setData(null);
      }
    }
    fetchFeatureImportance();
  }, [modelName]);

  if (!data) return <p style={{ color: '#777' }}>No feature importance data available.</p>;

  // ✅ Sort by importance descending
  const sorted = [...data].sort((a, b) => b.importance - a.importance);

  // ✅ Highlight top 3
  const colors = sorted.map((_, i) => {
    if (i === 0) return 'rgba(255, 99, 132, 0.8)';     // 🔴 Most important
    if (i === 1) return 'rgba(255, 159, 64, 0.8)';     // 🟠 2nd
    if (i === 2) return 'rgba(255, 205, 86, 0.8)';     // 🟡 3rd
    return 'rgba(54, 162, 235, 0.7)';                  // 🔵 Others
  });

  const chartData = {
    labels: sorted.map(d => d.feature),
    datasets: [
      {
        label: 'Feature Importance',
        data: sorted.map(d => d.importance),
        backgroundColor: colors,
        borderColor: 'rgba(0, 0, 0, 0.1)',
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: {
        display: false,
        text: `Feature Importance (${modelName})`,
        font: { size: 18, weight: 'bold' },
        color: '#222',
      },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.formattedValue} importance`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: '#444', font: { size: 13 } },
      },
      y: {
        beginAtZero: true,
        ticks: { color: '#555', font: { size: 12 } },
        title: { display: true, text: 'Importance Score', color: '#333' },
      },
    },
    animation: {
      duration: 900,
      easing: 'easeOutBounce',
    },
  };

  return (
    <div className="card" style={{ marginTop: '2rem', padding: '1rem' }}>
      <h2 className="card-title" style={{ marginBottom: '1rem' }}>
        Feature Importance ({modelName})
      </h2>
      <Bar data={chartData} options={chartOptions} />
      <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.8rem' }}>
        🔴 Top 3 most influential features are highlighted above.
      </p>
    </div>
  );
}
