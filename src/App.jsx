import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import UploadData from './pages/UploadData';
import TrainModels from './pages/TrainModels';
import Prediction from './pages/Prediction';
import Comparison from './pages/Comparison';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<UploadData />} />
          <Route path="/train" element={<TrainModels />} />
          <Route path="/predict" element={<Prediction />} />
          <Route path="/comparison" element={<Comparison />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
